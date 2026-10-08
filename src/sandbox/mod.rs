//! Running a model's code against a benchmark's tests in a Docker container
//! that cannot reach the network, write its own image or gain privileges,
//! under the limits the caller states.
//!
//! The files travel as a tar archive on the container's standard input and
//! are unpacked into a memory-backed `/work`, so nothing is written on the
//! host. The image needs `sh`, `tar` and the interpreter the caller names.
//! A run ends when its program does or when it spends its processor time;
//! there is no wall-clock limit, so a program that waits without computing
//! holds its evaluation.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde_json::{Map, Value};

use crate::Options;

/// What `docker run` exits with when the daemon could not start the
/// container, when the command in it cannot be invoked, and when it is not
/// found.
// https://docs.docker.com/reference/cli/docker/container/run/#exit-status
const DOCKER_FAILED: i32 = 125;
// https://docs.docker.com/reference/cli/docker/container/run/#exit-status
const CANNOT_INVOKE: i32 = 126;
// https://docs.docker.com/reference/cli/docker/container/run/#exit-status
const NOT_FOUND: i32 = 127;

/// Owner read and write, everyone else read: what an unpacked file needs
/// when the container runs without the capability to override permissions.
// https://man7.org/linux/man-pages/man7/inode.7.html (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)
const READABLE: u32 = 0o644;

/// Printed by the container's shell only when the archive could not be
/// unpacked, so that failure is never read as the tests failing.
const UNPACK_FAILED: &str = "wisent-evaluators: the sandbox could not unpack the files";

/// Every option a sandboxed evaluator reads, with what it decides.
pub const OPTIONS: &[(&str, &str)] = &[
    ("image", "the Docker image the code runs in; required"),
    (
        "interpreter",
        "the program in the image that runs the test file (the image's Python); required",
    ),
    (
        "runtime",
        "a Docker runtime to run the container with, such as gVisor's runsc",
    ),
    (
        "cpu_seconds",
        "the processor time the run may spend before it is killed; required",
    ),
    ("memory_bytes", "the container's memory limit; required"),
    (
        "file_size_bytes",
        "the largest file the run may write; required",
    ),
    (
        "processes",
        "the most processes the container may hold; required",
    ),
    (
        "open_files",
        "the most files a process may hold open; required",
    ),
];

/// How one run ended and what it printed.
pub struct Outcome {
    /// Whether the test program exited successfully.
    pub passed: bool,
    /// The test program's exit status; `None` when a signal ended it.
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub elapsed: Duration,
}

impl Outcome {
    /// What the run produced, for an evaluation's meta.
    pub fn meta(&self) -> Map<String, Value> {
        let mut meta = Map::new();
        meta.insert("exit_code".into(), Value::from(self.exit_code));
        meta.insert(
            "elapsed_seconds".into(),
            Value::from(self.elapsed.as_secs_f64()),
        );
        meta.insert("stdout".into(), Value::from(self.stdout.as_str()));
        meta.insert("stderr".into(), Value::from(self.stderr.as_str()));
        meta
    }
}

/// Runs `program` with the image's interpreter in a fresh container holding
/// `files`, under the limits `options` states.
pub fn run(options: &Options<'_>, files: &[(&str, &str)], program: &str) -> Result<Outcome> {
    let image = options.text("image")?;
    let interpreter = options.text("interpreter")?;
    let runtime = options.opt_in_text("runtime")?;
    let cpu = options.count("cpu_seconds")?;
    let memory = options.count("memory_bytes")?;
    let file_size = options.count("file_size_bytes")?;
    let processes = options.count("processes")?;
    let open_files = options.count("open_files")?;
    let archive = archive(files)?;
    let mut command = Command::new("docker");
    command.arg("run");
    if let Some(runtime) = runtime {
        command.args(["--runtime", runtime]);
    }
    command
        .args(["-i", "--rm", "--name", &container_name()?])
        .args(["--network=none", "--read-only", "--cap-drop=ALL"])
        .arg("--security-opt=no-new-privileges")
        .arg(format!("--pids-limit={processes}"))
        .arg(format!("--memory={memory}"))
        .args(["--ulimit", &format!("cpu={cpu}:{cpu}")])
        .args(["--ulimit", &format!("fsize={file_size}:{file_size}")])
        .args(["--ulimit", &format!("nofile={open_files}:{open_files}")])
        .args(["--tmpfs", "/work:exec", "--tmpfs", "/tmp:exec"])
        .args(["--workdir", "/work", image, "sh", "-c"])
        .arg(format!(
            "tar -x -o -f - && exec \"$@\"; echo '{UNPACK_FAILED}'"
        ))
        .args(["sandbox", interpreter, program])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let started = Instant::now();
    let mut child = command
        .spawn()
        .context("failed to start docker; the sandboxed evaluators need the Docker CLI on PATH")?;
    let Some(mut stdin) = child.stdin.take() else {
        bail!("docker run gave no standard input to send the files through");
    };
    // The archive is sent from its own thread so a container that prints
    // before reading all of it cannot stall both sides.
    let sender = std::thread::spawn(move || stdin.write_all(&archive));
    let output = child
        .wait_with_output()
        .context("failed to wait for docker run")?;
    let elapsed = started.elapsed();
    match sender.join() {
        Ok(sent) => sent.context("failed to send the files to the container")?,
        Err(_) => bail!("the thread sending the files to the container panicked"),
    }
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if stdout.lines().any(|line| line == UNPACK_FAILED) {
        bail!(
            "the image {image} could not unpack the files with tar: {}",
            stderr.trim()
        );
    }
    match output.status.code() {
        Some(DOCKER_FAILED) => bail!("docker could not run the container: {}", stderr.trim()),
        Some(CANNOT_INVOKE | NOT_FOUND) => bail!(
            "the image {image} cannot run {interpreter}: {}",
            stderr.trim()
        ),
        exit_code => Ok(Outcome {
            passed: output.status.success(),
            exit_code,
            stdout,
            stderr,
            elapsed,
        }),
    }
}

fn archive(files: &[(&str, &str)]) -> Result<Vec<u8>> {
    let mut builder = tar::Builder::new(Vec::new());
    for (name, content) in files {
        let mut header = tar::Header::new_gnu();
        header.set_size(content.len() as u64);
        header.set_mode(READABLE);
        header.set_cksum();
        builder
            .append_data(&mut header, name, content.as_bytes())
            .with_context(|| format!("failed to pack {name} for the sandbox"))?;
    }
    builder
        .into_inner()
        .context("failed to finish the sandbox archive")
}

/// A container name no other run of this program shares.
fn container_name() -> Result<String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("the system clock reads before the Unix epoch")?;
    Ok(format!(
        "wisent-evaluators-{}-{}",
        std::process::id(),
        now.as_nanos()
    ))
}
