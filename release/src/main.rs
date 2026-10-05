//! `wisent-evaluators-release surface [root] [--tolerant]` prints the evaluator
//! names the package registers; `wisent-evaluators-release baseline [--best |
//! --cross-check VERSION]` rewrites `released-surface.json` from PyPI, prints
//! only what PyPI serves now, or compares the sdist and wheel readers. Exit 1
//! is a refusal, exit 2 an invocation the command does not take.

mod baseline;
mod surface;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str =
    "usage: wisent-evaluators-release surface [root] [--tolerant] | baseline [--best | --cross-check VERSION]";

fn print_surface(root: &Path, tolerant: bool) -> Result<(), String> {
    let (names, skipped) = surface::surface(root, tolerant)?;
    let mut document = serde_json::Map::new();
    document.insert("surface".to_string(), serde_json::json!(names));
    if !skipped.is_empty() {
        document.insert("unparseable".to_string(), serde_json::json!(skipped));
    }
    println!("{}", serde_json::to_string_pretty(&document).map_err(|error| error.to_string())?);
    Ok(())
}

fn main() -> ExitCode {
    let release = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repository = release.parent().unwrap_or(release).to_path_buf();
    let scratch = release.join("target").join("baseline-artifact");
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let outcome = match arguments.split_first() {
        Some((command, rest)) if command == "surface" => {
            let root = rest
                .iter()
                .find(|argument| !argument.starts_with('-'))
                .map(PathBuf::from)
                .unwrap_or_else(|| repository.clone());
            print_surface(&root, rest.iter().any(|argument| argument == "--tolerant"))
        }
        Some((command, rest)) if command == "baseline" => baseline::run(&repository, &scratch, rest),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(refusal) => {
            eprintln!("{refusal}");
            ExitCode::FAILURE
        }
    }
}
