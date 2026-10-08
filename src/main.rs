//! `wisent-evaluators list` prints every evaluator with what it measures;
//! `show NAME` prints one evaluator's options; `evaluate` reads requests as
//! JSON lines on standard input and writes one evaluation per line on
//! standard output, stopping at the first request it refuses.
//!
//! `surface` prints the evaluator names a release offers and `baseline
//! [--best | --cross-check VERSION]` reads the released contract. Exit 1 is a
//! refusal, exit 2 an invocation the program does not take.

mod release;

use std::io::{BufRead, Write};
use std::process::ExitCode;

use wisent_evaluators::{named, registered, Request};

const USAGE: &str = "usage: wisent-evaluators list | show NAME | evaluate < requests.jsonl | surface | baseline [--best | --cross-check VERSION]";

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let outcome = match arguments.split_first() {
        Some((command, rest)) if command == "list" && rest.is_empty() => list(),
        Some((command, [name])) if command == "show" => show(name),
        Some((command, rest)) if command == "evaluate" && rest.is_empty() => evaluate(),
        Some((command, rest)) if command == "surface" && rest.is_empty() => {
            release::print_surface()
        }
        Some((command, rest)) if command == "baseline" => release::baseline(rest),
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

fn list() -> Result<(), String> {
    for evaluator in registered() {
        println!("{}\t{}", evaluator.name(), evaluator.description());
    }
    Ok(())
}

fn show(name: &str) -> Result<(), String> {
    let evaluator = named(name).map_err(|refusal| refusal.to_string())?;
    println!("{}\n{}\n", evaluator.name(), evaluator.description());
    if evaluator.options().is_empty() {
        println!("options: none");
    }
    for (option, decides) in evaluator.options() {
        println!("options.{option}\t{decides}");
    }
    Ok(())
}

/// Each line of standard input is one request; each answer is written and
/// flushed before the next line is read, so a caller streaming requests
/// sees every evaluation as it is made.
fn evaluate() -> Result<(), String> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for (index, line) in stdin.lock().lines().enumerate() {
        let number = index + std::num::NonZeroUsize::MIN.get();
        let line =
            line.map_err(|error| format!("failed to read request line {number}: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let request: Request = serde_json::from_str(&line)
            .map_err(|error| format!("request line {number} is not a request: {error}"))?;
        let evaluation = named(&request.evaluator)
            .and_then(|evaluator| evaluator.evaluate(&request))
            .map_err(|refusal| format!("request line {number}: {refusal:#}"))?;
        let answer = serde_json::to_string(&evaluation)
            .map_err(|error| format!("failed to write evaluation {number}: {error}"))?;
        writeln!(stdout, "{answer}")
            .and_then(|()| stdout.flush())
            .map_err(|error| format!("failed to write evaluation {number}: {error}"))?;
    }
    Ok(())
}
