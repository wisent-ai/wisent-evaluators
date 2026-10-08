//! `wisent-evaluators surface [root] [--tolerant]` prints the evaluator
//! names the Python package registers; `wisent-evaluators baseline [--best |
//! --cross-check VERSION]` rewrites `released-surface.json` from PyPI, prints
//! only what PyPI serves now, or compares the sdist and wheel readers. Exit 1
//! is a refusal, exit 2 an invocation the command does not take.
//!
//! The evaluators themselves are the library beside this entry
//! (`wisent_evaluators`), which the Python evaluators are moving into.

mod release;

use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str =
    "usage: wisent-evaluators surface [root] [--tolerant] | baseline [--best | --cross-check VERSION]";

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let outcome = match arguments.split_first() {
        Some((command, rest)) if command == "surface" => {
            let root = rest
                .iter()
                .find(|argument| !argument.starts_with('-'))
                .map(PathBuf::from);
            release::print_surface(
                root.as_deref(),
                rest.iter().any(|argument| argument == "--tolerant"),
            )
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
