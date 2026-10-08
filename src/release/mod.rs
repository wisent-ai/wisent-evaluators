//! The release contract of the evaluator names: `surface` prints the names
//! this program's registry offers, the candidate a release is judged by;
//! `baseline [--best | --cross-check VERSION]` rewrites
//! `released-surface.json` from the release PyPI serves, prints only what
//! PyPI serves now, or compares the sdist and wheel readers. The released
//! artifact is the Python package, so its names are read from its source
//! with a parser.

mod baseline;
mod surface;

use std::path::PathBuf;

/// The repository this program was built from.
fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn print_surface() -> Result<(), String> {
    let names: Vec<&str> = wisent_evaluators::registered()
        .iter()
        .map(|evaluator| evaluator.name())
        .collect();
    let document = serde_json::json!({ "surface": names });
    println!(
        "{}",
        serde_json::to_string_pretty(&document).map_err(|error| error.to_string())?
    );
    Ok(())
}

pub fn baseline(flags: &[String]) -> Result<(), String> {
    let repository = repository();
    let scratch = repository.join("target").join("baseline-artifact");
    baseline::run(&repository, &scratch, flags)
}
