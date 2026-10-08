//! The release contract of the evaluator names the Python package still
//! publishes while its evaluators move to this program: `surface [ROOT]
//! [--tolerant]` prints the names the package registers; `baseline [--best |
//! --cross-check VERSION]` rewrites `released-surface.json` from PyPI, prints
//! only what PyPI serves now, or compares the sdist and wheel readers.

mod baseline;
mod surface;

use std::path::{Path, PathBuf};

/// The repository this program was built from.
fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn print_surface(root: Option<&Path>, tolerant: bool) -> Result<(), String> {
    let root = match root {
        Some(root) => root.to_path_buf(),
        None => repository(),
    };
    let (names, skipped) = surface::surface(&root, tolerant)?;
    let mut document = serde_json::Map::new();
    document.insert("surface".to_string(), serde_json::json!(names));
    if !skipped.is_empty() {
        document.insert("unparseable".to_string(), serde_json::json!(skipped));
    }
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
