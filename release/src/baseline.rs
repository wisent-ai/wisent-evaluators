//! `released-surface.json` from the artifact PyPI actually serves: the version
//! is PyPI's, never the one setup.py declares, and the tier is the best that
//! exists (sdist, then a pure-Python wheel), never a lower one when a better
//! one exists. The first token of `source` is that tier's marker, which the
//! version-check workflow matches on.
//!
//! The newest release is wheel-only, so the wheel reader carries the baseline
//! alone; `--cross-check VERSION` points both readers at a version that
//! published an sdist and a wheel and requires the same names out of each.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::surface::surface;

const PROJECT: &str = "wisent-evaluators";
const SDIST_MARKER: &str = "pypi-sdist";
const WHEEL_MARKER: &str = "pypi-wheel";
const PURE_PYTHON_WHEEL: &str = "-py3-none-any.whl";

fn fetch(url: &str) -> Result<Vec<u8>, String> {
    let response = ureq::get(url).call().map_err(|error| format!("{url}: {error}"))?;
    let mut body = Vec::new();
    response.into_reader().read_to_end(&mut body).map_err(|error| format!("{url}: {error}"))?;
    Ok(body)
}

fn json_at(url: &str) -> Result<Value, String> {
    serde_json::from_slice(&fetch(url)?).map_err(|error| format!("{url} is not JSON: {error}"))
}

/// Every recoverable artifact of one published version, by marker.
fn candidates(version: &str) -> Result<Map<String, Value>, String> {
    let release = json_at(&format!("https://pypi.org/pypi/{PROJECT}/{version}/json"))?;
    let mut found = Map::new();
    for entry in release["urls"].as_array().into_iter().flatten() {
        let filename = entry["filename"].as_str().unwrap_or_default();
        if entry["packagetype"].as_str() == Some("sdist") {
            found.entry(SDIST_MARKER).or_insert_with(|| entry.clone());
        } else if filename.ends_with(PURE_PYTHON_WHEEL) {
            found.entry(WHEEL_MARKER).or_insert_with(|| entry.clone());
        }
    }
    Ok(found)
}

/// The best recoverable artifact of one version, as (marker, entry).
fn artifact(version: &str) -> Result<(&'static str, Value), String> {
    let found = candidates(version)?;
    for marker in [SDIST_MARKER, WHEEL_MARKER] {
        if let Some(entry) = found.get(marker) {
            return Ok((marker, entry.clone()));
        }
    }
    Err(format!(
        "{PROJECT} {version} publishes no sdist and no pure-Python wheel, so its surface cannot be recovered from the registry"
    ))
}

/// Unpack an artifact into `scratch` and return the root that holds `wisent/`.
fn unpack(marker: &str, entry: &Value, scratch: &Path) -> Result<PathBuf, String> {
    let filename = entry["filename"].as_str().unwrap_or_default();
    let payload = fetch(entry["url"].as_str().ok_or_else(|| format!("{filename} names no url"))?)?;
    if marker == SDIST_MARKER {
        tar::Archive::new(flate2::read::GzDecoder::new(payload.as_slice()))
            .unpack(scratch)
            .map_err(|error| format!("{filename} does not unpack: {error}"))?;
    } else {
        zip::ZipArchive::new(std::io::Cursor::new(&payload))
            .and_then(|mut archive| archive.extract(scratch))
            .map_err(|error| format!("{filename} does not unpack: {error}"))?;
    }
    if scratch.join("wisent").is_dir() {
        return Ok(scratch.to_path_buf());
    }
    let inner: Vec<PathBuf> = std::fs::read_dir(scratch)
        .map_err(|error| format!("{}: {error}", scratch.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|child| child.join("wisent").is_dir())
        .collect();
    match inner.as_slice() {
        [root] => Ok(root.clone()),
        _ => Err(format!("{filename}: expected exactly one tree containing `wisent`, found {}", inner.len())),
    }
}

/// The surface of one published artifact, read tolerantly: a module in it that
/// does not parse could not be imported by whoever installed it either.
fn recover(marker: &str, entry: &Value, scratch: &Path) -> Result<(Vec<String>, Vec<String>), String> {
    if scratch.exists() {
        std::fs::remove_dir_all(scratch).map_err(|error| format!("{}: {error}", scratch.display()))?;
    }
    std::fs::create_dir_all(scratch).map_err(|error| format!("{}: {error}", scratch.display()))?;
    let read = unpack(marker, entry, scratch).and_then(|root| surface(&root, true));
    let cleaned = std::fs::remove_dir_all(scratch);
    let read = read?;
    cleaned.map_err(|error| format!("{} was not removed: {error}", scratch.display()))?;
    Ok(read)
}

fn cross_check(version: &str, scratch: &Path) -> Result<(), String> {
    let found = candidates(version)?;
    let missing: Vec<&str> = [SDIST_MARKER, WHEEL_MARKER].into_iter().filter(|marker| !found.contains_key(*marker)).collect();
    if !missing.is_empty() {
        return Err(format!(
            "{PROJECT} {version} publishes no {}, so the two readers cannot be compared on it; pick a version that has both",
            missing.join(" and no ")
        ));
    }
    let mut surfaces = Map::new();
    let mut refused = Vec::new();
    for (marker, entry) in &found {
        match recover(marker, entry, scratch) {
            Ok((names, _)) => {
                surfaces.insert(marker.clone(), json!(names));
            }
            Err(why) => refused.push(format!("{marker} {}: {why}", entry["filename"].as_str().unwrap_or_default())),
        }
    }
    if !refused.is_empty() {
        refused.sort();
        return Err(format!("cannot compare the readers on {PROJECT} {version}; {}", refused.join("; ")));
    }
    let names = |marker: &str| -> Vec<String> {
        surfaces[marker].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect()
    };
    let (from_sdist, from_wheel) = (names(SDIST_MARKER), names(WHEEL_MARKER));
    let only_sdist: Vec<&String> = from_sdist.iter().filter(|name| !from_wheel.contains(name)).collect();
    let only_wheel: Vec<&String> = from_wheel.iter().filter(|name| !from_sdist.contains(name)).collect();
    if !only_sdist.is_empty() || !only_wheel.is_empty() {
        return Err(format!(
            "the two readers disagree on {PROJECT} {version}: sdist only {only_sdist:?}, wheel only {only_wheel:?}. \
             A wheel that drops names reads as removed capability, so the wheel tier cannot be trusted until this is explained"
        ));
    }
    println!(
        "{version}: {} and {} agree on {} names",
        found[SDIST_MARKER]["filename"].as_str().unwrap_or_default(),
        found[WHEEL_MARKER]["filename"].as_str().unwrap_or_default(),
        from_sdist.len()
    );
    Ok(())
}

/// What PyPI serves now: (version, marker, entry, source string). No download.
fn identity() -> Result<(String, &'static str, Value, String), String> {
    let published = json_at(&format!("https://pypi.org/pypi/{PROJECT}/json"))?;
    let version = published["info"]["version"].as_str().ok_or("PyPI names no info.version")?.to_string();
    let (marker, entry) = artifact(&version)?;
    let mut tail = format!(
        "{} unpacked and read by wisent-evaluators-release surface",
        entry["filename"].as_str().unwrap_or_default()
    );
    if marker == WHEEL_MARKER {
        tail.push_str("; that release publishes no sdist");
    }
    Ok((version, marker, entry, format!("{marker}:{tail}")))
}

/// `baseline [--best | --cross-check VERSION]` for the repository at `repository`.
pub fn run(repository: &Path, scratch: &Path, flags: &[String]) -> Result<(), String> {
    match flags {
        [flag, version] if flag == "--cross-check" => return cross_check(version, scratch),
        [flag] if flag == "--cross-check" => return Err("--cross-check needs the version to compare the readers on".to_string()),
        [flag] if flag == "--best" => {
            let (version, _, _, source) = identity()?;
            let document = json!({ "version": version, "source": source });
            println!("{}", serde_json::to_string_pretty(&document).map_err(|error| error.to_string())?);
            return Ok(());
        }
        [] => {}
        _ => return Err(format!("baseline does not take {}", flags.join(" "))),
    }
    let (version, marker, entry, source) = identity()?;
    let (names, skipped) = recover(marker, &entry, scratch)?;
    let mut document = Map::new();
    document.insert("version".to_string(), json!(version));
    document.insert("source".to_string(), json!(source));
    document.insert("surface".to_string(), json!(names));
    if !skipped.is_empty() {
        document.insert("unparseable".to_string(), json!(skipped));
    }
    let path = repository.join("released-surface.json");
    let rendered = serde_json::to_string_pretty(&document).map_err(|error| error.to_string())? + "\n";
    std::fs::write(&path, rendered).map_err(|error| format!("{}: {error}", path.display()))?;
    println!("released-surface.json: {marker} {version}, {} names", names.len());
    Ok(())
}
