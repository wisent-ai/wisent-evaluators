//! The public surface of wisent-evaluators: the evaluator names it registers.
//!
//! A caller reaches an evaluator by name (`EvaluatorRotator(evaluator="math")`,
//! `BaseEvaluator.get("math")`), and `BaseEvaluator.__init_subclass__` files
//! each subclass under its class attribute `name`. So the registered names are
//! the contract. They are read with a parser, never by importing (importing
//! needs wisent, torch, transformers and sympy), so this also runs against an
//! unpacked published artifact.
//!
//! A class is an evaluator when its base chain reaches `BaseEvaluator` through
//! classes defined in this tree; bases match on their final segment
//! (`BaseEvaluator`, `atoms.BaseEvaluator`, `BaseEvaluator[T]`), which errs
//! towards reporting a name, never towards dropping one.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use tree_sitter::{Node, Parser};

const ROOT_CLASS: &str = "BaseEvaluator";
const NAME_ATTRIBUTE: &str = "name";

/// One class: its name, the final segments of its bases, the literal `name` it registers ("" for none).
struct Class {
    name: String,
    bases: Vec<String>,
    registered: String,
}

fn text<'s>(node: Node, source: &'s str) -> &'s str {
    &source[node.byte_range()]
}

/// A plain string literal's value: no bytes or template prefix, no interpolation or escape.
fn string_literal(node: Node, source: &str) -> Option<String> {
    if node.kind() != "string" {
        return None;
    }
    let mut value = String::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "string_start" => {
                let prefix = text(child, source).trim_end_matches(['"', '\'']).to_ascii_lowercase();
                if prefix.contains('b') || prefix.contains('f') || prefix.contains('t') {
                    return None;
                }
            }
            "string_content" => {
                let mut inner = child.walk();
                if child.children(&mut inner).next().is_some() {
                    return None;
                }
                value.push_str(text(child, source));
            }
            "string_end" => {}
            _ => return None,
        }
    }
    Some(value)
}

/// The final segment of a base class expression, or "" when it has none.
fn base_name(node: Node, source: &str) -> String {
    match node.kind() {
        "identifier" => text(node, source).to_string(),
        "attribute" => node
            .child_by_field_name("attribute")
            .map(|attribute| text(attribute, source).to_string())
            .unwrap_or_default(),
        "subscript" => node.child_by_field_name("value").map(|value| base_name(value, source)).unwrap_or_default(),
        _ => String::new(),
    }
}

/// The literal assigned to `name` in a class body; the last assignment wins.
fn registered_name(class: Node, source: &str) -> String {
    let Some(body) = class.child_by_field_name("body") else { return String::new() };
    let mut found = String::new();
    let mut cursor = body.walk();
    for statement in body.named_children(&mut cursor).filter(|statement| statement.kind() == "expression_statement") {
        let mut inner = statement.walk();
        for assignment in statement.named_children(&mut inner).filter(|node| node.kind() == "assignment") {
            let Some(left) = assignment.child_by_field_name("left") else { continue };
            if left.kind() != "identifier" || text(left, source) != NAME_ATTRIBUTE {
                continue;
            }
            if let Some(value) = assignment.child_by_field_name("right").and_then(|value| string_literal(value, source)) {
                found = value;
            }
        }
    }
    found
}

/// Every class in one module; a module that does not parse is a refusal.
fn classes(path: &Path) -> Result<Vec<Class>, String> {
    let source = std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .map_err(|error| format!("the Python grammar did not load: {error}"))?;
    let tree = parser
        .parse(&source, None)
        .filter(|tree| !tree.root_node().has_error())
        .ok_or_else(|| format!("{}: does not parse, so the surface is unknown", path.display()))?;
    let mut found = Vec::new();
    let mut pending = vec![tree.root_node()];
    while let Some(node) = pending.pop() {
        let mut cursor = node.walk();
        pending.extend(node.children(&mut cursor));
        if node.kind() != "class_definition" {
            continue;
        }
        let Some(name) = node.child_by_field_name("name") else { continue };
        let mut bases = Vec::new();
        if let Some(arguments) = node.child_by_field_name("superclasses") {
            let mut inner = arguments.walk();
            bases.extend(
                arguments
                    .named_children(&mut inner)
                    .map(|base| base_name(base, &source))
                    .filter(|base| !base.is_empty()),
            );
        }
        found.push(Class { name: text(name, &source).to_string(), bases, registered: registered_name(node, &source) });
    }
    Ok(found)
}

fn python_files(directory: &Path, found: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(directory).map_err(|error| format!("{}: {error}", directory.display()))?;
    for entry in entries {
        let path = entry.map_err(|error| format!("{}: {error}", directory.display()))?.path();
        if path.is_dir() {
            python_files(&path, found)?;
        } else if path.extension().is_some_and(|extension| extension == "py") {
            found.push(path);
        }
    }
    Ok(())
}

/// The registered names under `root`, and the modules `tolerant` skipped.
pub fn surface(root: &Path, tolerant: bool) -> Result<(Vec<String>, Vec<String>), String> {
    let package = root.join("wisent");
    if !package.is_dir() {
        return Err(format!("{} is not a directory; is {} the repository root?", package.display(), root.display()));
    }
    let mut files = Vec::new();
    python_files(&package, &mut files)?;
    files.sort();
    let mut declared = Vec::new();
    let mut skipped = Vec::new();
    for path in files {
        match classes(&path) {
            Ok(found) => declared.extend(found),
            Err(_) if tolerant => skipped.push(path.strip_prefix(root).unwrap_or(&path).display().to_string()),
            Err(refusal) => return Err(refusal),
        }
    }
    let mut evaluators: BTreeSet<String> = BTreeSet::from([ROOT_CLASS.to_string()]);
    loop {
        let before = evaluators.len();
        for class in &declared {
            if class.bases.iter().any(|base| evaluators.contains(base)) {
                evaluators.insert(class.name.clone());
            }
        }
        if evaluators.len() == before {
            break;
        }
    }
    let mut names = BTreeSet::new();
    let mut anonymous = BTreeSet::new();
    for class in declared.iter().filter(|class| class.name != ROOT_CLASS && evaluators.contains(&class.name)) {
        if class.registered.is_empty() {
            anonymous.insert(class.name.clone());
        } else {
            names.insert(class.registered.clone());
        }
    }
    if !anonymous.is_empty() {
        return Err(format!(
            "these evaluator classes register under a name this cannot read, so the surface is unknown: {}",
            anonymous.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    if names.is_empty() {
        return Err(format!(
            "no evaluator names found under {}. Either the evaluators moved, or they stopped subclassing \
             {ROOT_CLASS} with a literal `name` — both change what this package promises, so refusing rather \
             than reporting an empty surface",
            package.display()
        ));
    }
    Ok((names.into_iter().collect(), skipped))
}
