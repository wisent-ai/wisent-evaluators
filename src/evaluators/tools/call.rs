//! A function call written in Python syntax, read with the Python grammar:
//! the called name and each argument's literal value, as Python's
//! `ast.literal_eval` would read it. A positional argument is keyed by its
//! position; one that is not a literal is kept as its source text.

use std::collections::BTreeMap;

use tree_sitter::{Node, Parser};

/// The bases Python's integer prefixes name:
/// https://docs.python.org/3/reference/lexical_analysis.html#integer-literals
const HEXADECIMAL: u32 = 16;
/// https://docs.python.org/3/reference/lexical_analysis.html#integer-literals
const OCTAL: u32 = 8;
/// https://docs.python.org/3/reference/lexical_analysis.html#integer-literals
const BINARY: u32 = 2;

/// A literal value. Numbers compare as Python compares an int with a float.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Literal {
    Text(String),
    Number(f64),
    Bool(bool),
    Nothing,
    List(Vec<Literal>),
    Tuple(Vec<Literal>),
    Set(Vec<Literal>),
    Dict(Vec<(Literal, Literal)>),
    /// An argument that is not a literal, as written (spaces removed).
    Source(String),
}

/// Where an argument sits: its keyword, or its position.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Key {
    Keyword(String),
    Position(usize),
}

pub(super) struct Call {
    pub name: String,
    pub arguments: BTreeMap<Key, Literal>,
}

fn text<'s>(node: Node, source: &'s str) -> &'s str {
    &source[node.byte_range()]
}

fn named(node: Node) -> Vec<Node> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor).collect()
}

/// The call `source` makes, when it is one call expression and nothing else.
pub(super) fn parse(source: &str) -> Option<Call> {
    let source = source.trim();
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .ok()?;
    let tree = parser.parse(source, None)?;
    let root = tree.root_node();
    if root.has_error() {
        return None;
    }
    let [statement] = named(root)[..] else {
        return None;
    };
    let [call] = named(statement)[..] else {
        return None;
    };
    if statement.kind() != "expression_statement" || call.kind() != "call" {
        return None;
    }
    let function = call.child_by_field_name("function")?;
    let name: String = text(function, source).split_whitespace().collect();
    let mut arguments = BTreeMap::new();
    for (position, argument) in named(call.child_by_field_name("arguments")?)
        .into_iter()
        .enumerate()
    {
        if argument.kind() == "keyword_argument" {
            let keyword = text(argument.child_by_field_name("name")?, source).to_owned();
            let value = argument.child_by_field_name("value")?;
            arguments.insert(Key::Keyword(keyword), literal(value, source)?);
        } else {
            let value = match literal(argument, source) {
                Some(value) => value,
                None => Literal::Source(text(argument, source).split_whitespace().collect()),
            };
            arguments.insert(Key::Position(position), value);
        }
    }
    Some(Call { name, arguments })
}

fn items(node: Node, source: &str) -> Option<Vec<Literal>> {
    named(node)
        .into_iter()
        .map(|child| literal(child, source))
        .collect()
}

/// The literal `node` writes, `None` when it is not one.
fn literal(node: Node, source: &str) -> Option<Literal> {
    match node.kind() {
        "string" => string(node, source).map(Literal::Text),
        "concatenated_string" => {
            let parts: Option<Vec<String>> = named(node)
                .into_iter()
                .map(|part| string(part, source))
                .collect();
            parts.map(|parts| Literal::Text(parts.concat()))
        }
        "integer" | "float" => number(text(node, source)).map(Literal::Number),
        "true" => Some(Literal::Bool(true)),
        "false" => Some(Literal::Bool(false)),
        "none" => Some(Literal::Nothing),
        "list" => items(node, source).map(Literal::List),
        "tuple" => items(node, source).map(Literal::Tuple),
        "set" => items(node, source).map(Literal::Set),
        "parenthesized_expression" => match named(node)[..] {
            [inner] => literal(inner, source),
            _ => None,
        },
        "unary_operator" => {
            let operator = text(node.child_by_field_name("operator")?, source);
            let argument = literal(node.child_by_field_name("argument")?, source)?;
            match (operator, argument) {
                ("-", Literal::Number(value)) => Some(Literal::Number(-value)),
                ("+", Literal::Number(value)) => Some(Literal::Number(value)),
                _ => None,
            }
        }
        "dictionary" => {
            let pairs: Option<Vec<(Literal, Literal)>> = named(node)
                .into_iter()
                .map(|pair| {
                    Some((
                        literal(pair.child_by_field_name("key")?, source)?,
                        literal(pair.child_by_field_name("value")?, source)?,
                    ))
                })
                .collect();
            pairs.map(Literal::Dict)
        }
        _ => None,
    }
}

/// A Python number's value: underscores dropped, hexadecimal, octal and
/// binary integers read in their base.
fn number(written: &str) -> Option<f64> {
    let lowered: String = written
        .chars()
        .filter(|ch| *ch != '_')
        .collect::<String>()
        .to_ascii_lowercase();
    let based = [("0x", HEXADECIMAL), ("0o", OCTAL), ("0b", BINARY)]
        .into_iter()
        .find_map(|(prefix, radix)| lowered.strip_prefix(prefix).map(|digits| (digits, radix)));
    match based {
        Some((digits, radix)) => i128::from_str_radix(digits, radix)
            .ok()
            .map(|value| value as f64),
        None => lowered.parse::<f64>().ok(),
    }
}

/// A string literal's value with its escapes read; `None` for a bytes or
/// formatted string, which `literal_eval` does not read as text.
fn string(node: Node, source: &str) -> Option<String> {
    if node.kind() != "string" {
        return None;
    }
    let mut value = String::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "string_start" => {
                let prefix = text(child, source)
                    .trim_end_matches(['"', '\''])
                    .to_ascii_lowercase();
                if prefix.contains('b') || prefix.contains('f') {
                    return None;
                }
            }
            "string_content" => {
                let mut inner = child.walk();
                let mut last = child.start_byte();
                for part in child.children(&mut inner) {
                    value.push_str(&source[last..part.start_byte()]);
                    if part.kind() != "escape_sequence" {
                        return None;
                    }
                    value.push_str(escape(text(part, source))?);
                    last = part.end_byte();
                }
                value.push_str(&source[last..child.end_byte()]);
            }
            "string_end" => {}
            _ => return None,
        }
    }
    Some(value)
}

/// The text one escape sequence stands for.
fn escape(sequence: &str) -> Option<&'static str> {
    match sequence.strip_prefix('\\')? {
        "n" => Some("\n"),
        "t" => Some("\t"),
        "r" => Some("\r"),
        "\\" => Some("\\"),
        "'" => Some("'"),
        "\"" => Some("\""),
        "\n" => Some(""),
        _ => None,
    }
}
