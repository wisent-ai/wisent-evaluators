//! One evaluation request: a JSON object per line on the `evaluate` input.

use anyhow::{bail, Result};
use serde::Deserialize;
use serde_json::{Map, Value};

/// A response to score and what the benchmark expects of it.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    /// The evaluator's name, as `wisent-evaluators list` prints it.
    pub evaluator: String,
    pub response: String,
    pub expected: Expected,
    /// The answers a benchmark poses as choices, when it poses them: for a
    /// contrastive check, the correct one first and the incorrect one second.
    #[serde(default)]
    pub choices: Vec<String>,
    /// The evaluator's options, by name.
    #[serde(default)]
    pub options: Map<String, Value>,
}

/// One expected answer, or several acceptable ones.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Expected {
    One(String),
    Many(Vec<String>),
}

impl Expected {
    /// Every acceptable answer, in the order the request gave them.
    pub fn answers(&self) -> &[String] {
        match self {
            Self::One(answer) => std::slice::from_ref(answer),
            Self::Many(answers) => answers,
        }
    }
}

impl Request {
    /// The request's options, read for `evaluator`.
    pub fn options(&self, evaluator: &'static str) -> Options<'_> {
        Options {
            values: &self.options,
            evaluator,
        }
    }
}

/// An evaluator's options, each read by name and refused by name when the
/// evaluator needs it and the request does not state it.
pub struct Options<'a> {
    values: &'a Map<String, Value>,
    evaluator: &'static str,
}

impl Options<'_> {
    /// A number the evaluator decides with; the caller states every one.
    pub fn number(&self, name: &str) -> Result<f64> {
        match self.values.get(name) {
            Some(value) => match value.as_f64() {
                Some(number) => Ok(number),
                None => bail!(
                    "{} reads options.{name} as a number, and the request states {value}",
                    self.evaluator
                ),
            },
            None => bail!(
                "{} decides with options.{name}, and the request does not state it; state the value your calibration run measured",
                self.evaluator
            ),
        }
    }

    /// A switch: on when the request states `true`, off when it states
    /// `false` or leaves it out.
    pub fn switch(&self, name: &str) -> Result<bool> {
        match self.values.get(name) {
            Some(Value::Bool(on)) => Ok(*on),
            Some(other) => bail!(
                "{} reads options.{name} as true or false, and the request states {other}",
                self.evaluator
            ),
            None => Ok(false),
        }
    }
}
