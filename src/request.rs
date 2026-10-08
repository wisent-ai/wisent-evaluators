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
    /// The benchmark's question or task, which judged evaluators show their
    /// judge beside the response.
    #[serde(default)]
    pub prompt: Option<String>,
    /// The answers a benchmark poses as choices, when it poses them: for a
    /// contrastive check, the correct one first and the incorrect one second.
    #[serde(default)]
    pub choices: Vec<String>,
    /// The answers the benchmark marks wrong, when it lists them
    /// (TruthfulQA's incorrect answers).
    #[serde(default)]
    pub incorrect: Vec<String>,
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

    /// The expected answer as a judge reads it: the one answer, or every
    /// acceptable one as a JSON list.
    pub fn shown(&self) -> String {
        match self {
            Self::One(answer) => answer.clone(),
            Self::Many(answers) => {
                serde_json::to_string(answers).expect("a list of strings always serializes")
            }
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

    /// The benchmark's question, which `evaluator` needs.
    pub fn question(&self, evaluator: &str) -> Result<&str> {
        match &self.prompt {
            Some(prompt) => Ok(prompt),
            None => bail!(
                "{evaluator} shows its judge the benchmark's question, and the request carries no prompt"
            ),
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

    /// A number that turns on a comparison the evaluator otherwise does not
    /// make: `None` when the request leaves it out.
    pub fn opt_in_number(&self, name: &str) -> Result<Option<f64>> {
        match self.values.get(name) {
            Some(_) => self.number(name).map(Some),
            None => Ok(None),
        }
    }

    /// A text the evaluator needs, such as the judge's Brama route.
    pub fn text(&self, name: &str) -> Result<&str> {
        match self.values.get(name) {
            Some(Value::String(text)) if !text.trim().is_empty() => Ok(text.trim()),
            Some(other) => bail!(
                "{} reads options.{name} as a non-empty text, and the request states {other}",
                self.evaluator
            ),
            None => bail!(
                "{} needs options.{name}, and the request does not state it",
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
