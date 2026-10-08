//! What an evaluator answers.

use serde::Serialize;
use serde_json::{Map, Value};

/// Whether the response holds the expected answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Truthful,
    Untruthful,
    /// The measurement does not decide either way (a partial overlap, or both
    /// choices scoring alike).
    Unknown,
}

/// One evaluated response.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Evaluation {
    pub evaluator: &'static str,
    pub verdict: Verdict,
    /// The score the verdict was read from, when the evaluator measures one
    /// (an F1, a similarity); absent for a yes-or-no comparison.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    pub details: String,
    /// What the evaluator measured besides the score, and the options it
    /// decided with.
    pub meta: Map<String, Value>,
}
