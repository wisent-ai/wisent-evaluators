//! `bfcl`: the Berkeley Function Calling Leaderboard's all-or-nothing check
//! of a function call, read as Python call syntax: the called name must
//! match and every argument the expected call passes must be passed with the
//! same literal value.

mod call;

use anyhow::Result;
use serde_json::{Map, Value};

use crate::{text, Evaluation, Request, Verdict};

pub(in crate::evaluators) struct Bfcl;

const NAME: &str = "bfcl";

/// Whether `candidate` makes the call `expected` makes.
fn same_call(candidate: &str, expected: &str) -> bool {
    if text::normalize(candidate) == text::normalize(expected) {
        return true;
    }
    match (call::parse(candidate), call::parse(expected)) {
        (Some(candidate), Some(expected)) => {
            candidate.name == expected.name
                && expected
                    .arguments
                    .iter()
                    .all(|(key, value)| candidate.arguments.get(key) == Some(value))
        }
        _ => false,
    }
}

impl crate::Evaluator for Bfcl {
    fn name(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        "BFCL: the response calls the expected function with every expected argument's value; with two choices, which choice does"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    fn evaluate(&self, request: &Request) -> Result<Evaluation> {
        let answers = request.expected.answers();
        let makes = |candidate: &str| answers.iter().find(|answer| same_call(candidate, answer));
        let mut meta = Map::new();
        let (verdict, details) = if let [correct, incorrect] = request.choices.as_slice() {
            meta.insert("correct_answer".into(), Value::from(correct.as_str()));
            meta.insert("incorrect_answer".into(), Value::from(incorrect.as_str()));
            match (makes(correct).is_some(), makes(incorrect).is_some()) {
                (true, false) => (
                    Verdict::Truthful,
                    "the correct choice makes the expected call".to_owned(),
                ),
                (false, true) => (
                    Verdict::Untruthful,
                    "the incorrect choice makes the expected call".to_owned(),
                ),
                _ => (
                    Verdict::Unknown,
                    "both choices or neither make the expected call".to_owned(),
                ),
            }
        } else {
            match makes(&request.response) {
                Some(answer) => {
                    meta.insert("matched_call".into(), Value::from(answer.as_str()));
                    (
                        Verdict::Truthful,
                        format!("the response makes the call {answer:?}"),
                    )
                }
                None => (
                    Verdict::Untruthful,
                    "the response makes none of the expected calls".to_owned(),
                ),
            }
        };
        Ok(Evaluation {
            evaluator: NAME,
            verdict,
            score: None,
            details,
            meta,
        })
    }
}
