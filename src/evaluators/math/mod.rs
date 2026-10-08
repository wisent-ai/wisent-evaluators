//! Evaluators that read a final answer out of a math solution: `math`
//! (competition math, MATH's LaTeX equivalence) and `aime` (AIME's integer
//! answers).

use anyhow::Result;
use serde_json::{Map, Value};

use crate::{math, Evaluation, Request, Verdict};

pub(in crate::evaluators) struct MathAnswer;
pub(in crate::evaluators) struct Aime;

const MATH: &str = "math";
const AIME: &str = "aime";

/// The response's final answer: its last `\boxed{}`, else the whole response
/// trimmed (pair sets store the bare answer); `None` for an empty response.
fn final_answer(response: &str) -> Option<&str> {
    match math::last_boxed(response) {
        Some(boxed) => Some(boxed),
        None => Some(response.trim()).filter(|answer| !answer.is_empty()),
    }
}

fn verdict(matched: bool) -> Verdict {
    if matched {
        Verdict::Truthful
    } else {
        Verdict::Untruthful
    }
}

impl crate::Evaluator for MathAnswer {
    fn name(&self) -> &'static str {
        MATH
    }

    fn description(&self) -> &'static str {
        "The response's last \\boxed{} answer equals an expected answer under MATH's LaTeX normalization (Hendrycks et al.)"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &[(
            "expected_as_written",
            "true compares against the expected answers as written instead of their last \\boxed{}",
        )]
    }

    fn evaluate(&self, request: &Request) -> Result<Evaluation> {
        let as_written = request.options(MATH).switch("expected_as_written")?;
        let answer = final_answer(&request.response);
        let expected: Vec<&str> = request
            .expected
            .answers()
            .iter()
            .map(|expected| match (as_written, math::last_boxed(expected)) {
                (false, Some(boxed)) => boxed,
                _ => expected.as_str(),
            })
            .collect();
        let matched = answer.and_then(|answer| {
            expected
                .iter()
                .find(|expected| math::equivalent(answer, expected))
        });
        let mut meta = Map::new();
        meta.insert("model_answer".into(), Value::from(answer));
        meta.insert("expected_answers".into(), Value::from(expected.clone()));
        meta.insert("expected_as_written".into(), Value::Bool(as_written));
        if let Some(matched) = matched {
            meta.insert("matched_answer".into(), Value::from(*matched));
        }
        Ok(Evaluation {
            evaluator: MATH,
            verdict: verdict(matched.is_some()),
            score: None,
            details: match answer {
                Some(answer) => format!("model answer {answer:?} against {expected:?}"),
                None => "the response is empty".to_owned(),
            },
            meta,
        })
    }
}

impl crate::Evaluator for Aime {
    fn name(&self) -> &'static str {
        AIME
    }

    fn description(&self) -> &'static str {
        "The response's last \\boxed{} answer, read as an integer, equals the expected integer"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    fn evaluate(&self, request: &Request) -> Result<Evaluation> {
        let answer = final_answer(&request.response);
        let model: Option<i64> = answer.and_then(|answer| answer.parse().ok());
        let matched = request
            .expected
            .answers()
            .iter()
            .find(|expected| model.is_some() && expected.trim().parse::<i64>().ok() == model);
        let mut meta = Map::new();
        meta.insert("model_answer".into(), Value::from(answer));
        meta.insert("model_integer".into(), Value::from(model));
        if let Some(matched) = matched {
            meta.insert("matched_answer".into(), Value::from(matched.as_str()));
        }
        Ok(Evaluation {
            evaluator: AIME,
            verdict: verdict(matched.is_some()),
            score: None,
            details: match (answer, model) {
                (Some(_), Some(model)) => {
                    format!(
                        "model integer {model} against {:?}",
                        request.expected.answers()
                    )
                }
                (Some(answer), None) => format!("model answer {answer:?} is not an integer"),
                (None, _) => "the response is empty".to_owned(),
            },
            meta,
        })
    }
}
