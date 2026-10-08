//! Evaluators that read a final answer out of a math solution: `math`
//! (competition math: MATH, OlympiadBench, CNMO, LiveMathBench, PolyMath —
//! MATH's LaTeX equivalence, and equal values when the caller states a
//! tolerance) and `aime` (AIME's integer answers).

use anyhow::Result;
use serde_json::{Map, Value};

use crate::{math, Evaluation, Request, Verdict};

pub(in crate::evaluators) struct MathAnswer;
pub(in crate::evaluators) struct Aime;

const MATH: &str = "math";
const AIME: &str = "aime";
const TOLERANCE: &str = "relative_tolerance";

/// GSM8K's marker before a solution's final answer.
const FINAL_MARKER: &str = "####";

/// The response's final answer: its last `\boxed{}`, else what follows its
/// last `####`, else the whole response trimmed (pair sets store the bare
/// answer); `None` for an empty response.
fn final_answer(response: &str) -> Option<&str> {
    let answer = match (
        math::last_boxed(response),
        response.rsplit_once(FINAL_MARKER),
    ) {
        (Some(boxed), _) => boxed,
        (None, Some((_, after))) => after.trim(),
        (None, None) => response.trim(),
    };
    Some(answer).filter(|answer| !answer.is_empty())
}

/// Whether two answers match: equal under MATH's normalization, or, with a
/// tolerance, equal in value.
fn same(answer: &str, expected: &str, tolerance: Option<f64>) -> bool {
    math::equivalent(answer, expected)
        || tolerance.is_some_and(|tolerance| {
            match (math::value::read(answer), math::value::read(expected)) {
                (Some(answer), Some(expected)) => math::value::same(&answer, &expected, tolerance),
                _ => false,
            }
        })
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
        "The response's final answer (last \\boxed{}, else after the last ####) equals an expected answer under MATH's LaTeX normalization (Hendrycks et al.), or in value within options.relative_tolerance; with two choices, which choice's answer does"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &[
            (
                "expected_as_written",
                "true compares against the expected answers as written instead of their last \\boxed{}",
            ),
            (
                TOLERANCE,
                "when stated, answers that read as the same numbers (or bracketed lists of them) within this relative difference also match",
            ),
        ]
    }

    fn evaluate(&self, request: &Request) -> Result<Evaluation> {
        let options = request.options(MATH);
        let as_written = options.switch("expected_as_written")?;
        let tolerance = options.opt_in_number(TOLERANCE)?;
        let expected: Vec<&str> = request
            .expected
            .answers()
            .iter()
            .map(|expected| match (as_written, math::last_boxed(expected)) {
                (false, Some(boxed)) => boxed,
                _ => expected.as_str(),
            })
            .collect();
        let matching = |text: &str| {
            final_answer(text).and_then(|answer| {
                expected
                    .iter()
                    .find(|expected| same(answer, expected, tolerance))
                    .copied()
            })
        };
        let mut meta = Map::new();
        meta.insert("expected_answers".into(), Value::from(expected.clone()));
        meta.insert("expected_as_written".into(), Value::Bool(as_written));
        meta.insert(TOLERANCE.into(), Value::from(tolerance));
        if let [correct, incorrect] = request.choices.as_slice() {
            let (correct_matches, incorrect_matches) =
                (matching(correct).is_some(), matching(incorrect).is_some());
            meta.insert("correct_answer".into(), Value::from(final_answer(correct)));
            meta.insert(
                "incorrect_answer".into(),
                Value::from(final_answer(incorrect)),
            );
            return Ok(Evaluation {
                evaluator: MATH,
                verdict: Verdict::contrast(correct_matches, incorrect_matches),
                score: None,
                details: format!(
                    "the correct choice's answer {} and the incorrect choice's answer {} an expected answer",
                    if correct_matches { "matches" } else { "does not match" },
                    if incorrect_matches { "matches" } else { "does not match" }
                ),
                meta,
            });
        }
        let answer = final_answer(&request.response);
        let matched = matching(&request.response);
        meta.insert("model_answer".into(), Value::from(answer));
        if let Some(matched) = matched {
            meta.insert("matched_answer".into(), Value::from(matched));
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
