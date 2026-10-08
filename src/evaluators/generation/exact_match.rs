//! `exact_match`: the response equals one of the acceptable answers (GSM8K's
//! final number, TriviaQA's aliases, LAMBADA's last word), or, with two
//! choices, which choice equals one (Okapi TruthfulQA).

use anyhow::Result;
use serde_json::{Map, Value};

use crate::{text, Evaluation, Request, Verdict};

pub(in crate::evaluators) struct ExactMatch;

const NAME: &str = "exact_match";

impl crate::Evaluator for ExactMatch {
    fn name(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        "The response equals one of the acceptable answers, compared after lenient normalization unless options.raw; with two choices, which choice equals one"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &[
            (
                "raw",
                "true compares the trimmed texts without normalizing accents, punctuation and spacing",
            ),
            ("case_sensitive", "true keeps letter case"),
        ]
    }

    fn evaluate(&self, request: &Request) -> Result<Evaluation> {
        let options = request.options(NAME);
        let (raw, case_sensitive) = (options.switch("raw")?, options.switch("case_sensitive")?);
        let prepare = |text: &str| {
            let text = if raw {
                text.trim().to_owned()
            } else {
                text::normalize(text)
            };
            if case_sensitive {
                text
            } else {
                text.to_lowercase()
            }
        };
        let response = prepare(&request.response);
        let answers = request.expected.answers();
        let mut meta = Map::new();
        meta.insert("raw".into(), Value::Bool(raw));
        meta.insert("case_sensitive".into(), Value::Bool(case_sensitive));
        if let [correct, incorrect] = request.choices.as_slice() {
            let holds = |choice: &str| {
                let choice = prepare(choice);
                answers.iter().any(|answer| prepare(answer) == choice)
            };
            let (correct_holds, incorrect_holds) = (holds(correct), holds(incorrect));
            meta.insert("correct_matches".into(), Value::Bool(correct_holds));
            meta.insert("incorrect_matches".into(), Value::Bool(incorrect_holds));
            return Ok(Evaluation {
                evaluator: NAME,
                verdict: Verdict::contrast(correct_holds, incorrect_holds),
                score: None,
                details: format!(
                    "the correct choice {} and the incorrect choice {} an acceptable answer",
                    if correct_holds {
                        "equals"
                    } else {
                        "does not equal"
                    },
                    if incorrect_holds {
                        "equals"
                    } else {
                        "does not equal"
                    }
                ),
                meta,
            });
        }
        let matched = answers.iter().find(|answer| prepare(answer) == response);
        Ok(match matched {
            Some(answer) => {
                meta.insert("matched_answer".into(), Value::from(answer.as_str()));
                Evaluation {
                    evaluator: NAME,
                    verdict: Verdict::Truthful,
                    score: None,
                    details: format!("{:?} matches {answer:?}", request.response),
                    meta,
                }
            }
            None => Evaluation {
                evaluator: NAME,
                verdict: Verdict::Untruthful,
                score: None,
                details: format!("{:?} matches none of {answers:?}", request.response),
                meta,
            },
        })
    }
}
