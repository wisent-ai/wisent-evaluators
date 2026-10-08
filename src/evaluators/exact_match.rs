//! `exact_match`: the response equals one of the acceptable answers (GSM8K's
//! final number, TriviaQA's aliases).

use anyhow::Result;
use serde_json::{Map, Value};

use crate::{text, Evaluation, Request, Verdict};

pub(super) struct ExactMatch;

const NAME: &str = "exact_match";

impl super::Evaluator for ExactMatch {
    fn name(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        "The response equals one of the acceptable answers, compared after lenient normalization unless options.raw"
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
        let matched = answers.iter().find(|answer| prepare(answer) == response);
        let mut meta = Map::new();
        meta.insert("raw".into(), Value::Bool(raw));
        meta.insert("case_sensitive".into(), Value::Bool(case_sensitive));
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
