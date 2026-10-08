//! `tag`: TAG-Bench's table answers compared after light normalization;
//! with two choices, which choice the expected answer matches.

use anyhow::Result;
use serde_json::{Map, Value};

use crate::{Evaluation, Request, Verdict};

pub(in crate::evaluators) struct Tag;

const NAME: &str = "tag";

/// Trimmed, lowercased, every run of whitespace one space, trailing
/// sentence punctuation dropped.
fn normalized(text: &str) -> String {
    let lowered = text.trim().to_lowercase();
    let spaced: Vec<&str> = lowered.split_whitespace().collect();
    spaced
        .join(" ")
        .trim_end_matches(['.', ',', ';', ':', '!', '?'])
        .to_owned()
}

/// Whether one normalized text holds the other.
fn overlaps(first: &str, second: &str) -> bool {
    first.contains(second) || second.contains(first)
}

fn evaluation(verdict: Verdict, details: String, meta: Map<String, Value>) -> Evaluation {
    Evaluation {
        evaluator: NAME,
        verdict,
        score: None,
        details,
        meta,
    }
}

impl crate::Evaluator for Tag {
    fn name(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        "TAG-Bench: the answer equals an expected answer after normalization; with two choices, which one the expected answer matches or contains"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    fn evaluate(&self, request: &Request) -> Result<Evaluation> {
        let answers = request.expected.answers();
        if let [correct, incorrect] = request.choices.as_slice() {
            let expected = normalized(&request.expected.shown());
            let (correct_text, incorrect_text) = (normalized(correct), normalized(incorrect));
            let mut meta = Map::new();
            meta.insert("correct_answer".into(), Value::from(correct.as_str()));
            meta.insert("incorrect_answer".into(), Value::from(incorrect.as_str()));
            let equal = (correct_text == expected, incorrect_text == expected);
            let contained = (
                overlaps(&correct_text, &expected),
                overlaps(&incorrect_text, &expected),
            );
            let (verdict, details) = match (equal, contained) {
                ((true, false), _) => (
                    Verdict::Truthful,
                    "the correct choice equals the expected answer",
                ),
                ((false, true), _) => (
                    Verdict::Untruthful,
                    "the incorrect choice equals the expected answer",
                ),
                (_, (true, false)) => (
                    Verdict::Truthful,
                    "the correct choice holds or is held by the expected answer",
                ),
                (_, (false, true)) => (
                    Verdict::Untruthful,
                    "the incorrect choice holds or is held by the expected answer",
                ),
                _ => (
                    Verdict::Unknown,
                    "neither choice alone matches the expected answer",
                ),
            };
            return Ok(evaluation(verdict, details.to_owned(), meta));
        }
        let response = normalized(&request.response);
        let mut meta = Map::new();
        meta.insert("response_normalized".into(), Value::from(response.as_str()));
        if response.is_empty() {
            return Ok(evaluation(
                Verdict::Unknown,
                "the response is empty".to_owned(),
                meta,
            ));
        }
        match answers.iter().find(|answer| normalized(answer) == response) {
            Some(answer) => {
                meta.insert("matched_answer".into(), Value::from(answer.as_str()));
                Ok(evaluation(
                    Verdict::Truthful,
                    format!("{response:?} matches {answer:?}"),
                    meta,
                ))
            }
            None => Ok(evaluation(
                Verdict::Untruthful,
                format!("{response:?} matches none of {answers:?}"),
                meta,
            )),
        }
    }
}
