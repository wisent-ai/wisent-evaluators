//! `f1`: token-overlap F1 between the response and the best-matching
//! acceptable answer (DROP, SQuAD), or, with two choices, which choice the
//! expected answer overlaps.

use std::num::NonZeroUsize;

use anyhow::Result;
use serde_json::{Map, Value};

use crate::{text, Evaluation, Request, Verdict};

pub(in crate::evaluators) struct F1;

const NAME: &str = "f1";

impl crate::Evaluator for F1 {
    fn name(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        "Token-overlap F1 against the best acceptable answer; with two choices, which choice the expected answer overlaps"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &[
            ("threshold", "an F1 at or above it is truthful"),
            (
                "partial_threshold",
                "without choices, an F1 below threshold but at or above this is undecided, and below it untruthful",
            ),
            ("raw", "true splits the texts as they are, without normalizing"),
        ]
    }

    fn evaluate(&self, request: &Request) -> Result<Evaluation> {
        let options = request.options(NAME);
        let threshold = options.number("threshold")?;
        let raw = options.switch("raw")?;
        let answers = request.expected.answers();
        let mut meta = Map::new();
        meta.insert("threshold".into(), Value::from(threshold));
        meta.insert("raw".into(), Value::Bool(raw));
        if let [correct, incorrect] = request.choices.as_slice() {
            let best = |choice: &str| best_f1(choice, answers, raw).map(|(score, _)| score);
            let (correct_f1, incorrect_f1) = (best(correct), best(incorrect));
            meta.insert("correct_f1".into(), Value::from(correct_f1));
            meta.insert("incorrect_f1".into(), Value::from(incorrect_f1));
            let passes = |score: Option<f64>| score.is_some_and(|score| score >= threshold);
            let (verdict, score) = match (passes(correct_f1), passes(incorrect_f1)) {
                (true, false) => (Verdict::Truthful, correct_f1),
                (false, true) => (Verdict::Untruthful, incorrect_f1),
                _ => (Verdict::Unknown, higher(correct_f1, incorrect_f1)),
            };
            return Ok(Evaluation {
                evaluator: NAME,
                verdict,
                score,
                details: format!(
                    "correct choice F1 {}, incorrect choice F1 {}",
                    shown(correct_f1),
                    shown(incorrect_f1)
                ),
                meta,
            });
        }
        let partial = options.number("partial_threshold")?;
        meta.insert("partial_threshold".into(), Value::from(partial));
        let best = best_f1(&request.response, answers, raw);
        let score = best.as_ref().map(|(score, _)| *score);
        if let Some((_, answer)) = &best {
            meta.insert("matched_answer".into(), Value::from(answer.as_str()));
        }
        let verdict = match score {
            Some(score) if score >= threshold => Verdict::Truthful,
            Some(score) if score >= partial => Verdict::Unknown,
            _ => Verdict::Untruthful,
        };
        Ok(Evaluation {
            evaluator: NAME,
            verdict,
            score,
            details: format!("best F1 {}", shown(score)),
            meta,
        })
    }
}

fn higher(first: Option<f64>, second: Option<f64>) -> Option<f64> {
    match (first, second) {
        (Some(first), Some(second)) => Some(first.max(second)),
        (Some(only), None) | (None, Some(only)) => Some(only),
        (None, None) => None,
    }
}

fn shown(score: Option<f64>) -> String {
    match score {
        Some(score) => format!("{score:.3}"),
        None => "none (no token in common)".to_owned(),
    }
}

/// The highest F1 of `text` against any answer, with that answer; the first
/// one wins a tie. `None` when no answer shares a token with `text`.
fn best_f1<'a>(text: &str, answers: &'a [String], raw: bool) -> Option<(f64, &'a String)> {
    answers
        .iter()
        .filter_map(|answer| f1(text, answer, raw).map(|score| (score, answer)))
        .fold(None, |best, candidate| match best {
            Some(kept) if kept.0 >= candidate.0 => Some(kept),
            _ => Some(candidate),
        })
}

/// The F1 of two texts' token sets, `None` when they share no token. For
/// sets the harmonic mean of precision and recall is twice the shared tokens
/// over both sizes.
fn f1(response: &str, expected: &str, raw: bool) -> Option<f64> {
    let (response, expected) = (text::tokens(response, raw), text::tokens(expected, raw));
    let shared = NonZeroUsize::new(response.intersection(&expected).count())?.get() as f64;
    let both = (response.len() + expected.len()) as f64;
    // https://en.wikipedia.org/wiki/F-score
    Some(2.0 * shared / both)
}
