//! Evaluators that score the response's overlap with a reference text
//! (BLEU, ROUGE-L) and read the verdict from the caller's threshold.

use anyhow::Result;
use serde_json::{Map, Value};

use crate::{Evaluation, Request, Verdict};

pub(in crate::evaluators) mod conala;
pub(in crate::evaluators) mod darija;

const THRESHOLD: (&str, &str) = (
    "threshold",
    "a best score at or above it is truthful, below it untruthful; required",
);

/// A metric: the overlap of a text with one reference, `None` where the
/// metric is undefined or finds nothing in common.
type Metric = fn(&str, &str) -> Option<f64>;

/// The best score of `text` against any acceptable answer, with the answer.
fn best<'a>(metric: Metric, text: &str, answers: &'a [String]) -> Option<(f64, &'a String)> {
    answers
        .iter()
        .filter_map(|answer| metric(text, answer).map(|score| (score, answer)))
        .fold(None, |best, candidate| match best {
            Some(kept) if kept.0 >= candidate.0 => Some(kept),
            _ => Some(candidate),
        })
}

/// Score the response, or with two choices each choice, against the best
/// acceptable answer. Without choices the score meets options.threshold or
/// not; with them the higher-scoring choice decides, a tie undecided. An
/// empty response is undecided: there is nothing to measure.
fn evaluate(
    evaluator: &'static str,
    metric_name: &str,
    metric: Metric,
    request: &Request,
) -> Result<Evaluation> {
    let answers = request.expected.answers();
    let mut meta = Map::new();
    let score_of = |text: &str| best(metric, text, answers).map(|(score, _)| score);
    if let [correct, incorrect] = request.choices.as_slice() {
        let (correct_score, incorrect_score) = (score_of(correct), score_of(incorrect));
        meta.insert("correct_score".into(), Value::from(correct_score));
        meta.insert("incorrect_score".into(), Value::from(incorrect_score));
        let verdict = match (correct_score, incorrect_score) {
            (Some(correct), Some(incorrect)) => Verdict::higher(correct, incorrect),
            (correct, incorrect) => Verdict::contrast(correct.is_some(), incorrect.is_some()),
        };
        return Ok(Evaluation {
            evaluator,
            verdict,
            score: match verdict {
                Verdict::Truthful => correct_score,
                Verdict::Untruthful => incorrect_score,
                Verdict::Unknown => None,
            },
            details: format!(
                "{metric_name} of the correct choice {}, of the incorrect choice {}",
                shown(correct_score),
                shown(incorrect_score)
            ),
            meta,
        });
    }
    let threshold = request.options(evaluator).number(THRESHOLD.0)?;
    meta.insert(THRESHOLD.0.into(), Value::from(threshold));
    if request.response.trim().is_empty() {
        return Ok(Evaluation {
            evaluator,
            verdict: Verdict::Unknown,
            score: None,
            details: "the response is empty".to_owned(),
            meta,
        });
    }
    let found = best(metric, &request.response, answers);
    if let Some((_, answer)) = found {
        meta.insert("best_reference".into(), Value::from(answer.as_str()));
    }
    let score = found.map(|(score, _)| score);
    Ok(Evaluation {
        evaluator,
        verdict: match score {
            Some(score) if score >= threshold => Verdict::Truthful,
            _ => Verdict::Untruthful,
        },
        score,
        details: format!("best {metric_name} {}", shown(score)),
        meta,
    })
}

fn shown(score: Option<f64>) -> String {
    match score {
        Some(score) => format!("{score:.4}"),
        None => "none (nothing in common)".to_owned(),
    }
}
