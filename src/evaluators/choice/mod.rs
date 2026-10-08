//! `choice`: the option a response picks in a multiple-choice or binary
//! benchmark (MMLU-Redux, Okapi MMLU and HellaSwag, EusExams, CLUE-WSC,
//! PAWS-X, Inverse Scaling, MedConceptsQA, MMMU), named either by its text or
//! by its letter.

use anyhow::Result;
use serde_json::{Map, Value};

use crate::{text, Evaluation, Request, Verdict};

pub(in crate::evaluators) struct Choice;

const NAME: &str = "choice";

impl crate::Evaluator for Choice {
    fn name(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        "The response picks the expected option: its normalized text equals the option's, or a one-letter option is the response's first word (\"B) Paris\" picks B); with two choices, which choice picks it"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    fn evaluate(&self, request: &Request) -> Result<Evaluation> {
        let answers = request.expected.answers();
        let mut meta = Map::new();
        if let [correct, incorrect] = request.choices.as_slice() {
            let (correct_picks, incorrect_picks) = (
                answers.iter().any(|answer| picks(correct, answer)),
                answers.iter().any(|answer| picks(incorrect, answer)),
            );
            meta.insert("correct_picks".into(), Value::Bool(correct_picks));
            meta.insert("incorrect_picks".into(), Value::Bool(incorrect_picks));
            return Ok(Evaluation {
                evaluator: NAME,
                verdict: Verdict::contrast(correct_picks, incorrect_picks),
                score: None,
                details: format!(
                    "the correct choice {} and the incorrect choice {} the expected option",
                    if correct_picks {
                        "picks"
                    } else {
                        "does not pick"
                    },
                    if incorrect_picks {
                        "picks"
                    } else {
                        "does not pick"
                    }
                ),
                meta,
            });
        }
        Ok(
            match answers
                .iter()
                .find(|answer| picks(&request.response, answer))
            {
                Some(answer) => {
                    meta.insert("matched_answer".into(), Value::from(answer.as_str()));
                    Evaluation {
                        evaluator: NAME,
                        verdict: Verdict::Truthful,
                        score: None,
                        details: format!("{:?} picks {answer:?}", request.response),
                        meta,
                    }
                }
                None => Evaluation {
                    evaluator: NAME,
                    verdict: Verdict::Untruthful,
                    score: None,
                    details: format!("{:?} picks none of {answers:?}", request.response),
                    meta,
                },
            },
        )
    }
}

/// Whether `text` picks `option`: equal once normalized, or one of them is a
/// single letter that the other begins with as its own word. A response that
/// merely starts with the letter ("Absolutely" for A) does not pick it.
fn picks(text: &str, option: &str) -> bool {
    let (text, option) = (text::normalize(text), text::normalize(option));
    text == option
        || (single_letter(&option) && first_word(&text) == Some(option.as_str()))
        || (single_letter(&text) && first_word(&option) == Some(text.as_str()))
}

fn single_letter(text: &str) -> bool {
    let mut letters = text.chars();
    letters.next().is_some_and(char::is_alphabetic) && letters.next().is_none()
}

fn first_word(text: &str) -> Option<&str> {
    text.split_whitespace().next()
}
