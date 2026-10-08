//! Evaluators that ask a judge model, through Brama, whether a response
//! does what the benchmark asks: each states the benchmark's own judging
//! question and the answers the judge may give, and a verdict is read from
//! the judge's answer by exact label, never by finding a word inside it.

mod catalog;

use anyhow::{bail, Result};
use serde_json::{Map, Value};

use crate::{brama, Evaluation, Request, Verdict};

pub(in crate::evaluators) use catalog::JUDGED;

/// What the judge may answer, and what each answer means.
pub(in crate::evaluators) enum Answers {
    /// Each label the prompt names, with the verdict it gives.
    Labels(&'static [(&'static str, Verdict)]),
    /// The judge classifies the response with one of these labels, and the
    /// request's expected answer names the label that is truthful.
    Expected(&'static [&'static str]),
}

/// One judged evaluator.
pub(in crate::evaluators) struct Judged {
    pub name: &'static str,
    pub description: &'static str,
    /// The judging question, with `{prompt}` (the benchmark's question),
    /// `{expected}` and `{response}` filled in.
    pub prompt: &'static str,
    pub answers: Answers,
}

const PROMPT: &str = "{prompt}";
const EXPECTED: &str = "{expected}";
const RESPONSE: &str = "{response}";

impl Judged {
    fn question(&self, request: &Request) -> Result<String> {
        let mut text = self.prompt.to_owned();
        if text.contains(PROMPT) {
            text = text.replace(PROMPT, request.question(self.name)?);
        }
        Ok(text
            .replace(EXPECTED, &request.expected.shown())
            .replace(RESPONSE, &request.response))
    }

    /// The label the judge answered, when its answer is exactly one label
    /// (surrounding punctuation and case aside).
    fn label<'a>(&self, answer: &str, labels: impl Iterator<Item = &'a str>) -> Option<&'a str> {
        let said = answer
            .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '_')
            .to_uppercase();
        labels.into_iter().find(|label| *label == said)
    }
}

impl crate::Evaluator for Judged {
    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        brama::JUDGE_OPTIONS
    }

    fn evaluate(&self, request: &Request) -> Result<Evaluation> {
        let question = self.question(request)?;
        if let Answers::Expected(labels) = self.answers {
            if self
                .label(&request.expected.shown(), labels.iter().copied())
                .is_none()
            {
                bail!(
                    "{} reads the expected answer as the label that holds, one of {}; the request expects {:?}",
                    self.name,
                    labels.join(", "),
                    request.expected.shown()
                );
            }
        }
        let answer = brama::ask(&request.options(self.name), &question)?;
        let verdict = match self.answers {
            Answers::Labels(labels) => self
                .label(&answer, labels.iter().map(|(label, _)| *label))
                .and_then(|said| labels.iter().find(|(label, _)| *label == said))
                .map(|(_, verdict)| *verdict),
            Answers::Expected(labels) => {
                let holds = self.label(&request.expected.shown(), labels.iter().copied());
                self.label(&answer, labels.iter().copied()).map(|said| {
                    if Some(said) == holds {
                        Verdict::Truthful
                    } else {
                        Verdict::Untruthful
                    }
                })
            }
        };
        let mut meta = Map::new();
        meta.insert("judge_answer".into(), Value::from(answer.as_str()));
        meta.insert(
            "judge_model".into(),
            Value::from(request.options(self.name).text("judge_model")?),
        );
        Ok(Evaluation {
            evaluator: self.name,
            verdict: match verdict {
                Some(verdict) => verdict,
                None => Verdict::Unknown,
            },
            score: None,
            details: match verdict {
                Some(_) => format!("the judge answered {answer:?}"),
                None => format!("the judge answered {answer:?}, which is none of its labels"),
            },
            meta,
        })
    }
}
