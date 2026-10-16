//! Evaluators that ask a judge model, through Brama, whether a response
//! does what the benchmark asks: each states the benchmark's own judging
//! question and the answers the judge may give, the judge's answer is bound
//! by a strict JSON schema to one of those labels, and a verdict is read
//! from the label it chose, never by finding a word inside prose.

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

    /// The label the request's expected answer names, when it is exactly
    /// one label (surrounding punctuation and case aside).
    fn label<'a>(&self, answer: &str, labels: impl Iterator<Item = &'a str>) -> Option<&'a str> {
        let said = answer
            .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '_')
            .to_uppercase();
        labels.into_iter().find(|label| *label == said)
    }

    /// The verdict the judge's label at `index`, in the order the evaluator
    /// declares its labels, gives.
    fn verdict(&self, index: usize, request: &Request) -> Verdict {
        match self.answers {
            Answers::Labels(labels) => {
                let (_, verdict) = labels[index];
                verdict
            }
            Answers::Expected(labels) => {
                let holds = self.label(&request.expected.shown(), labels.iter().copied());
                if holds == Some(labels[index]) {
                    Verdict::Truthful
                } else {
                    Verdict::Untruthful
                }
            }
        }
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
        let labels: Vec<&str> = match self.answers {
            Answers::Labels(labels) => labels.iter().map(|(label, _)| *label).collect(),
            Answers::Expected(labels) => labels.to_vec(),
        };
        let answer = brama::judge(&request.options(self.name), &question, &labels)?;
        // A judge that chose no label gives no verdict: `unknown`, with its
        // answer and why in the evaluation.
        let (verdict, details) = match brama::chosen(&answer, &labels) {
            Ok(index) => (
                self.verdict(index, request),
                format!("the judge chose {}", labels[index]),
            ),
            Err(why) => (
                Verdict::Unknown,
                format!("the judge answered {answer:?}: {why}"),
            ),
        };
        let mut meta = Map::new();
        meta.insert("judge_answer".into(), Value::from(answer.as_str()));
        meta.insert(
            "judge_model".into(),
            Value::from(request.options(self.name).text("judge_model")?),
        );
        Ok(Evaluation {
            evaluator: self.name,
            verdict,
            score: None,
            details,
            meta,
        })
    }
}
