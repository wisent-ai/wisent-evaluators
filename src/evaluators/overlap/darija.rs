//! `darija_bench`: DarijaBench (Atlas-Chat) scores translation with BLEU and
//! summarization with ROUGE-L; one request does not say which task it came
//! from, so the higher of the two decides.

use anyhow::Result;

use crate::metrics::{bleu, rouge};
use crate::{Evaluation, Request};

pub(in crate::evaluators) struct DarijaBench;

const NAME: &str = "darija_bench";

impl crate::Evaluator for DarijaBench {
    fn name(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        "DarijaBench: the higher of BLEU (13a tokens) and ROUGE-L against the best reference; with two choices, which choice scores higher"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        std::slice::from_ref(&super::THRESHOLD)
    }

    fn evaluate(&self, request: &Request) -> Result<Evaluation> {
        super::evaluate(NAME, "max(BLEU, ROUGE-L)", score, request)
    }
}

fn score(text: &str, reference: &str) -> Option<f64> {
    let (text, reference) = (text.trim(), reference.trim());
    let translation = bleu::bleu(&bleu::tokenize_13a(text), &[bleu::tokenize_13a(reference)]);
    let summary = rouge::rouge_l(text, reference);
    match (translation, summary) {
        (Some(translation), Some(summary)) => Some(translation.max(summary)),
        (Some(only), None) | (None, Some(only)) => Some(only),
        (None, None) => None,
    }
}
