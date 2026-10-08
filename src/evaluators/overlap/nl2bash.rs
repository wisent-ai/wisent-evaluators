//! `nl2bash`: NL2Bash scores a generated shell command against the
//! reference with BLEU over characters, as its paper does.

use anyhow::Result;

use crate::metrics::bleu;
use crate::{Evaluation, Request};

pub(in crate::evaluators) struct Nl2Bash;

const NAME: &str = "nl2bash";

impl crate::Evaluator for Nl2Bash {
    fn name(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        "NL2Bash: character-level BLEU against the best reference command; with two choices, which choice scores higher"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        std::slice::from_ref(&super::THRESHOLD)
    }

    fn evaluate(&self, request: &Request) -> Result<Evaluation> {
        super::evaluate(NAME, "character BLEU", score, request)
    }
}

// https://arxiv.org/abs/1802.08979 (section 5.2: BLEU over characters)
fn score(command: &str, reference: &str) -> Option<f64> {
    let characters =
        |text: &str| -> Vec<String> { text.trim().chars().map(String::from).collect() };
    bleu::bleu(&characters(command), &[characters(reference)])
}
