//! `longform_writing`: LongForm scores generated text against the reference
//! with METEOR, as its paper does (here METEOR's exact-match stage).

use anyhow::Result;

use crate::metrics::meteor;
use crate::{Evaluation, Request};

pub(in crate::evaluators) struct Longform;

const NAME: &str = "longform_writing";

impl crate::Evaluator for Longform {
    fn name(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        "LongForm: METEOR (exact word matches, no stems or synonyms) against the best reference; with two choices, which choice scores higher"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        std::slice::from_ref(&super::THRESHOLD)
    }

    fn evaluate(&self, request: &Request) -> Result<Evaluation> {
        super::evaluate(NAME, "METEOR", meteor::meteor, request)
    }
}
