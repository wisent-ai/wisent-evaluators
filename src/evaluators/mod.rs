//! Every evaluator, by name.

use anyhow::{bail, Result};

use crate::{Evaluation, Request};

mod exact_match;
mod f1;

/// One way of scoring a response.
pub trait Evaluator: Sync {
    /// The name a request selects it by.
    fn name(&self) -> &'static str;
    /// One sentence on what it measures.
    fn description(&self) -> &'static str;
    /// The options it reads from a request, each with what it decides.
    fn options(&self) -> &'static [(&'static str, &'static str)];
    fn evaluate(&self, request: &Request) -> Result<Evaluation>;
}

static REGISTERED: &[&dyn Evaluator] = &[&exact_match::ExactMatch, &f1::F1];

/// Every evaluator, ordered by name.
pub fn registered() -> Vec<&'static dyn Evaluator> {
    let mut all = REGISTERED.to_vec();
    all.sort_by_key(|evaluator| evaluator.name());
    all
}

/// The evaluator `name` selects, or the refusal that lists the names.
pub fn named(name: &str) -> Result<&'static dyn Evaluator> {
    match REGISTERED.iter().find(|evaluator| evaluator.name() == name) {
        Some(found) => Ok(*found),
        None => {
            let names: Vec<&str> = registered()
                .iter()
                .map(|evaluator| evaluator.name())
                .collect();
            bail!(
                "no evaluator is named {name:?}; the evaluators are {}",
                names.join(", ")
            )
        }
    }
}
