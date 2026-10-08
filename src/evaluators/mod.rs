//! Every evaluator, by name.

use anyhow::{bail, Result};

use crate::{Evaluation, Request};

mod choice;
mod code;
mod generation;
mod judged;
mod label;
mod math;
mod overlap;
mod similarity;
mod tools;

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

/// The evaluators with code of their own; the judged ones are data in
/// `judged::JUDGED`.
static CODED: &[&dyn Evaluator] = &[
    &generation::exact_match::ExactMatch,
    &generation::f1::F1,
    &generation::halueval::HaluEval,
    &generation::tag::Tag,
    &tools::Bfcl,
    &label::UserSpecified,
    &choice::Choice,
    &overlap::darija::DarijaBench,
    &overlap::conala::Conala,
    &overlap::nl2bash::Nl2Bash,
    &overlap::longform::Longform,
    &code::CodeTests,
    &similarity::Generation,
    &math::MathAnswer,
    &math::Aime,
];

/// Every evaluator, ordered by name.
pub fn registered() -> Vec<&'static dyn Evaluator> {
    let mut all: Vec<&'static dyn Evaluator> = CODED.to_vec();
    all.extend(judged::JUDGED.iter().map(|judged| judged as &dyn Evaluator));
    all.sort_by_key(|evaluator| evaluator.name());
    all
}

/// The evaluator `name` selects, or the refusal that lists the names.
pub fn named(name: &str) -> Result<&'static dyn Evaluator> {
    let all = registered();
    match all.iter().find(|evaluator| evaluator.name() == name) {
        Some(found) => Ok(*found),
        None => {
            let names: Vec<&str> = all.iter().map(|evaluator| evaluator.name()).collect();
            bail!(
                "no evaluator is named {name:?}; the evaluators are {}",
                names.join(", ")
            )
        }
    }
}
