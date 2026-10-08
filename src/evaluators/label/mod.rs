//! `user_specified`: the verdict a person already gave the response, carried
//! into the same output every other evaluator writes.

use anyhow::{bail, Result};
use serde_json::{Map, Value};

use crate::{Evaluation, Request, Verdict};

pub(in crate::evaluators) struct UserSpecified;

const NAME: &str = "user_specified";
const OPTION: &str = "truthful";

impl crate::Evaluator for UserSpecified {
    fn name(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        "The verdict a person gave: options.truthful true or false, or null when they could not decide"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &[(
            OPTION,
            "the person's verdict: true, false, or null for undecided; required",
        )]
    }

    fn evaluate(&self, request: &Request) -> Result<Evaluation> {
        let verdict = match request.options.get(OPTION) {
            Some(Value::Bool(true)) => Verdict::Truthful,
            Some(Value::Bool(false)) => Verdict::Untruthful,
            Some(Value::Null) => Verdict::Unknown,
            Some(other) => {
                bail!("{NAME} reads options.{OPTION} as true, false or null, and the request states {other}")
            }
            None => bail!(
                "{NAME} carries a person's verdict, and the request states no options.{OPTION}"
            ),
        };
        Ok(Evaluation {
            evaluator: NAME,
            verdict,
            score: None,
            details: "the verdict a person gave".to_owned(),
            meta: Map::new(),
        })
    }
}
