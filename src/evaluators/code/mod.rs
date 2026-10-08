//! `code_tests`: a model's Python passes the benchmark's tests when they run
//! against it in the sandbox (HumanEval, MBPP, APPS, LiveCodeBench,
//! Codeforces, OJBench, SciCode, Mercury, DS-One-Thousand in its own image).

use std::borrow::Cow;
use std::sync::LazyLock;

use anyhow::{bail, Result};
use regex::Regex;
use serde_json::{Map, Value};

use crate::{sandbox, Evaluation, Request, Verdict};

pub(in crate::evaluators) struct CodeTests;

const NAME: &str = "code_tests";
const SOLUTION: &str = "solution.py";
const TESTS: &str = "tests.py";

/// The options this evaluator reads besides the sandbox's.
const OWN_OPTIONS: &[(&str, &str)] = &[
    (
        "entry_point",
        "the function the tests check, HumanEval-style: the tests import it from solution and end by calling check on it",
    ),
    (
        "prelude",
        "text written above the code, such as the imports a benchmark's own harness provides",
    ),
];

static OPTIONS: LazyLock<Vec<(&'static str, &'static str)>> = LazyLock::new(|| {
    let mut all = OWN_OPTIONS.to_vec();
    all.extend_from_slice(sandbox::OPTIONS);
    all
});

impl crate::Evaluator for CodeTests {
    fn name(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        "The response's code, written to solution.py, passes the request's tests (tests.py, run with the image's interpreter in a sandbox without network); with two choices, which choice passes"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        &OPTIONS
    }

    fn evaluate(&self, request: &Request) -> Result<Evaluation> {
        let Some(tests) = &request.tests else {
            bail!("{NAME} runs the benchmark's tests, and the request carries no tests");
        };
        let options = request.options(NAME);
        let tests = match options.opt_in_text("entry_point")? {
            Some(entry) => format!("from solution import {entry}\n\n{tests}\n\ncheck({entry})\n"),
            None => tests.clone(),
        };
        let prelude = options.opt_in_text("prelude")?;
        let run = |answer: &str| {
            let code = match prelude {
                Some(prelude) => format!("{prelude}\n{}", code(answer)),
                None => code(answer).into_owned(),
            };
            sandbox::run(&options, &[(SOLUTION, &code), (TESTS, &tests)], TESTS)
        };
        if let [correct, incorrect] = request.choices.as_slice() {
            let (correct_run, incorrect_run) = (run(correct)?, run(incorrect)?);
            let mut meta = Map::new();
            meta.insert("correct_run".into(), Value::Object(correct_run.meta()));
            meta.insert("incorrect_run".into(), Value::Object(incorrect_run.meta()));
            return Ok(Evaluation {
                evaluator: NAME,
                verdict: Verdict::contrast(correct_run.passed, incorrect_run.passed),
                score: None,
                details: format!(
                    "the correct choice {} the tests and the incorrect choice {} them",
                    if correct_run.passed {
                        "passes"
                    } else {
                        "fails"
                    },
                    if incorrect_run.passed {
                        "passes"
                    } else {
                        "fails"
                    }
                ),
                meta,
            });
        }
        let outcome = run(&request.response)?;
        Ok(Evaluation {
            evaluator: NAME,
            verdict: if outcome.passed {
                Verdict::Truthful
            } else {
                Verdict::Untruthful
            },
            score: None,
            details: match (outcome.passed, outcome.exit_code) {
                (true, _) => "the tests pass".to_owned(),
                (false, Some(code)) => format!("the tests exit with status {code}"),
                (false, None) => "a signal ended the tests (a resource limit)".to_owned(),
            },
            meta: outcome.meta(),
        })
    }
}

static FENCE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)```([[:alnum:]_+-]*)[ \t]*\n(.*?)(?:```|\z)")
        .expect("the fence pattern compiles")
});

/// The code in a model's answer: the `code` field of an answer that is a
/// JSON object (the form APPS asks for), else the longest fenced block
/// labelled Python, else the longest fenced block, else the answer as it is.
fn code(answer: &str) -> Cow<'_, str> {
    if let Ok(Value::Object(mut object)) = serde_json::from_str::<Value>(answer.trim()) {
        if let Some(Value::String(code)) = object.remove("code") {
            return Cow::Owned(code);
        }
    }
    let blocks: Vec<(&str, &str)> = FENCE
        .captures_iter(answer)
        .filter_map(|found| Some((found.get(1)?.as_str(), found.get(2)?.as_str())))
        .collect();
    let longest = |python_only: bool| {
        blocks
            .iter()
            .filter(|(label, _)| {
                !python_only
                    || label.eq_ignore_ascii_case("python")
                    || label.eq_ignore_ascii_case("py")
            })
            .map(|(_, body)| body.trim())
            .max_by_key(|body| body.len())
    };
    Cow::Borrowed(match (longest(true), longest(false)) {
        (Some(python), _) => python,
        (None, Some(any)) => any,
        (None, None) => answer.trim(),
    })
}
