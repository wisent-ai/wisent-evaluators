//! `conala`: CoNaLa scores generated Python against the reference snippet
//! with BLEU over code tokens, as the official baseline does.

use std::sync::LazyLock;

use anyhow::Result;
use regex::Regex;

use crate::metrics::bleu;
use crate::{Evaluation, Request};

pub(in crate::evaluators) struct Conala;

const NAME: &str = "conala";

impl crate::Evaluator for Conala {
    fn name(&self) -> &'static str {
        NAME
    }

    fn description(&self) -> &'static str {
        "CoNaLa: BLEU over code tokens (every symbol its own token, camelCase split, quotes unified) against the best reference snippet; with two choices, which choice scores higher"
    }

    fn options(&self) -> &'static [(&'static str, &'static str)] {
        std::slice::from_ref(&super::THRESHOLD)
    }

    fn evaluate(&self, request: &Request) -> Result<Evaluation> {
        super::evaluate(NAME, "BLEU", score, request)
    }
}

fn score(code: &str, reference: &str) -> Option<f64> {
    bleu::bleu(&tokens(code.trim()), &[tokens(reference.trim())])
}

static SYMBOL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([^[:alnum:]_])").expect("the code symbol pattern compiles"));
static CAMEL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([a-z])([A-Z])").expect("the camelCase pattern compiles"));
static QUOTE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"["']"#).expect("the quote pattern compiles"));

/// The CoNaLa baseline's tokenization (Ling et al., "Latent Predictor
/// Networks for Code Generation", 2016).
// https://github.com/conala-corpus/conala-baseline/blob/master/eval/conala_eval.py
fn tokens(code: &str) -> Vec<String> {
    let code = SYMBOL.replace_all(code, " $1 ");
    let code = CAMEL.replace_all(&code, "$1 $2");
    let code = QUOTE.replace_all(&code, "`");
    code.split_whitespace().map(str::to_owned).collect()
}
