//! The text handling evaluators share.

use std::collections::BTreeSet;
use std::sync::LazyLock;

use regex::Regex;
use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};

static PUNCTUATION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[^\w\s]").expect("the punctuation pattern compiles"));
static WHITESPACE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+").expect("the whitespace pattern compiles"));

/// Lenient normalization for natural-language comparison: accents removed
/// (NFKD, combining marks dropped), lowercased, punctuation turned into
/// spaces, whitespace collapsed. "Crème brûlée!" reads "creme brulee".
pub fn normalize(text: &str) -> String {
    let stripped: String = text.nfkd().filter(|ch| !is_combining_mark(*ch)).collect();
    let lowered = stripped.to_lowercase();
    let spaced = PUNCTUATION.replace_all(&lowered, " ");
    WHITESPACE.replace_all(&spaced, " ").trim().to_owned()
}

/// The distinct whitespace-separated tokens of `text`, normalized first
/// unless `raw`.
pub fn tokens(text: &str, raw: bool) -> BTreeSet<String> {
    let text = if raw {
        text.to_owned()
    } else {
        normalize(text)
    };
    text.split_whitespace().map(str::to_owned).collect()
}
