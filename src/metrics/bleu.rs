//! BLEU for one prediction, as Hugging Face `evaluate`'s `bleu` metric
//! computes it with its defaults: clipped n-gram precisions up to the fourth
//! order, their geometric mean, and the brevity penalty against the shortest
//! reference, with no smoothing.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::LazyLock;

use regex::Regex;

/// The highest n-gram order BLEU counts.
// https://aclanthology.org/P02-1040.pdf (section 2.3: N = 4)
pub const MAX_ORDER: usize = 4;

/// BLEU of `candidate` against `references`, each already split into
/// tokens; `None` for an empty candidate or no non-empty reference, where
/// BLEU is undefined.
pub fn bleu(candidate: &[String], references: &[Vec<String>]) -> Option<f64> {
    let candidate_length = NonZeroUsize::new(candidate.len())?.get() as f64;
    let reference_length = references
        .iter()
        .filter_map(|reference| NonZeroUsize::new(reference.len()))
        .min()?
        .get() as f64;
    let log_precisions: f64 = (NonZeroUsize::MIN.get()..=MAX_ORDER)
        .map(|order| {
            let counted = grams(candidate, order);
            let reference_counts: Vec<HashMap<&[String], usize>> = references
                .iter()
                .map(|reference| grams(reference, order))
                .collect();
            // Each n-gram counts at most as often as one reference holds it.
            let matched: usize = counted
                .iter()
                .filter_map(|(gram, count)| {
                    reference_counts
                        .iter()
                        .filter_map(|reference| reference.get(gram).copied())
                        .max()
                        .map(|most| most.min(*count))
                })
                .sum();
            match NonZeroUsize::new(candidate.windows(order).count()) {
                Some(possible) => (matched as f64 / possible.get() as f64).ln(),
                None => f64::NEG_INFINITY,
            }
        })
        .sum();
    // A zero precision makes the logarithm negative infinity, and the
    // geometric mean the zero that unsmoothed BLEU gives.
    let geometric_mean = (log_precisions / MAX_ORDER as f64).exp();
    Some(if candidate_length > reference_length {
        geometric_mean
    } else {
        geometric_mean * ((candidate_length - reference_length) / candidate_length).exp()
    })
}

fn grams(tokens: &[String], order: usize) -> HashMap<&[String], usize> {
    let mut counts = HashMap::new();
    for gram in tokens.windows(order) {
        *counts.entry(gram).or_default() += NonZeroUsize::MIN.get();
    }
    counts
}

static PUNCTUATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"([{-~\[-` -&(-+:-@/])").expect("the 13a punctuation pattern compiles")
});
static MARK_AFTER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([^[:digit:]])([.,])").expect("the 13a period pattern compiles"));
static MARK_BEFORE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([.,])([^[:digit:]])").expect("the 13a comma pattern compiles"));
static DASH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([[:digit:]])(-)").expect("the 13a dash pattern compiles"));

/// The `13a` tokenization BLEU uses by default (mteval-v13a.pl, as
/// sacreBLEU and Hugging Face `evaluate` port it): HTML entities decoded,
/// punctuation split off, a period or comma split off unless it sits
/// between digits, and a dash split off after a digit.
// https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/tokenizers/tokenizer_13a.py
pub fn tokenize_13a(text: &str) -> Vec<String> {
    let mut line = text
        .replace("<skipped>", "")
        .replace("-\n", "")
        .replace('\n', " ");
    if line.contains('&') {
        line = line
            .replace("&quot;", "\"")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">");
    }
    let line = format!(" {line} ");
    let line = PUNCTUATION.replace_all(&line, " $1 ");
    let line = MARK_AFTER.replace_all(&line, "$1 $2 ");
    let line = MARK_BEFORE.replace_all(&line, " $1 $2");
    let line = DASH.replace_all(&line, "$1 $2 ");
    line.split_whitespace().map(str::to_owned).collect()
}
