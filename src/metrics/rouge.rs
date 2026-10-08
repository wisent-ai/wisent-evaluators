//! ROUGE-L: the F-measure of the longest common subsequence of words, as
//! Google's `rouge_score` computes it without stemming.

use std::collections::HashMap;
use std::num::NonZeroUsize;

/// ROUGE-L F-measure of `prediction` against `reference`; `None` when they
/// share no word. Words are the lowercased runs of letters and digits; where
/// `rouge_score` keeps only ASCII ones, every script's letters count here, so
/// an Arabic or Chinese answer is measured instead of scoring nothing.
pub fn rouge_l(prediction: &str, reference: &str) -> Option<f64> {
    let (prediction, reference) = (words(prediction), words(reference));
    let common = NonZeroUsize::new(longest_common_subsequence(&prediction, &reference))?.get();
    let both = (prediction.len() + reference.len()) as f64;
    // https://github.com/google-research/google-research/blob/master/rouge/scoring.py
    // (fmeasure: the harmonic mean of LCS/|prediction| and LCS/|reference|)
    Some(2.0 * common as f64 / both)
}

fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_owned)
        .collect()
}

/// The length of the longest common subsequence, as the longest strictly
/// increasing run of matched positions (Hunt and Szymanski): each word of
/// `first` visits its positions in `second` from last to first, so it
/// extends a run at most once.
fn longest_common_subsequence(first: &[String], second: &[String]) -> usize {
    let mut positions: HashMap<&str, Vec<usize>> = HashMap::new();
    for (position, word) in second.iter().enumerate() {
        positions.entry(word.as_str()).or_default().push(position);
    }
    let mut tails: Vec<usize> = Vec::new();
    for word in first {
        if let Some(found) = positions.get(word.as_str()) {
            for &position in found.iter().rev() {
                let at = tails.partition_point(|&tail| tail < position);
                match tails.get_mut(at) {
                    Some(tail) => *tail = position,
                    None => tails.push(position),
                }
            }
        }
    }
    tails.len()
}
