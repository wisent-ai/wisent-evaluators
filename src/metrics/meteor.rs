//! METEOR (Banerjee and Lavie, 2005) with its exact-match stage: words are
//! aligned when they are equal once lowercased, and the score is the
//! recall-weighted harmonic mean of precision and recall, discounted by how
//! fragmented the alignment is. The stemming and WordNet-synonym stages are
//! not run, so a paraphrase counts only where it repeats the reference's
//! words.

use std::num::NonZeroUsize;

/// The weight of precision against recall in the mean.
// https://www.nltk.org/api/nltk.translate.meteor_score.html (alpha=0.9)
const ALPHA: f64 = 0.9;
/// How sharply fragmentation is punished.
// https://www.nltk.org/api/nltk.translate.meteor_score.html (beta=3)
const BETA: f64 = 3.0;
/// The largest share of the score fragmentation may take.
// https://www.nltk.org/api/nltk.translate.meteor_score.html (gamma=0.5)
const GAMMA: f64 = 0.5;

/// METEOR of `hypothesis` against `reference`; `None` when no word aligns.
pub fn meteor(hypothesis: &str, reference: &str) -> Option<f64> {
    let words =
        |text: &str| -> Vec<String> { text.split_whitespace().map(str::to_lowercase).collect() };
    let (hypothesis, reference) = (words(hypothesis), words(reference));
    let mut taken = vec![false; reference.len()];
    // Each hypothesis word aligns with the first unaligned equal reference
    // word, in hypothesis order.
    let alignment: Vec<(usize, usize)> = hypothesis
        .iter()
        .enumerate()
        .filter_map(|(at, word)| {
            let found = reference
                .iter()
                .enumerate()
                .position(|(other, candidate)| !taken[other] && candidate == word)?;
            taken[found] = true;
            Some((at, found))
        })
        .collect();
    let matched = NonZeroUsize::new(alignment.len())?.get() as f64;
    let precision = matched / hypothesis.len() as f64;
    let recall = matched / reference.len() as f64;
    let one = NonZeroUsize::MIN.get();
    let mean = precision * recall / (ALPHA * precision + (one as f64 - ALPHA) * recall);
    // A chunk is a run of aligned words adjacent in both texts.
    let chunks = alignment
        .windows(NonZeroUsize::MIN.saturating_add(one).get())
        .filter(|pair| match pair {
            [(first_at, first_found), (second_at, second_found)] => {
                *second_at != first_at + one || *second_found != first_found + one
            }
            _ => false,
        })
        .count()
        + one;
    let penalty = GAMMA * (chunks as f64 / matched).powf(BETA);
    Some((one as f64 - penalty) * mean)
}
