//! The metrics the evaluators score with: reference overlap (BLEU,
//! ROUGE-L) and the repeated-sample probabilities (pass at k, G-Pass).

pub mod bleu;
pub mod rouge;
pub mod sampling;
