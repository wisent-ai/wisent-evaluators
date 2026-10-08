//! The metrics the evaluators score with: reference overlap (BLEU,
//! ROUGE-L, METEOR) and the repeated-sample probabilities (pass at k,
//! G-Pass).

pub mod bleu;
pub mod meteor;
pub mod rouge;
pub mod sampling;
