//! Benchmark evaluators: each one scores a model's answer against what a
//! benchmark expects and says whether the answer is truthful, untruthful or
//! undecided, with what it measured.
//!
//! A request names its evaluator, the response, the expected answer (one
//! string or several acceptable ones), the choices when the benchmark poses
//! them, and the evaluator's options. Every number an evaluator decides with
//! is an option the caller states; an evaluator that needs one the request
//! does not carry refuses it by name.

pub mod evaluators;
pub mod math;
pub mod request;
pub mod result;
pub mod text;

pub use evaluators::{named, registered, Evaluator};
pub use request::{Expected, Options, Request};
pub use result::{Evaluation, Verdict};
