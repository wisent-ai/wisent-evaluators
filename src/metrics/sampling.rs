//! Metrics over repeated samples of one problem: of `samples` answers,
//! `correct` were judged correct, and the metric asks how likely a draw of
//! `draws` of them is to be good enough. Every probability is an exact ratio
//! of binomial coefficients, computed through logarithms so a few hundred
//! samples do not overflow.

use std::num::NonZeroU64;

/// `ln C(total, chosen)`; `None` when `chosen` exceeds `total`.
fn ln_choose(total: u64, chosen: u64) -> Option<f64> {
    let rest = total.checked_sub(chosen)?;
    Some(
        (NonZeroU64::MIN.get()..=chosen)
            .map(|step| ((rest + step) as f64 / step as f64).ln())
            .sum(),
    )
}

/// The probability that `chosen` draws without replacement from `total`
/// answers, `correct` of them correct, hold exactly `hits` correct ones;
/// `None` where that cannot happen.
fn hypergeometric(total: u64, correct: u64, chosen: u64, hits: u64) -> Option<f64> {
    let right = ln_choose(correct, hits)?;
    let wrong = ln_choose(total - correct, chosen.checked_sub(hits)?)?;
    let all = ln_choose(total, chosen)?;
    Some((right + wrong - all).exp())
}

/// Why a sample count cannot be scored.
fn checked(samples: u64, correct: u64, draws: NonZeroU64) -> Result<(), String> {
    if correct > samples {
        return Err(format!(
            "{correct} correct answers out of {samples} samples is impossible"
        ));
    }
    if draws.get() > samples {
        return Err(format!(
            "a draw of {draws} needs at least {draws} samples, and there are {samples}"
        ));
    }
    Ok(())
}

/// Pass at k: the probability that at least one of `draws` answers drawn
/// from `samples`, `correct` of them correct, is correct (Chen et al.'s
/// unbiased estimator, one minus the chance that every draw is wrong).
// https://arxiv.org/abs/2107.03374 (section 2.1, equation 1)
pub fn pass_at_k(samples: u64, correct: u64, draws: NonZeroU64) -> Result<f64, String> {
    checked(samples, correct, draws)?;
    let certain = NonZeroU64::MIN.get() as f64;
    Ok(
        match (
            ln_choose(samples - correct, draws.get()),
            ln_choose(samples, draws.get()),
        ) {
            (Some(all_wrong), Some(all)) => certain - (all_wrong - all).exp(),
            // Fewer wrong answers than draws: some draw is always correct.
            _ => certain,
        },
    )
}

/// G-Pass at k for `required`: the probability that at least `required` of
/// `draws` answers drawn from `samples` are correct (LiveMathBench).
// https://arxiv.org/abs/2412.13147 (section 3, G-Pass)
pub fn g_pass_at_k(
    samples: u64,
    correct: u64,
    draws: NonZeroU64,
    required: NonZeroU64,
) -> Result<f64, String> {
    checked(samples, correct, draws)?;
    if required > draws {
        return Err(format!(
            "G-Pass over {draws} draws cannot require {required} correct ones"
        ));
    }
    Ok((required.get()..=draws.get().min(correct))
        .filter_map(|hits| hypergeometric(samples, correct, draws.get(), hits))
        .sum())
}

/// mG-Pass at k: the mean of G-Pass over the thresholds above one half,
/// `2/k` times the sum of G-Pass for `required` from `ceil(k/2) + 1` to `k`.
// https://arxiv.org/abs/2412.13147 (section 3, mG-Pass)
pub fn m_g_pass_at_k(samples: u64, correct: u64, draws: NonZeroU64) -> Result<f64, String> {
    checked(samples, correct, draws)?;
    // https://arxiv.org/abs/2412.13147 (the thresholds start above one half)
    const HALVES: u64 = 2;
    let first = draws.get().div_ceil(HALVES) + NonZeroU64::MIN.get();
    let total = (first..=draws.get())
        .filter_map(NonZeroU64::new)
        .map(|required| g_pass_at_k(samples, correct, draws, required))
        .sum::<Result<f64, String>>()?;
    Ok(HALVES as f64 * total / draws.get() as f64)
}
