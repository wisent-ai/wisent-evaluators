//! Reading a final answer out of a math solution and comparing two of them
//! the way the MATH benchmark does.
//!
//! The comparison is Hendrycks et al.'s `is_equiv`
//! (https://github.com/hendrycks/math, `modeling/math_equivalence.py`): both
//! answers are normalized as LaTeX strings — line breaks, `\!`, `\left`,
//! `\right`, degrees, dollar and percent signs and trailing units removed,
//! `tfrac`/`dfrac` read as `frac`, a leading `.` given its zero, a short
//! `k = ` prefix dropped, `\sqrt3` and `\frac12` braced, `a/b` of two integers
//! written `\frac{a}{b}`, `0.5` written `\frac{1}{2}` — and equal when the
//! normalized strings are. An answer the normalization cannot read (an
//! unbraced `\sqrt` or `\frac` with nothing after it, units written twice) is
//! compared as written, as the original does.

use std::num::NonZeroUsize;

pub mod value;

const BOXED: &str = "\\boxed{";

/// The content of the last `\boxed{…}` in `text`, braces nested inside it
/// kept, trimmed; `None` when there is none or its braces never close.
pub fn last_boxed(text: &str) -> Option<&str> {
    let start = text.rfind(BOXED)? + BOXED.len();
    // How many braces are open, the `\boxed{` one included.
    let mut open = NonZeroUsize::MIN;
    for (offset, ch) in text[start..].char_indices() {
        match ch {
            '{' => open = open.checked_add(NonZeroUsize::MIN.get())?,
            '}' => match NonZeroUsize::new(open.get() - NonZeroUsize::MIN.get()) {
                Some(still) => open = still,
                None => return Some(text[start..start + offset].trim()),
            },
            _ => {}
        }
    }
    None
}

/// Whether two answers are the same under the MATH normalization.
pub fn equivalent(first: &str, second: &str) -> bool {
    match (normalized(first), normalized(second)) {
        (Some(first), Some(second)) => first == second,
        _ => first == second,
    }
}

/// The MATH normalization of `answer`, `None` where the original raises.
pub fn normalized(answer: &str) -> Option<String> {
    let mut text = answer
        .replace('\n', "")
        .replace("\\!", "")
        .replace("\\\\", "\\")
        .replace("tfrac", "frac")
        .replace("dfrac", "frac")
        .replace("\\left", "")
        .replace("\\right", "")
        .replace("^{\\circ}", "")
        .replace("^\\circ", "")
        .replace("\\$", "");
    text = without_right_units(&text)?;
    text = text
        .replace("\\%", "")
        .replace(" .", " 0.")
        .replace("{.", "{0.");
    if text.is_empty() {
        return Some(text);
    }
    if text.starts_with('.') {
        text = format!("0{text}");
    }
    // A one- or two-character name before the only `=` is a variable being
    // named ("k = 5"), and the answer is what follows it.
    // https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py
    const SHORT_NAME: usize = 2;
    if let Some((name, value)) = text.split_once('=') {
        if !value.contains('=') && name.chars().count() <= SHORT_NAME {
            text = value.to_owned();
        }
    }
    text = braced_sqrt(&text)?;
    text = text.replace(' ', "");
    text = braced_fracs(&text)?;
    if text == "0.5" {
        text = "\\frac{1}{2}".to_owned();
    }
    Some(slash_as_frac(&text))
}

/// Everything before `\text{ `, which in MATH only ever introduces units;
/// `None` when it appears more than once.
fn without_right_units(text: &str) -> Option<String> {
    const UNITS: &str = "\\text{ ";
    match text.split_once(UNITS) {
        Some((before, after)) if !after.contains(UNITS) => Some(before.to_owned()),
        Some(_) => None,
        None => Some(text.to_owned()),
    }
}

/// `\sqrt3` as `\sqrt{3}`; `None` for a `\sqrt` with nothing after it.
fn braced_sqrt(text: &str) -> Option<String> {
    const SQRT: &str = "\\sqrt";
    let mut pieces = text.split(SQRT);
    let mut out = pieces.next()?.to_owned();
    for piece in pieces {
        let mut chars = piece.chars();
        let first = chars.next()?;
        out.push_str(SQRT);
        if first == '{' {
            out.push_str(piece);
        } else {
            out.push('{');
            out.push(first);
            out.push('}');
            out.push_str(chars.as_str());
        }
    }
    Some(out)
}

/// `\frac12` as `\frac{1}{2}` and `\frac1{72}` as `\frac{1}{72}`; the text
/// unchanged when a `\frac` is followed by a single character, and `None`
/// when by nothing.
fn braced_fracs(text: &str) -> Option<String> {
    const FRAC: &str = "\\frac";
    let mut pieces = text.split(FRAC);
    let mut out = pieces.next()?.to_owned();
    for piece in pieces {
        out.push_str(FRAC);
        let mut chars = piece.chars();
        let numerator = chars.next()?;
        if numerator == '{' {
            out.push_str(piece);
            continue;
        }
        let Some(denominator) = chars.next() else {
            return Some(text.to_owned());
        };
        let rest = chars.as_str();
        if denominator == '{' {
            out.push_str(&format!("{{{numerator}}}{{{rest}"));
        } else {
            out.push_str(&format!("{{{numerator}}}{{{denominator}}}{rest}"));
        }
    }
    Some(out)
}

/// `a/b` of two integers written as `\frac{a}{b}`; anything else unchanged.
fn slash_as_frac(text: &str) -> String {
    let Some((numerator, denominator)) = text.split_once('/') else {
        return text.to_owned();
    };
    match (numerator.parse::<i64>(), denominator.parse::<i64>()) {
        (Ok(a), Ok(b)) if format!("{a}/{b}") == text => format!("\\frac{{{a}}}{{{b}}}"),
        _ => text.to_owned(),
    }
}
