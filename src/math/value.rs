//! Reading a math answer as a number, or as a bracketed list of numbers (a
//! point, an interval, a set of roots), so two answers written differently
//! (`\frac{1}{2}`, `0.5`, `50\%`; `2\sqrt{2}`, `\sqrt{8}`) compare by value.
//!
//! The grammar is arithmetic in LaTeX: numbers, `+ - * /`, `\cdot`,
//! `\times`, `\div`, implicit multiplication (`2\pi`, `3(4)`), powers with
//! `^`, `\frac`, `\sqrt` and `\sqrt[n]`, `\pi`, `\ln`, `\exp`, `\sin`,
//! `\cos`, `\tan`, and a trailing percent sign. An answer that names a
//! variable, or anything else outside the grammar, has no numeric reading
//! and is left to the textual comparison.

/// One percent.
// https://en.wikipedia.org/wiki/Percentage
const PER_CENT: f64 = 100.0;

/// What an answer is worth as numbers.
#[derive(Debug, Clone, PartialEq)]
pub enum Reading {
    Number(f64),
    /// Comma-separated values and the brackets around them (a space where
    /// there are none).
    List {
        open: char,
        close: char,
        items: Vec<f64>,
    },
}

/// Whether two readings are the same within `tolerance`, relative to the
/// larger magnitude: one number each, or lists with the same brackets and
/// the same values in the same order.
pub fn same(first: &Reading, second: &Reading, tolerance: f64) -> bool {
    let close = |a: f64, b: f64| a == b || (a - b).abs() <= tolerance * a.abs().max(b.abs());
    match (first, second) {
        (Reading::Number(a), Reading::Number(b)) => close(*a, *b),
        (
            Reading::List {
                open: first_open,
                close: first_close,
                items: first_items,
            },
            Reading::List {
                open: second_open,
                close: second_close,
                items: second_items,
            },
        ) => {
            first_open == second_open
                && first_close == second_close
                && first_items.len() == second_items.len()
                && first_items
                    .iter()
                    .zip(second_items)
                    .all(|(a, b)| close(*a, *b))
        }
        _ => false,
    }
}

/// The numeric reading of `answer`, `None` outside the grammar.
pub fn read(answer: &str) -> Option<Reading> {
    let text = prepared(answer);
    let chars: Vec<char> = text.chars().collect();
    let (open, inner, close) = match chars.as_slice() {
        [open @ ('(' | '['), inner @ .., close @ (')' | ']')] if !inner.is_empty() => {
            (*open, inner, *close)
        }
        _ => (' ', chars.as_slice(), ' '),
    };
    let parts = top_level_parts(inner);
    if let [only] = parts.as_slice() {
        // Without a comma there is one value; brackets around it only group.
        let whole = if open == ' ' {
            only.as_slice()
        } else {
            chars.as_slice()
        };
        return number(whole).map(Reading::Number);
    }
    let items: Option<Vec<f64>> = parts.iter().map(|part| number(part)).collect();
    Some(Reading::List {
        open,
        close,
        items: items?,
    })
}

/// The answer with layout, delimiters, units and degree marks removed and
/// the operator commands written as operators.
fn prepared(answer: &str) -> String {
    let mut text = answer.to_owned();
    for (from, to) in [
        ("\\left", ""),
        ("\\right", ""),
        ("\\displaystyle", ""),
        ("\\!", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("\\:", ""),
        ("\\ ", ""),
        ("\\$", ""),
        ("$", ""),
        ("^{\\circ}", ""),
        ("^\\circ", ""),
        ("\\%", "%"),
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
        ("\\cdot", "*"),
        ("\\times", "*"),
        ("\\div", "/"),
    ] {
        text = text.replace(from, to);
    }
    if let Some((before, _)) = text.split_once("\\text{") {
        text = before.to_owned();
    }
    text.chars().filter(|ch| !ch.is_whitespace()).collect()
}

/// `chars` split at the commas outside any bracket or brace.
fn top_level_parts(chars: &[char]) -> Vec<Vec<char>> {
    let mut parts = vec![Vec::new()];
    let mut depth: Vec<char> = Vec::new();
    for &ch in chars {
        match ch {
            '(' | '[' | '{' => depth.push(ch),
            ')' | ']' | '}' => {
                depth.pop();
            }
            ',' if depth.is_empty() => {
                parts.push(Vec::new());
                continue;
            }
            _ => {}
        }
        if let Some(part) = parts.last_mut() {
            part.push(ch);
        }
    }
    parts
}

fn number(chars: &[char]) -> Option<f64> {
    let mut parser = Parser { rest: chars };
    let value = parser.expression()?;
    (parser.rest.is_empty() && value.is_finite()).then_some(value)
}

/// A recursive-descent reader over what is left of the answer.
struct Parser<'a> {
    rest: &'a [char],
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<char> {
        self.rest.first().copied()
    }

    fn bump(&mut self) {
        if let Some((_, tail)) = self.rest.split_first() {
            self.rest = tail;
        }
    }

    fn eat(&mut self, ch: char) -> bool {
        let found = self.peek() == Some(ch);
        if found {
            self.bump();
        }
        found
    }

    /// The leading run of characters `keep` accepts, consumed.
    fn take(&mut self, keep: impl Fn(char) -> bool) -> &'a [char] {
        let length = self.rest.iter().take_while(|ch| keep(**ch)).count();
        let (taken, tail) = self.rest.split_at(length);
        self.rest = tail;
        taken
    }

    fn expression(&mut self) -> Option<f64> {
        let mut value = self.term()?;
        loop {
            if self.eat('+') {
                value += self.term()?;
            } else if self.eat('-') {
                value -= self.term()?;
            } else {
                return Some(value);
            }
        }
    }

    fn term(&mut self) -> Option<f64> {
        let mut value = self.unary()?;
        loop {
            if self.eat('*') {
                value *= self.unary()?;
            } else if self.eat('/') {
                value /= self.unary()?;
            } else if matches!(self.peek(), Some(ch) if ch.is_ascii_digit() || matches!(ch, '.' | '(' | '{' | '\\'))
            {
                value *= self.power()?;
            } else {
                return Some(value);
            }
        }
    }

    fn unary(&mut self) -> Option<f64> {
        if self.eat('-') {
            return self.unary().map(|value| -value);
        }
        if self.eat('+') {
            return self.unary();
        }
        self.power()
    }

    fn power(&mut self) -> Option<f64> {
        let mut base = self.atom()?;
        if self.eat('%') {
            base /= PER_CENT;
        }
        if !self.eat('^') {
            return Some(base);
        }
        // LaTeX raises to one character unless braces group more.
        let exponent = match self.peek() {
            Some('{') => self.group()?,
            Some(ch) if ch.is_ascii_digit() => {
                self.bump();
                ch.to_string().parse().ok()?
            }
            Some('-') => {
                self.bump();
                -self.atom()?
            }
            _ => self.atom()?,
        };
        Some(base.powf(exponent))
    }

    fn group(&mut self) -> Option<f64> {
        if !self.eat('{') {
            return None;
        }
        let value = self.expression()?;
        self.eat('}').then_some(value)
    }

    fn atom(&mut self) -> Option<f64> {
        match self.peek()? {
            '(' => {
                self.bump();
                let value = self.expression()?;
                self.eat(')').then_some(value)
            }
            '{' => self.group(),
            '\\' => {
                self.bump();
                self.command()
            }
            ch if ch.is_ascii_digit() || ch == '.' => self
                .take(|ch| ch.is_ascii_digit() || ch == '.')
                .iter()
                .collect::<String>()
                .parse()
                .ok(),
            _ => None,
        }
    }

    fn command(&mut self) -> Option<f64> {
        let name: String = self.take(|ch| ch.is_ascii_alphabetic()).iter().collect();
        match name.as_str() {
            "frac" => {
                let numerator = self.group()?;
                Some(numerator / self.group()?)
            }
            "sqrt" if self.eat('[') => {
                let index = self.expression()?;
                if !self.eat(']') {
                    return None;
                }
                Some(self.group()?.powf(index.recip()))
            }
            "sqrt" => Some(self.group()?.sqrt()),
            "pi" => Some(std::f64::consts::PI),
            "ln" => self.power().map(f64::ln),
            "exp" => self.power().map(f64::exp),
            "sin" => self.power().map(f64::sin),
            "cos" => self.power().map(f64::cos),
            "tan" => self.power().map(f64::tan),
            _ => None,
        }
    }
}
