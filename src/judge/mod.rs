//! The judge a judged evaluator asks: a model reached through Brama, the
//! gateway every Wisent model call goes through, never through a provider
//! SDK. One synchronous chat completion per evaluation.
//!
//! The gateway's base URL and this program's bearer come from `BRAMA_URL`
//! and `BRAMA_BEARER`, which whoever runs the evaluators exports; the route,
//! the token budget and the temperature come from the request's options,
//! because each is the caller's decision.

use std::sync::LazyLock;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::Options;

pub const URL_VAR: &str = "BRAMA_URL";
pub const BEARER_VAR: &str = "BRAMA_BEARER";

/// The options every judged evaluator reads, with what each decides.
pub const OPTIONS: &[(&str, &str)] = &[
    (
        "judge_model",
        "the Brama route that judges: an alias, a provider/model route or a selector",
    ),
    ("judge_max_tokens", "the judge's answer budget in tokens"),
    ("judge_temperature", "the judge's sampling temperature"),
];

/// One connection pool for every request a run evaluates.
static AGENT: LazyLock<ureq::Agent> = LazyLock::new(|| ureq::AgentBuilder::new().build());

/// Asks the judge the request's options name and returns its answer,
/// trimmed.
pub fn ask(options: &Options<'_>, prompt: &str) -> Result<String> {
    let model = options.text("judge_model")?;
    let max_tokens = options.number("judge_max_tokens")?;
    let temperature = options.number("judge_temperature")?;
    let base = variable(URL_VAR)?;
    let base = base.trim_end_matches('/');
    let bearer = variable(BEARER_VAR)?;
    let body = json!({
        "model": model,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": max_tokens,
        "temperature": temperature,
    })
    .to_string();
    let response = AGENT
        .post(&format!("{base}/v1/chat/completions"))
        .set("authorization", &format!("Bearer {bearer}"))
        .set("content-type", "application/json")
        .send_string(&body);
    let response = match response {
        Ok(response) => response,
        Err(ureq::Error::Status(status, response)) => {
            let text = response
                .into_string()
                .context("Brama refused the judge call and its body could not be read")?;
            bail!(
                "Brama refused the judge call with {status}: {}",
                refusal(&text)
            );
        }
        Err(ureq::Error::Transport(transport)) => {
            bail!("failed to reach Brama at {base}: {}", transport.kind());
        }
    };
    let text = response
        .into_string()
        .context("failed to read Brama's judge answer")?;
    let value: Value = serde_json::from_str(&text).context("Brama's judge answer is not JSON")?;
    match value
        .pointer("/choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.pointer("/message/content"))
        .and_then(Value::as_str)
    {
        Some(content) => Ok(content.trim().to_owned()),
        None => bail!("Brama's judge answer carried no assistant message"),
    }
}

/// A gateway variable, refused by name when unset or empty; its value is
/// never echoed, since the bearer is one of them.
fn variable(name: &str) -> Result<String> {
    match std::env::var(name) {
        Ok(value) if !value.trim().is_empty() => Ok(value.trim().to_owned()),
        _ => bail!("{name} is unset or empty; export it before running a judged evaluator"),
    }
}

/// Brama's own sentence from its error envelope, else the body as sent.
fn refusal(body: &str) -> String {
    let message = serde_json::from_str::<Value>(body).ok().and_then(|value| {
        value
            .pointer("/error/message")
            .and_then(Value::as_str)
            .map(str::to_owned)
    });
    match message {
        Some(message) => message,
        None => body.trim().to_owned(),
    }
}
