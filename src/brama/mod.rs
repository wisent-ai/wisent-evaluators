//! The models an evaluator consults, reached through Brama, the gateway
//! every Wisent model call goes through, never through a provider SDK: the
//! judge a judged evaluator asks (one chat completion per evaluation) and the
//! embedding model a similarity evaluator compares with.
//!
//! The gateway's base URL and this program's bearer come from `BRAMA_URL`
//! and `BRAMA_BEARER`, which whoever runs the evaluators exports; the routes,
//! the token budget and the temperature come from the request's options,
//! because each is the caller's decision.

use std::sync::LazyLock;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::Options;

pub const URL_VAR: &str = "BRAMA_URL";
pub const BEARER_VAR: &str = "BRAMA_BEARER";

/// The options every judged evaluator reads, with what each decides.
pub const JUDGE_OPTIONS: &[(&str, &str)] = &[
    (
        "judge_model",
        "the Brama route that judges: an alias, a provider/model route or a selector",
    ),
    ("judge_max_tokens", "the judge's answer budget in tokens"),
    ("judge_temperature", "the judge's sampling temperature"),
];

/// The option naming the embedding model, read by every similarity
/// evaluator.
pub const EMBEDDING_OPTION: (&str, &str) = (
    "embedding_model",
    "the embedding model Brama's /v1/embeddings serves; required",
);

/// One connection pool for every request a run evaluates.
static AGENT: LazyLock<ureq::Agent> = LazyLock::new(|| ureq::AgentBuilder::new().build());

/// The one property a judge's answer holds: the label it chose.
const LABEL: &str = "label";

/// The function the judge answers through: its one parameter is the label.
const JUDGE_FUNCTION: &str = "judge_label";

/// Asks the judge the request's options name to choose one of `labels`,
/// as one required call of `judge_label` whose only parameter is the label
/// (strict), and returns the call's arguments as the judge sent them,
/// trimmed: the evaluator reads the label from them and keeps the whole
/// answer. An answer without exactly that call is refused with it.
pub fn judge(options: &Options<'_>, prompt: &str, labels: &[&str]) -> Result<String> {
    let body = json!({
        "model": options.text("judge_model")?,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": options.number("judge_max_tokens")?,
        "temperature": options.number("judge_temperature")?,
        "tools": [{
            "type": "function",
            "function": {
                "name": JUDGE_FUNCTION,
                "description": "The label the judge chose.",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "properties": { LABEL: { "type": "string", "enum": labels } },
                    "required": [LABEL],
                    "additionalProperties": false,
                },
            },
        }],
        "tool_choice": { "type": "function", "function": { "name": JUDGE_FUNCTION } },
    });
    let value = post("/v1/chat/completions", &body, "judge call")?;
    let calls = value
        .pointer("/choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.pointer("/message/tool_calls"))
        .and_then(Value::as_array);
    let Some([call]) = calls.map(Vec::as_slice) else {
        bail!("Brama's judge did not answer by exactly one call of {JUDGE_FUNCTION}: {value}");
    };
    if call.pointer("/function/name").and_then(Value::as_str) != Some(JUDGE_FUNCTION) {
        bail!("Brama's judge called a function other than {JUDGE_FUNCTION}: {value}");
    }
    match call.pointer("/function/arguments").and_then(Value::as_str) {
        Some(arguments) => Ok(arguments.trim().to_owned()),
        None => bail!("Brama's judge call of {JUDGE_FUNCTION} carried no arguments: {value}"),
    }
}

/// Which of `labels` a judge's `answer` chose, by its position, when the
/// answer is the object the schema asks for and its label is one of them;
/// otherwise why it is not.
pub fn chosen(answer: &str, labels: &[&str]) -> Result<usize, String> {
    let value = serde_json::from_str::<Value>(answer).map_err(|error| {
        format!("the answer is not the label object the schema asks for: {error}")
    })?;
    let Some(said) = value.get(LABEL).and_then(Value::as_str) else {
        return Err(format!("the answer names no `{LABEL}`"));
    };
    match labels.iter().position(|label| *label == said) {
        Some(index) => Ok(index),
        None => Err(format!("{said:?} is none of {}", labels.join(", "))),
    }
}

/// The embedding of each text, in order, from the model the request's
/// options name.
pub fn embed(options: &Options<'_>, texts: &[&str]) -> Result<Vec<Vec<f64>>> {
    let body = json!({
        "model": options.text(EMBEDDING_OPTION.0)?,
        "input": texts,
    });
    let value = post("/v1/embeddings", &body, "embedding call")?;
    let Some(data) = value.pointer("/data").and_then(Value::as_array) else {
        bail!("Brama's embedding answer carried no data list");
    };
    if data.len() != texts.len() {
        bail!(
            "Brama answered {} embeddings for {} texts",
            data.len(),
            texts.len()
        );
    }
    let mut embeddings: Vec<Option<Vec<f64>>> = vec![None; texts.len()];
    for item in data {
        let Some(index) = item.pointer("/index").and_then(Value::as_u64) else {
            bail!("an embedding in Brama's answer carries no index");
        };
        let index = usize::try_from(index).with_context(|| {
            format!("embedding index {index} in Brama's answer is out of range")
        })?;
        let Some(vector) = item.pointer("/embedding").and_then(Value::as_array) else {
            bail!("embedding {index} in Brama's answer carries no vector");
        };
        let vector: Option<Vec<f64>> = vector.iter().map(Value::as_f64).collect();
        match (vector, embeddings.get_mut(index)) {
            (Some(vector), Some(slot)) => *slot = Some(vector),
            (None, _) => bail!("embedding {index} in Brama's answer holds a non-number"),
            (_, None) => bail!(
                "Brama's answer names embedding {index} for {} texts",
                texts.len()
            ),
        }
    }
    match embeddings.into_iter().collect::<Option<Vec<_>>>() {
        Some(embeddings) => Ok(embeddings),
        None => bail!("Brama's answer repeats an embedding index and leaves another out"),
    }
}

/// The cosine similarity of two vectors; `None` when either has no length.
pub fn cosine(first: &[f64], second: &[f64]) -> Option<f64> {
    let dot: f64 = first.iter().zip(second).map(|(a, b)| a * b).sum();
    let norm = |vector: &[f64]| vector.iter().map(|x| x * x).sum::<f64>().sqrt();
    let scale = norm(first) * norm(second);
    (scale > f64::EPSILON).then(|| dot / scale)
}

fn post(path: &str, body: &Value, what: &str) -> Result<Value> {
    let base = variable(URL_VAR)?;
    let base = base.trim_end_matches('/');
    let bearer = variable(BEARER_VAR)?;
    let response = AGENT
        .post(&format!("{base}{path}"))
        .set("authorization", &format!("Bearer {bearer}"))
        .set("content-type", "application/json")
        .send_string(&body.to_string());
    let response = match response {
        Ok(response) => response,
        Err(ureq::Error::Status(status, response)) => {
            let text = response.into_string().with_context(|| {
                format!("Brama refused the {what} and its body could not be read")
            })?;
            bail!("Brama refused the {what} with {status}: {}", refusal(&text));
        }
        Err(ureq::Error::Transport(transport)) => {
            bail!("failed to reach Brama at {base}: {}", transport.kind());
        }
    };
    let text = response
        .into_string()
        .with_context(|| format!("failed to read Brama's answer to the {what}"))?;
    serde_json::from_str(&text).with_context(|| format!("Brama's answer to the {what} is not JSON"))
}

/// A gateway variable, refused by name when unset or empty; its value is
/// never echoed, since the bearer is one of them.
fn variable(name: &str) -> Result<String> {
    match std::env::var(name) {
        Ok(value) if !value.trim().is_empty() => Ok(value.trim().to_owned()),
        _ => bail!(
            "{name} is unset or empty; export it before running an evaluator that consults a model"
        ),
    }
}

/// Brama's own sentence from its error envelope, else the body as sent,
/// saying why it holds no sentence.
fn refusal(body: &str) -> String {
    match serde_json::from_str::<Value>(body) {
        Ok(value) => match value.pointer("/error/message").and_then(Value::as_str) {
            Some(message) => message.to_owned(),
            None => format!("{} (an error envelope with no error.message)", body.trim()),
        },
        Err(error) => format!("{} (not a JSON error envelope: {error})", body.trim()),
    }
}
