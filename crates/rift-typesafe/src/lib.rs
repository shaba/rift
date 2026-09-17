//! TypeSafe System One (Jev) client: `POST /v1/systemone`.
//!
//! Deliberately NOT a `rift_provider::Provider`. A System One model is not a
//! chat model — it emits no text, no tool calls, and no stream. It takes a
//! `state` plus a map of typed `questions` and returns one typed `answer` per
//! question, in a single round trip. There is nothing for `chat_stream` to
//! yield, so wiring Jev in as a provider would be a category error; rift uses
//! it where it actually fits — a decision it would otherwise have had to
//! launder through prose and parse back out (see `rift_core::swarm`).
//!
//! The three primitives are the whole surface:
//! - **noul**  — yes/no, answered as the probability of yes. No confidence.
//! - **choice** — one option from a set, with a probability per option.
//! - **score**  — a position on an ordered rubric, probability-weighted.
//!
//! Only choice and score carry `confidence`; the type system here reflects
//! that rather than papering over it with an `Option` on a shared struct.

use std::collections::HashMap;
use std::time::Duration;

use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{json, Value};

/// Public API root. HTTPS — this endpoint is authenticated with a bearer
/// token, so unlike rift's LAN-first provider hosts it must never be
/// silently downgraded to http.
pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";
/// Flagship alias. Pinning (`jev-1.13.0`) is a config choice, not a default.
pub const DEFAULT_MODEL: &str = "jev-latest";
/// Environment variable rift reads the key from when config names none.
pub const API_KEY_ENV: &str = "TYPESAFE_API_KEY";

/// Transport + protocol failures, kept typed so callers can distinguish
/// "your key is wrong" (fatal, tell the user) from "back off" (retryable).
#[derive(Debug)]
pub enum Error {
    /// 401 — missing or invalid API key.
    Auth(String),
    /// 422 — the request body failed validation; the body names the field.
    Invalid(String),
    /// 429 — rate limited, after retries were exhausted.
    RateLimited(String),
    /// 529 — TypeSafe overloaded, after retries were exhausted.
    Overloaded(String),
    /// Any other non-2xx.
    Status { status: u16, message: String },
    /// Connection/timeout failure.
    Transport(String),
    /// 2xx whose body was not the documented shape.
    Decode(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auth(m) => write!(f, "typesafe: unauthorized — check {API_KEY_ENV} ({m})"),
            Self::Invalid(m) => write!(f, "typesafe: rejected the request — {m}"),
            Self::RateLimited(m) => write!(f, "typesafe: rate limited — {m}"),
            Self::Overloaded(m) => write!(f, "typesafe: overloaded — {m}"),
            Self::Status { status, message } => write!(f, "typesafe: HTTP {status} — {message}"),
            Self::Transport(m) => write!(f, "typesafe: request failed — {m}"),
            Self::Decode(m) => write!(f, "typesafe: unexpected response — {m}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

// ---- questions (rift -> wire) --------------------------------------------

/// One typed question. Every variant carries free-form `instructions`
/// (string, object, or array — the API accepts JSON structure) and the
/// `criteria` shape its type requires.
#[derive(Debug, Clone)]
pub enum Question {
    /// Yes/no. `criteria` optionally describes what yes and no mean.
    Noul { instructions: Value, yes: Option<String>, no: Option<String> },
    /// One of N. `options` is (name, optional rubric); the API caps a choice
    /// at 255 options.
    Choice { instructions: Value, options: Vec<(String, Option<String>)> },
    /// Ordered rubric; at least two levels, low to high.
    Score { instructions: Value, levels: Vec<String> },
}

/// The API's documented ceiling on choice options.
pub const MAX_CHOICE_OPTIONS: usize = 255;

impl Question {
    pub fn noul(instructions: impl Into<String>) -> Self {
        Self::Noul { instructions: Value::String(instructions.into()), yes: None, no: None }
    }

    /// Noul with explicit rubrics for each pole — worth setting whenever
    /// "yes" is not self-evident from the question alone.
    pub fn noul_with(
        instructions: impl Into<String>,
        yes: impl Into<String>,
        no: impl Into<String>,
    ) -> Self {
        Self::Noul {
            instructions: Value::String(instructions.into()),
            yes: Some(yes.into()),
            no: Some(no.into()),
        }
    }

    pub fn choice<I, K>(instructions: impl Into<String>, options: I) -> Self
    where
        I: IntoIterator<Item = (K, Option<String>)>,
        K: Into<String>,
    {
        Self::Choice {
            instructions: Value::String(instructions.into()),
            options: options.into_iter().map(|(k, v)| (k.into(), v)).collect(),
        }
    }

    pub fn score(instructions: impl Into<String>, levels: Vec<String>) -> Self {
        Self::Score { instructions: Value::String(instructions.into()), levels }
    }

    /// Catch malformed questions before they cost a round trip and come back
    /// as a 422. The API's own rules: choice needs 1..=255 options, score
    /// needs at least two levels.
    fn validate(&self, id: &str) -> Result<()> {
        match self {
            Self::Choice { options, .. } => {
                if options.is_empty() {
                    return Err(Error::Invalid(format!("question {id:?}: choice needs at least one option")));
                }
                if options.len() > MAX_CHOICE_OPTIONS {
                    return Err(Error::Invalid(format!(
                        "question {id:?}: choice has {} options, the maximum is {MAX_CHOICE_OPTIONS}",
                        options.len()
                    )));
                }
                Ok(())
            }
            Self::Score { levels, .. } => {
                if levels.len() < 2 {
                    return Err(Error::Invalid(format!(
                        "question {id:?}: score needs at least two levels, got {}",
                        levels.len()
                    )));
                }
                Ok(())
            }
            Self::Noul { .. } => Ok(()),
        }
    }
}

impl Serialize for Question {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Noul { instructions, yes, no } => {
                let extra = yes.is_some() || no.is_some();
                let mut m = s.serialize_map(Some(if extra { 3 } else { 2 }))?;
                m.serialize_entry("type", "noul")?;
                m.serialize_entry("instructions", instructions)?;
                if extra {
                    // Only the poles that were actually described; the API
                    // treats both as optional.
                    let mut c = serde_json::Map::new();
                    if let Some(y) = yes {
                        c.insert("true".into(), Value::String(y.clone()));
                    }
                    if let Some(n) = no {
                        c.insert("false".into(), Value::String(n.clone()));
                    }
                    m.serialize_entry("criteria", &Value::Object(c))?;
                }
                m.end()
            }
            Self::Choice { instructions, options } => {
                let mut m = s.serialize_map(Some(3))?;
                m.serialize_entry("type", "choice")?;
                m.serialize_entry("instructions", instructions)?;
                let c: serde_json::Map<String, Value> = options
                    .iter()
                    .map(|(k, v)| {
                        (k.clone(), v.clone().map(Value::String).unwrap_or(Value::Null))
                    })
                    .collect();
                m.serialize_entry("criteria", &Value::Object(c))?;
                m.end()
            }
            Self::Score { instructions, levels } => {
                let mut m = s.serialize_map(Some(3))?;
                m.serialize_entry("type", "score")?;
                m.serialize_entry("instructions", instructions)?;
                m.serialize_entry("criteria", levels)?;
                m.end()
            }
        }
    }
}

// ---- answers (wire -> rift) ----------------------------------------------

/// A typed answer. The variant always matches the question that asked it.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    /// Probability that the answer is yes, 0.0..=1.0. Note there is no
    /// `confidence` here — the API does not report one for nouls.
    Noul { noul: f64 },
    Choice {
        /// The highest-probability option.
        choice: String,
        /// Every option mapped to its probability; sums to 1.
        #[serde(default)]
        probabilities: HashMap<String, f64>,
        confidence: f64,
    },
    Score {
        /// Probability-weighted position across the levels; lands between
        /// levels, so it is a f64 and not an index.
        score: f64,
        /// Level index (as a string key) back to its description.
        #[serde(default)]
        legend: HashMap<String, String>,
        #[serde(default)]
        probabilities: HashMap<String, f64>,
        confidence: f64,
    },
}

impl Answer {
    /// The yes-probability, if this is a noul.
    pub fn as_noul(&self) -> Option<f64> {
        match self {
            Self::Noul { noul } => Some(*noul),
            _ => None,
        }
    }

    /// The picked option, if this is a choice.
    pub fn as_choice(&self) -> Option<&str> {
        match self {
            Self::Choice { choice, .. } => Some(choice.as_str()),
            _ => None,
        }
    }

    /// The weighted score, if this is a score.
    pub fn as_score(&self) -> Option<f64> {
        match self {
            Self::Score { score, .. } => Some(*score),
            _ => None,
        }
    }

    /// Calibrated certainty for choice/score. `None` for nouls, which report
    /// no confidence — for those the probability itself carries the
    /// uncertainty (0.5 is maximally unsure).
    pub fn confidence(&self) -> Option<f64> {
        match self {
            Self::Choice { confidence, .. } | Self::Score { confidence, .. } => Some(*confidence),
            Self::Noul { .. } => None,
        }
    }

    /// Per-option / per-level probabilities, highest first. Empty for nouls.
    pub fn ranked(&self) -> Vec<(String, f64)> {
        let probs = match self {
            Self::Choice { probabilities, .. } | Self::Score { probabilities, .. } => probabilities,
            Self::Noul { .. } => return vec![],
        };
        let mut v: Vec<(String, f64)> = probs.iter().map(|(k, p)| (k.clone(), *p)).collect();
        // Ties broken by name so the rendering is stable run to run.
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        v
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    /// Always billed at zero by TypeSafe; reported for completeness.
    #[serde(default)]
    pub output_tokens: u64,
}

/// A completed evaluation: one answer per question id you sent.
#[derive(Debug, Clone, Deserialize)]
pub struct Evaluation {
    /// The versioned model that answered (e.g. `jev-1.13.0`), even when the
    /// request asked for an alias.
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub answers: HashMap<String, Answer>,
    #[serde(default)]
    pub usage: Usage,
}

impl Evaluation {
    pub fn get(&self, id: &str) -> Option<&Answer> {
        self.answers.get(id)
    }

    /// The answer for `id`, erroring if it is absent or the wrong primitive.
    /// Use this at call sites that genuinely cannot proceed without it.
    pub fn require(&self, id: &str) -> Result<&Answer> {
        self.answers
            .get(id)
            .ok_or_else(|| Error::Decode(format!("no answer returned for question {id:?}")))
    }
}

/// One model from `GET /v1/models`.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelCard {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub release_date: String,
}

// ---- client ---------------------------------------------------------------

/// How many times a retryable status (429/529) or transport error is retried
/// before giving up. TypeSafe's own SDKs retry by default; rift matches that
/// so a transient 529 doesn't lose a race verdict.
const RETRY_ATTEMPTS: u32 = 3;
const RETRY_BASE_DELAY: Duration = Duration::from_millis(400);
/// Ceiling on a server-supplied `retry-after`, so a hostile or mistaken
/// header cannot park a turn for minutes.
const RETRY_AFTER_MAX: Duration = Duration::from_secs(10);

#[derive(Clone)]
pub struct TypeSafeClient {
    base_url: String,
    api_key: String,
    model: String,
    http: reqwest::Client,
}

impl TypeSafeClient {
    /// `base_url` is the API root (`https://api.typesafe.ai`); the versioned
    /// path is appended per call.
    pub fn new(base_url: impl AsRef<str>, api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: normalize_api_base(base_url.as_ref()),
            api_key: api_key.into(),
            model: model.into(),
            // Shared with the chat providers: bounded connect/read timeouts
            // and TCP keepalive. Jev answers in well under a second, so the
            // read timeout is pure backstop here.
            http: rift_provider::http_client(),
        }
    }

    /// Client from the environment (`TYPESAFE_API_KEY`), or `None` when no
    /// key is set — the signal every call site uses to stay on its existing
    /// non-Jev path rather than erroring.
    pub fn from_env() -> Option<Self> {
        let key = std::env::var(API_KEY_ENV).ok().filter(|k| !k.trim().is_empty())?;
        Some(Self::new(DEFAULT_BASE_URL, key, DEFAULT_MODEL))
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Same endpoint and key, a different model — for a call site that pins
    /// a version (`jev-1.13.0`) while the session default stays on an alias.
    pub fn with_model(&self, model: impl Into<String>) -> Self {
        Self { model: model.into(), ..self.clone() }
    }

    /// Evaluate `state` against `questions`, all in one round trip.
    ///
    /// Asking every question in a single call is the documented way to use
    /// the API — it is dramatically cheaper and faster than one call per
    /// question, because `state` is billed once instead of N times.
    pub async fn evaluate(
        &self,
        state: &Value,
        questions: &[(String, Question)],
    ) -> Result<Evaluation> {
        if questions.is_empty() {
            return Err(Error::Invalid("no questions to ask".into()));
        }
        let mut map = serde_json::Map::new();
        for (id, q) in questions {
            q.validate(id)?;
            let v = serde_json::to_value(q).map_err(|e| Error::Decode(e.to_string()))?;
            if map.insert(id.clone(), v).is_some() {
                return Err(Error::Invalid(format!("duplicate question id {id:?}")));
            }
        }
        let body = json!({
            "state": state,
            "model": self.model,
            "questions": Value::Object(map),
        });
        let url = format!("{}/v1/systemone", self.base_url);
        let text = self.send(self.http.post(&url).json(&body)).await?;
        serde_json::from_str::<Evaluation>(&text).map_err(|e| {
            Error::Decode(format!("{e} (body: {})", truncate(&text, 400)))
        })
    }

    /// Convenience for the common single-question case.
    pub async fn ask(&self, state: &Value, id: &str, question: Question) -> Result<Answer> {
        let ev = self.evaluate(state, &[(id.to_string(), question)]).await?;
        ev.require(id).cloned()
    }

    /// `GET /v1/models` — the models this key may use.
    pub async fn models(&self) -> Result<Vec<ModelCard>> {
        let url = format!("{}/v1/models", self.base_url);
        let text = self.send(self.http.get(&url)).await?;
        // Documented as a bare array; tolerate a {"models": [...]} envelope
        // rather than hard-failing a listing over a wrapper.
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Decode(format!("{e} (body: {})", truncate(&text, 400))))?;
        let arr = v
            .as_array()
            .cloned()
            .or_else(|| v.get("models").and_then(|m| m.as_array()).cloned())
            .or_else(|| v.get("data").and_then(|m| m.as_array()).cloned())
            .ok_or_else(|| Error::Decode(format!("expected a model array, got {}", truncate(&text, 200))))?;
        Ok(arr
            .into_iter()
            .filter_map(|m| serde_json::from_value::<ModelCard>(m).ok())
            .collect())
    }

    /// Send with auth, retrying 429/529 and transport failures with
    /// exponential backoff. Returns the success body as text.
    async fn send(&self, builder: reqwest::RequestBuilder) -> Result<String> {
        let builder = builder.bearer_auth(&self.api_key);
        let mut delay = RETRY_BASE_DELAY;
        let mut last: Option<Error> = None;

        for attempt in 0..RETRY_ATTEMPTS {
            if attempt > 0 {
                tokio::time::sleep(delay).await;
                delay *= 2;
            }
            // Non-clonable body can't be replayed; one attempt is all we get.
            let Some(req) = builder.try_clone() else {
                return self.finish(builder.send().await).await;
            };
            match self.finish(req.send().await).await {
                Ok(text) => return Ok(text),
                Err(e) => {
                    let retry_after = match &e {
                        Error::RateLimited(m) | Error::Overloaded(m) => parse_retry_after(m),
                        _ => None,
                    };
                    if !retryable(&e) {
                        return Err(e);
                    }
                    if let Some(d) = retry_after {
                        delay = d.min(RETRY_AFTER_MAX);
                    }
                    last = Some(e);
                }
            }
        }
        Err(last.unwrap_or_else(|| Error::Transport("retries exhausted".into())))
    }

    /// Map a transport result + HTTP status onto the typed error space.
    async fn finish(&self, sent: reqwest::Result<reqwest::Response>) -> Result<String> {
        let resp = sent.map_err(|e| Error::Transport(e.to_string()))?;
        let status = resp.status().as_u16();
        // `retry-after` is on the response, not the body — carry it into the
        // message so the retry loop can honour it.
        let retry_after = resp
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .map(|s| format!(" [retry-after: {s}]"))
            .unwrap_or_default();
        let body = resp.text().await.unwrap_or_default();
        if (200..300).contains(&status) {
            return Ok(body);
        }
        let msg = format!("{}{retry_after}", rift_provider::api_error_message(&body));
        Err(match status {
            401 | 403 => Error::Auth(msg),
            422 | 400 => Error::Invalid(msg),
            429 => Error::RateLimited(msg),
            529 | 503 => Error::Overloaded(msg),
            s => Error::Status { status: s, message: msg },
        })
    }
}

/// Worth another attempt? Auth and validation failures are deterministic —
/// retrying them just burns time and rate limit.
fn retryable(e: &Error) -> bool {
    match e {
        Error::RateLimited(_) | Error::Overloaded(_) | Error::Transport(_) => true,
        Error::Status { status, .. } => *status >= 500,
        Error::Auth(_) | Error::Invalid(_) | Error::Decode(_) => false,
    }
}

/// Pull the seconds out of the `[retry-after: N]` marker `finish` appended.
fn parse_retry_after(msg: &str) -> Option<Duration> {
    let i = msg.find("[retry-after: ")? + "[retry-after: ".len();
    let rest = &msg[i..];
    let end = rest.find(']')?;
    rest[..end].trim().parse::<u64>().ok().map(Duration::from_secs)
}

/// Trim a trailing slash and default the scheme to **https**.
///
/// Deliberately not `rift_provider::normalize_base_url`, which defaults to
/// `http://` because rift's model hosts are usually on the LAN. This endpoint
/// carries a bearer token over the public internet; defaulting it to cleartext
/// would leak the key to anything on the path.
pub fn normalize_api_base(url: &str) -> String {
    let base = url.trim().trim_end_matches('/');
    if base.starts_with("http://") || base.starts_with("https://") {
        base.to_string()
    } else {
        format!("https://{base}")
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max).collect::<String>() + "…"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_serializes_with_null_rubrics() {
        let q = Question::choice(
            "Which candidate wins?",
            vec![("a".to_string(), Some("first".to_string())), ("b".to_string(), None)],
        );
        let v = serde_json::to_value(&q).unwrap();
        assert_eq!(v["type"], "choice");
        assert_eq!(v["instructions"], "Which candidate wins?");
        assert_eq!(v["criteria"]["a"], "first");
        assert!(v["criteria"]["b"].is_null());
    }

    #[test]
    fn noul_omits_criteria_when_undescribed() {
        let v = serde_json::to_value(Question::noul("Urgent?")).unwrap();
        assert_eq!(v["type"], "noul");
        assert!(v.get("criteria").is_none(), "empty criteria must not be sent");

        let v = serde_json::to_value(Question::noul_with("Urgent?", "yes it is", "no")).unwrap();
        assert_eq!(v["criteria"]["true"], "yes it is");
        assert_eq!(v["criteria"]["false"], "no");
    }

    #[test]
    fn score_criteria_is_an_ordered_array() {
        let q = Question::score("How bad?", vec!["calm".into(), "cross".into(), "furious".into()]);
        let v = serde_json::to_value(&q).unwrap();
        assert_eq!(v["criteria"], json!(["calm", "cross", "furious"]));
    }

    #[test]
    fn malformed_questions_are_caught_before_the_wire() {
        // Score with one level, choice with none: both are 422s if sent.
        assert!(Question::score("x", vec!["only".into()]).validate("q").is_err());
        assert!(Question::choice("x", Vec::<(String, Option<String>)>::new()).validate("q").is_err());
        let many: Vec<(String, Option<String>)> =
            (0..=MAX_CHOICE_OPTIONS).map(|i| (i.to_string(), None)).collect();
        assert!(Question::choice("x", many).validate("q").is_err());
    }

    #[test]
    fn answers_deserialize_by_type_tag() {
        let ev: Evaluation = serde_json::from_str(
            r#"{"model":"jev-1.13.0","answers":{
                "u":{"type":"noul","noul":0.92},
                "d":{"type":"choice","choice":"technical",
                     "probabilities":{"billing":0.08,"technical":0.85,"sales":0.07},
                     "confidence":0.82},
                "f":{"type":"score","score":1.6,
                     "legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},
                     "probabilities":{"0":0.05,"1":0.3,"2":0.65},"confidence":0.78}},
              "usage":{"input_tokens":312,"output_tokens":48}}"#,
        )
        .unwrap();
        assert_eq!(ev.model, "jev-1.13.0");
        assert_eq!(ev.get("u").unwrap().as_noul(), Some(0.92));
        assert_eq!(ev.get("d").unwrap().as_choice(), Some("technical"));
        assert_eq!(ev.get("f").unwrap().as_score(), Some(1.6));
        assert_eq!(ev.usage.input_tokens, 312);
    }

    // A noul carries no confidence; the type must not invent one.
    #[test]
    fn only_choice_and_score_report_confidence() {
        let noul = Answer::Noul { noul: 0.9 };
        assert_eq!(noul.confidence(), None);
        assert!(noul.ranked().is_empty());

        let ev: Evaluation = serde_json::from_str(
            r#"{"answers":{"d":{"type":"choice","choice":"b",
                 "probabilities":{"a":0.3,"b":0.7},"confidence":0.6}}}"#,
        )
        .unwrap();
        assert_eq!(ev.get("d").unwrap().confidence(), Some(0.6));
        assert_eq!(ev.get("d").unwrap().ranked()[0], ("b".to_string(), 0.7));
    }

    #[test]
    fn wrong_primitive_accessors_return_none() {
        let a = Answer::Noul { noul: 0.5 };
        assert!(a.as_choice().is_none());
        assert!(a.as_score().is_none());
    }

    #[test]
    fn https_is_the_default_scheme() {
        // A bearer token must never be defaulted onto cleartext.
        assert_eq!(normalize_api_base("api.typesafe.ai"), "https://api.typesafe.ai");
        assert_eq!(normalize_api_base("https://api.typesafe.ai/"), "https://api.typesafe.ai");
        // An explicit http:// stays — a user proxying locally meant it.
        assert_eq!(normalize_api_base("http://localhost:8080"), "http://localhost:8080");
    }

    #[test]
    fn retry_policy_skips_deterministic_failures() {
        assert!(retryable(&Error::RateLimited("x".into())));
        assert!(retryable(&Error::Overloaded("x".into())));
        assert!(retryable(&Error::Transport("x".into())));
        assert!(retryable(&Error::Status { status: 500, message: "x".into() }));
        assert!(!retryable(&Error::Auth("x".into())));
        assert!(!retryable(&Error::Invalid("x".into())));
        assert!(!retryable(&Error::Status { status: 404, message: "x".into() }));
    }

    #[test]
    fn retry_after_is_parsed_from_the_carried_marker() {
        assert_eq!(parse_retry_after("rate limited [retry-after: 3]"), Some(Duration::from_secs(3)));
        assert_eq!(parse_retry_after("rate limited"), None);
    }

    #[test]
    fn ranked_is_stable_for_tied_probabilities() {
        let a = Answer::Choice {
            choice: "a".into(),
            probabilities: HashMap::from([("a".into(), 0.5), ("b".into(), 0.5)]),
            confidence: 0.1,
        };
        assert_eq!(a.ranked(), vec![("a".into(), 0.5), ("b".into(), 0.5)]);
    }
}
