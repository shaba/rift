//! Live checks against the real TypeSafe API. Skipped unless TYPESAFE_API_KEY
//! is set (Jev is early access, so CI and most checkouts have no key):
//!
//!   TYPESAFE_API_KEY=sk-... cargo test -p rift-typesafe --test live -- --nocapture
//!
//! RIFT_LIVE_TYPESAFE_MODEL pins a model (default: jev-latest).
//!
//! Assertions are protocol-level — that the documented fields come back with
//! sane values. Model judgement varies, so nothing here asserts a particular
//! answer; the one semantic check uses a question with an unambiguous answer.

use rift_typesafe::{Question, TypeSafeClient, DEFAULT_BASE_URL};
use serde_json::json;

fn live_client() -> Option<TypeSafeClient> {
    let key = std::env::var("TYPESAFE_API_KEY").ok().filter(|k| !k.trim().is_empty())?;
    let model =
        std::env::var("RIFT_LIVE_TYPESAFE_MODEL").unwrap_or_else(|_| "jev-latest".to_string());
    let base = std::env::var("RIFT_LIVE_TYPESAFE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
    Some(TypeSafeClient::new(base, key, model))
}

macro_rules! client_or_skip {
    () => {
        match live_client() {
            Some(c) => c,
            None => {
                eprintln!("skipped: set TYPESAFE_API_KEY to run the live TypeSafe checks");
                return;
            }
        }
    };
}

#[tokio::test]
async fn live_model_listing() {
    let client = client_or_skip!();
    let models = client.models().await.expect("GET /v1/models");
    assert!(!models.is_empty(), "account lists no models");
    eprintln!("models: {:?}", models.iter().map(|m| &m.name).collect::<Vec<_>>());
}

#[tokio::test]
async fn live_all_three_primitives_in_one_call() {
    let client = client_or_skip!();
    let state = json!({
        "ticket": "My payouts have been failing for three days and nobody has replied.",
        "plan": "enterprise"
    });
    let questions = vec![
        ("urgent".to_string(), Question::noul("Does this convey urgency?")),
        (
            "team".to_string(),
            Question::choice(
                "Which team should handle this?",
                vec![
                    ("billing".to_string(), Some("Payments, invoicing, refunds".into())),
                    ("technical".to_string(), Some("Bugs, outages, integrations".into())),
                    ("sales".to_string(), Some("Pricing, upgrades, new accounts".into())),
                ],
            ),
        ),
        (
            "frustration".to_string(),
            Question::score(
                "How frustrated is the customer?",
                vec!["Calm".into(), "Frustrated".into(), "Very angry".into()],
            ),
        ),
    ];

    let ev = client.evaluate(&state, &questions).await.expect("evaluate");
    eprintln!("model={} usage={:?}", ev.model, ev.usage);

    // The response reports the versioned model that actually answered.
    assert!(!ev.model.is_empty());

    let urgent = ev.require("urgent").expect("noul answer");
    let p = urgent.as_noul().expect("noul primitive");
    assert!((0.0..=1.0).contains(&p), "noul out of range: {p}");
    assert_eq!(urgent.confidence(), None, "nouls report no confidence");
    // Unambiguous enough to assert: this ticket is urgent.
    assert!(p > 0.5, "expected an urgent reading, got {p}");

    let team = ev.require("team").expect("choice answer");
    let picked = team.as_choice().expect("choice primitive");
    assert!(
        ["billing", "technical", "sales"].contains(&picked),
        "choice returned an option that was never offered: {picked}"
    );
    let conf = team.confidence().expect("choice reports confidence");
    assert!((0.0..=1.0).contains(&conf), "confidence out of range: {conf}");
    let total: f64 = team.ranked().iter().map(|(_, p)| p).sum();
    assert!((total - 1.0).abs() < 0.05, "probabilities should sum to ~1, got {total}");

    let frustration = ev.require("frustration").expect("score answer");
    let s = frustration.as_score().expect("score primitive");
    assert!((0.0..=2.0).contains(&s), "score outside the level range: {s}");
    assert!(frustration.confidence().is_some());

    assert!(ev.usage.input_tokens > 0, "input tokens should be reported");
}

// The shape the swarm judge sends: structured program state, a choice over
// names, and a noul gate in the same call.
#[tokio::test]
async fn live_judge_shaped_call() {
    let client = client_or_skip!();
    let state = json!({
        "task": "Fix the off-by-one in the pagination offset.",
        "candidates": [
            {"name": "a", "status": "changed", "diff_stat": "1 file changed, 1 insertion(+), 1 deletion(-)",
             "patch": "@@\n- let start = page * size + 1;\n+ let start = page * size;\n"},
            {"name": "b", "status": "changed", "diff_stat": "1 file changed, 40 insertions(+)",
             "patch": "@@\n+ // TODO: rewrite the whole paginator\n+ fn unrelated() {}\n"}
        ]
    });
    let questions = vec![
        (
            "winner".to_string(),
            Question::choice(
                "Which candidate's diff best accomplishes the task? Judge the diffs only: \
                 correctness first, then prefer the smaller, cleaner change.",
                vec![("a".to_string(), None), ("b".to_string(), None)],
            ),
        ),
        (
            "any_correct".to_string(),
            Question::noul("Does at least one candidate's diff correctly accomplish the task?"),
        ),
    ];
    let ev = client.evaluate(&state, &questions).await.expect("evaluate");
    let winner = ev.require("winner").unwrap().as_choice().unwrap().to_string();
    assert!(["a", "b"].contains(&winner.as_str()));
    eprintln!(
        "winner={winner} confidence={:?} any_correct={:?}",
        ev.require("winner").unwrap().confidence(),
        ev.require("any_correct").unwrap().as_noul()
    );
    // `a` is the actual fix; `b` changes nothing relevant.
    assert_eq!(winner, "a", "expected the real fix to win");
}

// A bad key must come back as a clean auth error, not a retry storm.
#[tokio::test]
async fn live_bad_key_is_an_auth_error() {
    if live_client().is_none() {
        eprintln!("skipped: set TYPESAFE_API_KEY to run the live TypeSafe checks");
        return;
    }
    let bad = TypeSafeClient::new(DEFAULT_BASE_URL, "sk-definitely-not-a-real-key", "jev-latest");
    let e = bad
        .ask(&json!("hello"), "q", Question::noul("Is this a test?"))
        .await
        .expect_err("a bogus key must fail");
    assert!(matches!(e, rift_typesafe::Error::Auth(_)), "got {e:?}");
}
