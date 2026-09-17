//! Hardening suite for the TypeSafe System One client, run against a mock
//! server so the wire format and every documented failure mode are pinned
//! deterministically. The live-service variant is in `live.rs`.
//!
//! What matters here: rift has no way to re-ask a question cheaply mid-race,
//! so a malformed request, a silently-dropped answer, or a retry that fires
//! on a deterministic 401 all cost a real verdict.

use rift_provider::test_support::{MockResponse, MockServer};
use rift_typesafe::{Answer, Error, Question, TypeSafeClient};
use serde_json::{json, Value};

fn client(server: &MockServer) -> TypeSafeClient {
    TypeSafeClient::new(&server.base_url, "test-key", "jev-latest")
}

/// Parse the JSON body out of a recorded raw HTTP request.
fn body_of(raw: &str) -> Value {
    let (_, body) = raw.split_once("\r\n\r\n").expect("request has a body");
    serde_json::from_str(body).expect("body is JSON")
}

const OK_NOUL: &str = r#"{"model":"jev-1.13.0","answers":{"q":{"type":"noul","noul":0.92}},
                          "usage":{"input_tokens":312,"output_tokens":0}}"#;

// The documented request shape, exactly: state / model / questions, with the
// question keyed by the id the caller chose.
#[tokio::test]
async fn request_matches_the_documented_shape() {
    let server = MockServer::start(vec![MockResponse::json(200, OK_NOUL)]).await;
    client(&server)
        .ask(&json!("Help! My payouts have been failing."), "q", Question::noul("Urgent?"))
        .await
        .expect("evaluate");

    let reqs = server.requests().await;
    assert_eq!(reqs.len(), 1);
    let raw = &reqs[0];
    assert!(raw.starts_with("POST /v1/systemone "), "wrong method/path: {}", &raw[..40]);
    assert!(
        raw.to_lowercase().contains("authorization: bearer test-key"),
        "bearer token missing"
    );
    let body = body_of(raw);
    assert_eq!(body["state"], "Help! My payouts have been failing.");
    assert_eq!(body["model"], "jev-latest");
    assert_eq!(body["questions"]["q"]["type"], "noul");
    assert_eq!(body["questions"]["q"]["instructions"], "Urgent?");
}

// Structured state (an object) must travel as JSON, not stringified — it is
// the form the API is built around, and the form the swarm judge sends.
#[tokio::test]
async fn structured_state_is_sent_as_json() {
    let server = MockServer::start(vec![MockResponse::json(200, OK_NOUL)]).await;
    let state = json!({"task": "fix the bug", "candidates": [{"name": "a", "status": "changed"}]});
    client(&server).ask(&state, "q", Question::noul("Done?")).await.expect("evaluate");

    let body = body_of(&server.requests().await[0]);
    assert!(body["state"].is_object(), "state must stay structured");
    assert_eq!(body["state"]["candidates"][0]["name"], "a");
}

// Every question in one call: the whole point of the fan-out pattern is that
// `state` is billed once, so the client must never split them up.
#[tokio::test]
async fn all_questions_go_in_a_single_request() {
    let body = r#"{"model":"jev-1.13.0","answers":{
        "a":{"type":"noul","noul":0.1},
        "b":{"type":"choice","choice":"x","probabilities":{"x":0.9,"y":0.1},"confidence":0.8},
        "c":{"type":"score","score":1.5,"legend":{"0":"lo","1":"hi"},
             "probabilities":{"0":0.5,"1":0.5},"confidence":0.4}},
        "usage":{"input_tokens":10,"output_tokens":0}}"#;
    let server = MockServer::start(vec![MockResponse::json(200, body)]).await;
    let qs = vec![
        ("a".to_string(), Question::noul("yes?")),
        (
            "b".to_string(),
            Question::choice("pick", vec![("x".to_string(), None), ("y".to_string(), Some("why".into()))]),
        ),
        ("c".to_string(), Question::score("rate", vec!["lo".into(), "hi".into()])),
    ];
    let ev = client(&server).evaluate(&json!("s"), &qs).await.expect("evaluate");

    assert_eq!(server.requests().await.len(), 1, "questions must not be split across calls");
    assert_eq!(ev.get("a").unwrap().as_noul(), Some(0.1));
    assert_eq!(ev.get("b").unwrap().as_choice(), Some("x"));
    assert_eq!(ev.get("c").unwrap().as_score(), Some(1.5));
    // Each question keeps its own criteria shape on the wire.
    let sent = body_of(&server.requests().await[0]);
    assert!(sent["questions"]["b"]["criteria"]["x"].is_null());
    assert_eq!(sent["questions"]["b"]["criteria"]["y"], "why");
    assert_eq!(sent["questions"]["c"]["criteria"], json!(["lo", "hi"]));
}

#[tokio::test]
async fn choice_answer_exposes_distribution_and_confidence() {
    let body = r#"{"model":"jev-1.13.0","answers":{"d":{"type":"choice","choice":"technical",
        "probabilities":{"billing":0.08,"technical":0.85,"sales":0.07},"confidence":0.82}},
        "usage":{"input_tokens":312,"output_tokens":0}}"#;
    let server = MockServer::start(vec![MockResponse::json(200, body)]).await;
    let a = client(&server)
        .ask(&json!("s"), "d", Question::choice("which?", vec![("technical".to_string(), None)]))
        .await
        .expect("evaluate");
    assert_eq!(a.as_choice(), Some("technical"));
    assert_eq!(a.confidence(), Some(0.82));
    assert_eq!(a.ranked()[0], ("technical".to_string(), 0.85));
}

// A noul reports no confidence. Inventing one (e.g. defaulting to 0.0 or
// 1.0) would silently corrupt any confidence gate downstream.
#[tokio::test]
async fn noul_answer_has_no_confidence() {
    let server = MockServer::start(vec![MockResponse::json(200, OK_NOUL)]).await;
    let a = client(&server).ask(&json!("s"), "q", Question::noul("Urgent?")).await.unwrap();
    assert!(matches!(a, Answer::Noul { .. }));
    assert_eq!(a.confidence(), None);
    assert_eq!(a.as_noul(), Some(0.92));
}

// 401 is deterministic: retrying burns the rate limit and delays the error
// the user actually needs to see.
#[tokio::test]
async fn auth_failure_is_not_retried() {
    let server = MockServer::start(vec![
        MockResponse::json(401, r#"{"error":{"message":"invalid api key"}}"#),
        MockResponse::json(200, OK_NOUL),
    ])
    .await;
    let e = client(&server)
        .ask(&json!("s"), "q", Question::noul("x"))
        .await
        .expect_err("should fail");
    assert!(matches!(e, Error::Auth(_)), "got {e:?}");
    assert!(e.to_string().contains("invalid api key"));
    assert_eq!(server.requests().await.len(), 1, "401 must not be retried");
}

#[tokio::test]
async fn validation_failure_surfaces_the_offending_field_and_is_not_retried() {
    let server = MockServer::start(vec![
        MockResponse::json(422, r#"{"error":{"message":"questions.q.criteria: too few levels"}}"#),
        MockResponse::json(200, OK_NOUL),
    ])
    .await;
    let e = client(&server).ask(&json!("s"), "q", Question::noul("x")).await.unwrap_err();
    assert!(matches!(e, Error::Invalid(_)), "got {e:?}");
    assert!(e.to_string().contains("too few levels"));
    assert_eq!(server.requests().await.len(), 1, "422 must not be retried");
}

// 429 and 529 are the two documented "back off and retry" statuses.
#[tokio::test]
async fn rate_limit_is_retried_then_succeeds() {
    let server = MockServer::start(vec![
        MockResponse::json(429, r#"{"error":"slow down"}"#),
        MockResponse::json(200, OK_NOUL),
    ])
    .await;
    let a = client(&server).ask(&json!("s"), "q", Question::noul("x")).await.expect("retry");
    assert_eq!(a.as_noul(), Some(0.92));
    assert_eq!(server.requests().await.len(), 2);
}

#[tokio::test]
async fn overloaded_is_retried_then_succeeds() {
    let server = MockServer::start(vec![
        MockResponse::json(529, r#"{"error":"overloaded"}"#),
        MockResponse::json(200, OK_NOUL),
    ])
    .await;
    let a = client(&server).ask(&json!("s"), "q", Question::noul("x")).await.expect("retry");
    assert_eq!(a.as_noul(), Some(0.92));
    assert_eq!(server.requests().await.len(), 2);
}

#[tokio::test]
async fn exhausted_retries_report_the_last_status() {
    let server = MockServer::start(vec![
        MockResponse::json(529, r#"{"error":"overloaded"}"#),
        MockResponse::json(529, r#"{"error":"overloaded"}"#),
        MockResponse::json(529, r#"{"error":"overloaded"}"#),
    ])
    .await;
    let e = client(&server).ask(&json!("s"), "q", Question::noul("x")).await.unwrap_err();
    assert!(matches!(e, Error::Overloaded(_)), "got {e:?}");
    assert_eq!(server.requests().await.len(), 3, "should stop after the attempt budget");
}

// A 200 that isn't the documented shape must be a clean typed error, not a
// panic or a silently empty answer set.
#[tokio::test]
async fn malformed_success_body_is_a_decode_error() {
    let server = MockServer::start(vec![MockResponse::json(200, "not json at all")]).await;
    let e = client(&server).ask(&json!("s"), "q", Question::noul("x")).await.unwrap_err();
    assert!(matches!(e, Error::Decode(_)), "got {e:?}");
}

// An answer silently missing from the response must not read as "no".
#[tokio::test]
async fn a_missing_answer_is_an_error_not_a_default() {
    let server = MockServer::start(vec![MockResponse::json(
        200,
        r#"{"model":"jev-1.13.0","answers":{},"usage":{"input_tokens":1,"output_tokens":0}}"#,
    )])
    .await;
    let e = client(&server).ask(&json!("s"), "q", Question::noul("x")).await.unwrap_err();
    assert!(matches!(e, Error::Decode(_)), "got {e:?}");
    assert!(e.to_string().contains("\"q\""));
}

// Malformed questions are caught locally — a wasted round trip on something
// the caller could have been told about immediately is a bug.
#[tokio::test]
async fn invalid_questions_never_reach_the_wire() {
    let server = MockServer::start(vec![MockResponse::json(200, OK_NOUL)]).await;
    let c = client(&server);

    let e = c
        .ask(&json!("s"), "q", Question::score("rate", vec!["only one".into()]))
        .await
        .unwrap_err();
    assert!(matches!(e, Error::Invalid(_)), "got {e:?}");

    let e = c
        .evaluate(&json!("s"), &[("q".into(), Question::choice("pick", Vec::<(String, Option<String>)>::new()))])
        .await
        .unwrap_err();
    assert!(matches!(e, Error::Invalid(_)), "got {e:?}");

    let e = c.evaluate(&json!("s"), &[]).await.unwrap_err();
    assert!(matches!(e, Error::Invalid(_)), "got {e:?}");

    assert!(server.requests().await.is_empty(), "nothing should have been sent");
}

// Two questions sharing an id would have one silently overwrite the other.
#[tokio::test]
async fn duplicate_question_ids_are_rejected() {
    let server = MockServer::start(vec![MockResponse::json(200, OK_NOUL)]).await;
    let e = client(&server)
        .evaluate(
            &json!("s"),
            &[("q".into(), Question::noul("a")), ("q".into(), Question::noul("b"))],
        )
        .await
        .unwrap_err();
    assert!(matches!(e, Error::Invalid(_)), "got {e:?}");
    assert!(server.requests().await.is_empty());
}

#[tokio::test]
async fn model_listing_accepts_a_bare_array_and_an_envelope() {
    let arr = r#"[{"name":"jev-1.13.0","description":"flagship","release_date":"2026-09-01"}]"#;
    let server = MockServer::start(vec![MockResponse::json(200, arr)]).await;
    let models = client(&server).models().await.expect("models");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "jev-1.13.0");
    assert!(server.requests().await[0].starts_with("GET /v1/models "));

    let env = r#"{"models":[{"name":"jev-latest"},{"name":"jev-preview"}]}"#;
    let server = MockServer::start(vec![MockResponse::json(200, env)]).await;
    let models = client(&server).models().await.expect("models");
    assert_eq!(models.len(), 2);
}

// with_model must change only the model, keeping endpoint and key.
#[tokio::test]
async fn with_model_repoints_only_the_model() {
    let server = MockServer::start(vec![MockResponse::json(200, OK_NOUL)]).await;
    let pinned = client(&server).with_model("jev-1.13.0");
    pinned.ask(&json!("s"), "q", Question::noul("x")).await.expect("evaluate");
    let raw = &server.requests().await[0];
    assert_eq!(body_of(raw)["model"], "jev-1.13.0");
    assert!(raw.to_lowercase().contains("authorization: bearer test-key"));
}
