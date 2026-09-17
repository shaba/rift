//! End-to-end checks for the System One swarm judge, against a mock API.
//!
//! The unit tests in `swarm.rs` cover state building and verdict rendering in
//! isolation; these drive the whole path — request construction, the real
//! response shape, the correctness gate, and the eligibility re-check — so a
//! change to any one link can't quietly produce a wrong merge recommendation.

use std::path::PathBuf;

use rift_core::{judge_swarm_system_one, Candidate, CandidateOutcome, TurnStats};
use rift_provider::test_support::{MockResponse, MockServer};
use rift_typesafe::TypeSafeClient;
use serde_json::Value;

fn client(server: &MockServer) -> TypeSafeClient {
    TypeSafeClient::new(&server.base_url, "test-key", "jev-latest")
}

/// A candidate that produced a real patch file on disk.
fn changed(dir: &std::path::Path, name: &str, patch: &str) -> CandidateOutcome {
    let path = dir.join(format!("{name}.patch"));
    std::fs::write(&path, patch).expect("write patch");
    CandidateOutcome {
        candidate: Candidate { name: name.into(), model: "test-model".into(), temperature: None },
        worktree: PathBuf::new(),
        patch_path: Some(path),
        diff_stat: "1 file changed, 1 insertion(+)".into(),
        summary: format!("{name} says it fixed everything"),
        stats: TurnStats::default(),
        error: None,
    }
}

fn unchanged(name: &str) -> CandidateOutcome {
    CandidateOutcome {
        candidate: Candidate { name: name.into(), model: "test-model".into(), temperature: None },
        worktree: PathBuf::new(),
        patch_path: None,
        diff_stat: "no changes".into(),
        summary: String::new(),
        stats: TurnStats::default(),
        error: None,
    }
}

fn errored(name: &str) -> CandidateOutcome {
    CandidateOutcome {
        candidate: Candidate { name: name.into(), model: "test-model".into(), temperature: None },
        worktree: PathBuf::new(),
        patch_path: None,
        diff_stat: String::new(),
        summary: String::new(),
        stats: TurnStats::default(),
        error: Some("worktree blew up".into()),
    }
}

fn body_of(raw: &str) -> Value {
    let (_, body) = raw.split_once("\r\n\r\n").expect("request has a body");
    serde_json::from_str(body).expect("body is JSON")
}

/// A verdict response picking `winner` with `confidence`, gated by `any_correct`.
fn verdict(winner: &str, confidence: f64, any_correct: f64, probs: &str) -> String {
    format!(
        r#"{{"model":"jev-1.13.0","answers":{{
            "winner":{{"type":"choice","choice":"{winner}","probabilities":{probs},
                       "confidence":{confidence}}},
            "any_correct":{{"type":"noul","noul":{any_correct}}}}},
            "usage":{{"input_tokens":900,"output_tokens":0}}}}"#
    )
}

#[tokio::test]
async fn judge_asks_one_call_with_both_questions_and_only_eligible_options() {
    let dir = std::env::temp_dir().join("rift-so-judge-1");
    std::fs::create_dir_all(&dir).unwrap();
    let outcomes = vec![
        changed(&dir, "0-alpha", "@@ -1 +1 @@\n-bad\n+good\n"),
        unchanged("1-beta"),
        errored("2-gamma"),
    ];
    let server = MockServer::start(vec![MockResponse::json(
        200,
        &verdict("0-alpha", 0.81, 0.93, r#"{"0-alpha":1.0}"#),
    )])
    .await;

    let v = judge_swarm_system_one(&client(&server), "fix the parser", &outcomes)
        .await
        .expect("judge");

    let reqs = server.requests().await;
    assert_eq!(reqs.len(), 1, "the whole verdict must cost one call");
    let body = body_of(&reqs[0]);

    // Both questions ride in the same request.
    assert_eq!(body["questions"]["winner"]["type"], "choice");
    assert_eq!(body["questions"]["any_correct"]["type"], "noul");

    // Only the candidate that actually changed something is offered. This is
    // what makes an illegal pick unrepresentable instead of merely forbidden.
    let opts = body["questions"]["winner"]["criteria"].as_object().expect("criteria map");
    assert_eq!(opts.len(), 1, "only eligible candidates may be options: {opts:?}");
    assert!(opts.contains_key("0-alpha"));
    assert!(!opts.contains_key("1-beta"));
    assert!(!opts.contains_key("2-gamma"));

    // The state carries the task, every candidate's status, and the patch.
    assert_eq!(body["state"]["task"], "fix the parser");
    let cands = body["state"]["candidates"].as_array().unwrap();
    assert_eq!(cands.len(), 3, "the judge still sees the failures as context");
    assert_eq!(cands[0]["status"], "changed");
    assert!(cands[0]["patch"].as_str().unwrap().contains("+good"));
    assert_eq!(cands[1]["status"], "no changes");
    assert_eq!(cands[2]["status"], "errored");
    assert_eq!(cands[2]["error"], "worktree blew up");

    assert_eq!(v.winner.as_deref(), Some("0-alpha"));
    assert_eq!(v.confidence, Some(0.81));
    assert!(v.text.contains("winner: 0-alpha"));
}

// The correctness gate: if nothing is likely correct, a choice still comes
// back (the model must pick something) but it must not become a winner.
#[tokio::test]
async fn a_pick_is_discarded_when_no_candidate_is_likely_correct() {
    let dir = std::env::temp_dir().join("rift-so-judge-2");
    std::fs::create_dir_all(&dir).unwrap();
    let outcomes = vec![changed(&dir, "0-alpha", "patch a"), changed(&dir, "1-beta", "patch b")];
    let server = MockServer::start(vec![MockResponse::json(
        200,
        // Confident about which is better; confident that neither works.
        &verdict("1-beta", 0.95, 0.04, r#"{"0-alpha":0.05,"1-beta":0.95}"#),
    )])
    .await;

    let v = judge_swarm_system_one(&client(&server), "t", &outcomes).await.expect("judge");
    assert_eq!(v.winner, None, "a high-confidence pick must still fail the correctness gate");
    assert!(v.text.contains("no candidate is likely to have solved the task"));
}

#[tokio::test]
async fn a_pick_survives_when_something_is_probably_correct() {
    let dir = std::env::temp_dir().join("rift-so-judge-3");
    std::fs::create_dir_all(&dir).unwrap();
    let outcomes = vec![changed(&dir, "0-alpha", "patch a"), changed(&dir, "1-beta", "patch b")];
    let server = MockServer::start(vec![MockResponse::json(
        200,
        &verdict("1-beta", 0.42, 0.77, r#"{"0-alpha":0.3,"1-beta":0.7}"#),
    )])
    .await;

    let v = judge_swarm_system_one(&client(&server), "t", &outcomes).await.expect("judge");
    assert_eq!(v.winner.as_deref(), Some("1-beta"));
    // Low confidence is reported, not silently swallowed.
    assert!(v.text.contains("low confidence"));
    assert!(v.text.contains("1-beta: 0.70"));
}

// Defence in depth: the option list should make this impossible, but a name
// that isn't an eligible candidate must never become a merge recommendation.
#[tokio::test]
async fn an_ineligible_or_unknown_pick_is_rejected() {
    let dir = std::env::temp_dir().join("rift-so-judge-4");
    std::fs::create_dir_all(&dir).unwrap();
    let outcomes = vec![changed(&dir, "0-alpha", "patch a"), unchanged("1-beta")];
    let server = MockServer::start(vec![MockResponse::json(
        200,
        &verdict("1-beta", 0.99, 0.99, r#"{"1-beta":0.99}"#),
    )])
    .await;

    let v = judge_swarm_system_one(&client(&server), "t", &outcomes).await.expect("judge");
    assert_eq!(v.winner, None, "a candidate with no changes must never win");
}

// A missing noul must not read as "yes" and promote a winner by default.
#[tokio::test]
async fn a_missing_correctness_answer_blocks_the_winner() {
    let dir = std::env::temp_dir().join("rift-so-judge-5");
    std::fs::create_dir_all(&dir).unwrap();
    let outcomes = vec![changed(&dir, "0-alpha", "patch a")];
    let server = MockServer::start(vec![MockResponse::json(
        200,
        r#"{"model":"jev-1.13.0","answers":{"winner":{"type":"choice","choice":"0-alpha",
            "probabilities":{"0-alpha":1.0},"confidence":0.9}},
            "usage":{"input_tokens":1,"output_tokens":0}}"#,
    )])
    .await;

    let v = judge_swarm_system_one(&client(&server), "t", &outcomes).await.expect("judge");
    assert_eq!(v.winner, None, "an absent correctness gate must fail closed");
}

// An API failure must surface as an error, not as a silent "no winner" that
// looks like a real verdict.
#[tokio::test]
async fn an_api_failure_is_an_error_not_an_empty_verdict() {
    let dir = std::env::temp_dir().join("rift-so-judge-6");
    std::fs::create_dir_all(&dir).unwrap();
    let outcomes = vec![changed(&dir, "0-alpha", "patch a")];
    let server = MockServer::start(vec![MockResponse::json(
        401,
        r#"{"error":{"message":"invalid api key"}}"#,
    )])
    .await;

    let e = judge_swarm_system_one(&client(&server), "t", &outcomes)
        .await
        .expect_err("401 must propagate");
    assert!(format!("{e:#}").contains("invalid api key"), "got {e:#}");
}

// Nothing mergeable means nothing to decide — and no call to pay for.
#[tokio::test]
async fn an_empty_race_costs_no_request() {
    let outcomes = vec![unchanged("0-alpha"), errored("1-beta")];
    let server = MockServer::start(vec![MockResponse::json(200, "{}")]).await;
    let v = judge_swarm_system_one(&client(&server), "t", &outcomes).await.expect("judge");
    assert_eq!(v.winner, None);
    assert!(v.text.contains("nothing to judge"));
    assert!(server.requests().await.is_empty(), "no candidates should mean no request");
}
