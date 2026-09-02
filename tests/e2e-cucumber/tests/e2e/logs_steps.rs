// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm logs`. Plant deterministic command-log files in the isolated
//! data dir, then assert on the search output. `logs` reads only the TAIL of each
//! file, so the matching lines are planted WITHIN the tail window and the
//! assertion is on the MATCH COUNT (deterministic), not the recent-line total
//! (which also counts other log sources). Contracts verified against the running
//! Linux binary (EAI-8072). No GPU or network — mock lane.

use std::fmt::Write as _;

use cucumber::{given, then, when};

use crate::E2eWorld;

/// The topic term planted into the log and searched for. Distinctive so it can't
/// collide with anything the CLI itself writes into the isolated log dir.
const TOPIC: &str = "E2ENEEDLE";
/// Number of matching lines planted — asserted exactly in the search result.
const MATCH_COUNT: usize = 9;
/// Built-in engines that answer the engine-plugin `logs` method.
const ENGINES: [&str; 2] = ["lemonade", "vllm"];
/// Service whose log is planted for every engine.
const SERVICE_ID: &str = "e2e-tail-probe";
/// Lines planted per service log; more than the default tail so the limit bites.
const PLANTED_LINES: usize = 200;
/// `rocm_engine_protocol::DEFAULT_LOG_TAIL_LINES`, pinned here as the black-box
/// contract value every engine must honour when `tail_lines` is omitted (#17).
const PROTOCOL_DEFAULT_TAIL: usize = 80;

#[given("recorded command logs containing several lines about a topic")]
async fn plant_command_logs(world: &mut E2eWorld) {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let cli_logs = root.path().join("data").join("logs").join("cli");
    std::fs::create_dir_all(&cli_logs).expect("failed to create cli logs dir");
    // Write the matching lines LAST so they fall within the per-file tail window
    // `rocm logs` reads (a few leading non-matching lines are harmless context).
    let mut body = String::new();
    for i in 0..3 {
        let _ = writeln!(body, "2026-01-01T00:00:0{i} unrelated startup line");
    }
    for i in 1..=MATCH_COUNT {
        let _ = writeln!(
            body,
            "2026-01-01T00:01:00 event {TOPIC} occurred number {i}"
        );
    }
    std::fs::write(cli_logs.join("e2e-probe.log"), body).expect("failed to write log file");
}

#[given("a service log longer than the default tail for each built-in engine")]
async fn plant_engine_service_logs(world: &mut E2eWorld) {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let body: String = (1..=PLANTED_LINES).fold(String::new(), |mut acc, i| {
        let _ = writeln!(acc, "line {i}");
        acc
    });
    for engine in ENGINES {
        let logs = root
            .path()
            .join("data")
            .join("engines")
            .join(engine)
            .join("logs");
        std::fs::create_dir_all(&logs).expect("failed to create engine logs dir");
        std::fs::write(logs.join(format!("{SERVICE_ID}.log")), &body)
            .expect("failed to write engine service log");
    }
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user searches the logs for that topic")]
async fn search_topic(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["logs", "--search", TOPIC]);
    record(world, stdout, stderr, rc);
}

#[when("the user searches the logs for a term that appears nowhere")]
async fn search_absent(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["logs", "--search", "e2e-no-such-term"]);
    record(world, stdout, stderr, rc);
}

#[when("the user asks for one service's logs and a search term together")]
async fn service_and_search(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &["logs", "--service", "some-service", "--search", TOPIC],
    );
    record(world, stdout, stderr, rc);
}

#[when("each engine is asked for that service's logs without a line limit")]
async fn ask_each_engine_for_logs(world: &mut E2eWorld) {
    let envelope = serde_json::json!({
        "method": "logs",
        "payload": { "service_id": SERVICE_ID },
    })
    .to_string();
    // Record one "<engine> <count> <last line>" row per engine in the shared
    // output slot so the Then step can compare engines against each other.
    let mut rows = String::new();
    for engine in ENGINES {
        let (stdout, stderr, rc) =
            crate::run_rocm_with_stdin(world, &["__engine-stdio", engine], &envelope, &[]);
        assert_eq!(rc, 0, "{engine} logs request failed:\n{stdout}\n{stderr}");
        let response: serde_json::Value =
            serde_json::from_str(&stdout).expect("engine response is not JSON");
        assert_eq!(
            response["ok"], true,
            "{engine} logs request was rejected:\n{stdout}"
        );
        let lines = response["data"]["recent_lines"]
            .as_array()
            .expect("recent_lines missing");
        let last = lines.last().and_then(|v| v.as_str()).unwrap_or("");
        let _ = writeln!(rows, "{engine} {} {last}", lines.len());
    }
    record(world, rows, String::new(), 0);
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the CLI reports the matching recent lines")]
async fn reports_matches(world: &mut E2eWorld) {
    let out = ok_output(world);
    // The count of matching lines is deterministic (we planted exactly nine within
    // the tail); the "of <total>" denominator is not, so assert only the match side.
    assert!(
        out.contains(&format!("Lines: {MATCH_COUNT} of ")),
        "expected {MATCH_COUNT} matching lines, got:\n{out}"
    );
    assert!(
        out.contains(&format!("Showing: 1-{MATCH_COUNT} of {MATCH_COUNT}")),
        "expected the {MATCH_COUNT} matches to be listed, got:\n{out}"
    );
}

#[then("the CLI reports no matching lines")]
async fn reports_no_matches(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("Lines: 0 of ") && out.contains("Showing: 0 of 0"),
        "expected no matching lines, got:\n{out}"
    );
}

#[then("the CLI refuses and explains only one may be used")]
async fn refuses_conflict(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert!(rc != 0, "expected refusal, got rc=0:\n{}", combined(world));
    assert!(
        combined(world)
            .contains("accepts either --service <service-id> or a search query, not both"),
        "expected the service/search conflict message, got:\n{}",
        combined(world)
    );
}

#[then("every engine returns exactly the protocol default number of lines")]
async fn every_engine_returns_default_tail(world: &mut E2eWorld) {
    let out = ok_output(world);
    let expected_last = format!("line {PLANTED_LINES}");
    for engine in ENGINES {
        assert!(
            out.contains(&format!(
                "{engine} {PROTOCOL_DEFAULT_TAIL} {expected_last}\n"
            )),
            "expected {engine} to return the newest {PROTOCOL_DEFAULT_TAIL} lines, got:\n{out}"
        );
    }
}

// ── Helpers ────────────────────────────────────────────────────────

fn record(world: &mut E2eWorld, stdout: String, stderr: String, rc: i32) {
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

fn combined(world: &E2eWorld) -> String {
    format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    )
}

fn ok_output(world: &E2eWorld) -> String {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert_eq!(rc, 0, "expected success, got rc={rc}:\n{}", combined(world));
    world.cli_output.clone().unwrap_or_default()
}
