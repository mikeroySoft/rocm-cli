// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocmd_services.feature`: the rocmd managed-service stop path,
//! exercised black-box through `rocmd sandbox-run stop_server`.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use cucumber::{given, then, when};
use e2e_cucumber::mock_server::{ServiceRecordOptions, write_service_record_named_with};

use crate::E2eWorld;

const SERVICE_ID: &str = "e2e-rocmd-stop";

fn rocmd_binary() -> PathBuf {
    let configured = std::env::var_os("ROCM_CLI_ROCMD_BINARY").unwrap_or_else(|| {
        panic!(
            "this rocmd-backed scenario requires ROCM_CLI_ROCMD_BINARY; when using a prebuilt \
             ROCM_CLI_BINARY, provide the matching prebuilt rocmd path explicitly"
        )
    });
    PathBuf::from(configured)
        .canonicalize()
        .expect("failed to resolve ROCM_CLI_ROCMD_BINARY")
}

/// Kernel start-time (field 22 of `/proc/<pid>/stat`), the identity rocmd pairs
/// with a recorded PID. Parsed after the final `)` so a `comm` containing
/// spaces cannot shift the fields.
fn start_ticks(pid: u32) -> u64 {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).expect("read /proc stat");
    let rest = &stat[stat.rfind(')').expect("stat has comm") + 1..];
    rest.split_whitespace()
        .nth(19)
        .and_then(|field| field.parse().ok())
        .expect("stat has starttime")
}

fn spawn_target(world: &mut E2eWorld) -> u32 {
    let child = Command::new("sleep")
        .arg("60")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn sleep");
    let pid = child.id();
    world.stop_target = Some(child);
    pid
}

fn write_record(world: &E2eWorld, pid: u32, ticks: u64) {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let services = root.path().join("data").join("services");
    write_service_record_named_with(
        &services,
        SERVICE_ID,
        "Qwen/Qwen3.5",
        11435,
        ServiceRecordOptions {
            supervisor_pid: pid,
            engine_pid: Some(pid),
            supervisor_start_ticks: Some(ticks),
            engine_start_ticks: Some(ticks),
            ..ServiceRecordOptions::default()
        },
    );
}

fn stop_result(world: &E2eWorld) -> serde_json::Value {
    let stdout = world.cli_output.as_deref().unwrap_or("");
    let report: serde_json::Value = serde_json::from_str(stdout)
        .unwrap_or_else(|error| panic!("rocmd did not print JSON ({error}): {stdout}"));
    report
        .pointer("/output/result")
        .cloned()
        .unwrap_or_else(|| panic!("no output.result in rocmd report: {report}"))
}

fn pid_list(result: &serde_json::Value, key: &str) -> Vec<u64> {
    result
        .get(key)
        .and_then(serde_json::Value::as_array)
        .unwrap_or_else(|| panic!("no `{key}` list in stop result: {result}"))
        .iter()
        .filter_map(serde_json::Value::as_u64)
        .collect()
}

fn target_pid(world: &E2eWorld) -> u64 {
    u64::from(world.stop_target.as_ref().expect("no stop target").id())
}

fn target_running(world: &mut E2eWorld) -> bool {
    let child = world.stop_target.as_mut().expect("no stop target");
    child.try_wait().expect("try_wait").is_none()
}

#[given("a managed service record whose PID now belongs to an unrelated process")]
async fn record_with_recycled_pid(world: &mut E2eWorld) {
    let pid = spawn_target(world);
    // A start-time that cannot match the live process: exactly what the record
    // of an exited service looks like once the kernel hands its PID out again.
    write_record(world, pid, start_ticks(pid).wrapping_add(1));
}

#[given("a managed service record pointing at a live process it owns")]
async fn record_with_matching_pid(world: &mut E2eWorld) {
    let pid = spawn_target(world);
    write_record(world, pid, start_ticks(pid));
}

#[when("the user stops the service through rocmd")]
async fn stop_through_rocmd(world: &mut E2eWorld) {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let empty_path = root.path().join("empty-path");
    std::fs::create_dir_all(&empty_path).expect("failed to create isolated PATH directory");
    let mut command = Command::new(rocmd_binary());
    command.args([
        "sandbox-run",
        "stop_server",
        "--service-id",
        SERVICE_ID,
        "--allow-native-fallback",
    ]);
    world.isolate_cmd(&mut command);
    // Force the documented restricted-native fallback so the stop runs in this
    // PID namespace regardless of whether the host provides bubblewrap.
    command.env("PATH", empty_path);
    let output = command.output().expect("failed to run rocmd");
    world.cli_output = Some(String::from_utf8_lossy(&output.stdout).into_owned());
    world.cli_stderr = Some(String::from_utf8_lossy(&output.stderr).into_owned());
    world.cli_rc = Some(output.status.code().unwrap_or(-1));
    assert_eq!(
        world.cli_rc,
        Some(0),
        "rocmd stop_server failed: {}",
        world.cli_stderr.as_deref().unwrap_or("")
    );
}

#[then("the unrelated process is still running")]
async fn unrelated_still_running(world: &mut E2eWorld) {
    assert!(target_running(world), "rocmd signalled a recycled PID");
}

#[then("the owned process is no longer running")]
async fn owned_process_gone(world: &mut E2eWorld) {
    assert!(!target_running(world), "rocmd did not stop its own process");
}

#[then("rocmd reports the PID as skipped, not signaled")]
async fn reports_skipped(world: &mut E2eWorld) {
    let result = stop_result(world);
    let pid = target_pid(world);
    assert!(pid_list(&result, "signaled_pids").is_empty(), "{result}");
    assert!(
        pid_list(&result, "force_signaled_pids").is_empty(),
        "{result}"
    );
    assert_eq!(pid_list(&result, "skipped_pids"), vec![pid], "{result}");
}

#[then("rocmd reports the PID as signaled")]
async fn reports_signaled(world: &mut E2eWorld) {
    let result = stop_result(world);
    let pid = target_pid(world);
    assert_eq!(pid_list(&result, "signaled_pids"), vec![pid], "{result}");
    assert!(pid_list(&result, "skipped_pids").is_empty(), "{result}");
}

#[then("the service record is marked stopped, since the recorded process is gone")]
async fn record_stopped_after_recycle(world: &mut E2eWorld) {
    record_stopped(world).await;
}

#[then("the service record is marked stopped")]
async fn record_stopped(world: &mut E2eWorld) {
    let status = stop_result(world);
    assert_eq!(
        status
            .pointer("/service/status")
            .and_then(serde_json::Value::as_str),
        Some("stopped"),
        "{status}"
    );
}

/// Reap the spawned target so it never outlives the scenario.
pub fn teardown(target: &mut Child) {
    let _ = target.kill();
    let _ = target.wait();
}
