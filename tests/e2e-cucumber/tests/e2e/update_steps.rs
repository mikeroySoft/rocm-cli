// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm update` (report only), including offline metadata failure.

use cucumber::{given, then, when};

use crate::E2eWorld;

#[given("a machine with no managed runtimes")]
async fn no_managed_runtimes(_world: &mut E2eWorld) {
    // The World's isolated data dir starts with an empty runtimes registry, so
    // `update` has nothing to check against the network. No setup required.
}

#[when("the user checks for updates")]
async fn check_updates(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["update"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the report shows there are no managed runtimes to update")]
async fn no_runtimes_to_update(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("managed runtimes: none"),
        "expected 'managed runtimes: none', got:\n{out}"
    );
}

#[then("it reports each update feed's status, marking unpublished feeds as not configured")]
async fn reports_feed_status(world: &mut E2eWorld) {
    let out = ok_output(world);
    // The update_surfaces block reports one line per feed. Assert each feed's status
    // ON ITS OWN LINE, so a status attributed to the wrong feed fails — a check that
    // only looked for the substrings anywhere would pass even if `not_configured`
    // and `package_managed` were swapped between the cli and engines feeds. The CLI
    // feed is not published yet (the "not configured" side of the distinction);
    // engines and recipes report their own stable states.
    for (feed, status) in [
        ("cli:", "status=not_configured"),
        ("engines:", "status=package_managed"),
        ("model_recipes:", "status=built_in"),
    ] {
        let line = out
            .lines()
            .map(str::trim)
            .find(|line| line.starts_with(feed));
        match line {
            Some(line) => assert!(
                line.contains(status),
                "update feed {feed:?} did not report {status:?} on its own line; got {line:?}\n\nfull output:\n{out}"
            ),
            None => panic!("no update feed line for {feed:?} in:\n{out}"),
        }
    }
}

// Linux's full accept queue drops SYNs; Windows has different backlog behavior.
// Compile the interposer only for these scenarios, never into the shipped CLI.
#[cfg(target_os = "linux")]
#[when(expr = "the user runs {word} with blackholed metadata connections")]
async fn check_with_blackholed_metadata(world: &mut E2eWorld, command: String) {
    use std::process::Command;
    use std::time::{Duration, Instant};

    let upper_bound = match command.as_str() {
        "version" => 8,
        "update" => 18,
        _ => panic!("unsupported timeout scenario command: {command}"),
    };
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let library = root.path().join("blackhole-dns.so");
    let compiled = Command::new("cc")
        .args(["-shared", "-fPIC", "-Wall", "-Wextra", "-Werror"])
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/e2e/blackhole_dns.c"
        ))
        .args(["-ldl", "-o"])
        .arg(&library)
        .output()
        .expect("failed to run the Linux C compiler");
    assert!(
        compiled.status.success(),
        "failed to compile blackhole fixture: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );

    let started = Instant::now();
    let (stdout, stderr, rc) = crate::run_rocm_with_env(
        world,
        &[&command],
        &[(
            "LD_PRELOAD",
            library.to_str().expect("non-UTF-8 fixture path"),
        )],
    );
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(upper_bound),
        "{command} exceeded its connect budget: {elapsed:?}; rc={rc}\n{stdout}\n{stderr}"
    );
    // Reject immediate DNS errors/refusals: the fixture must really time out.
    let lower_bound = if command == "version" { 1 } else { 9 };
    assert!(
        elapsed >= Duration::from_secs(lower_bound),
        "blackhole fixture did not exercise a connect wait: {elapsed:?}\n{stdout}\n{stderr}"
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the startup check records a metadata timeout")]
async fn startup_metadata_timeout(world: &mut E2eWorld) {
    ok_output(world);
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let record: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.path().join("cache/therock/startup-update-check.json"))
            .expect("startup check did not record an outcome"),
    )
    .expect("invalid startup check record");
    assert_eq!(record["runtime_key"], "release-tarball-gfx942");
    assert_eq!(record["status"], "error");
    assert!(
        record["message"]
            .as_str()
            .unwrap_or("")
            .contains("timed out"),
        "expected a metadata transport timeout: {record}"
    );
}

#[then("the update report records a metadata timeout")]
async fn report_metadata_timeout(world: &mut E2eWorld) {
    let out = ok_output(world);
    let line = out
        .lines()
        .find(|line| {
            line.trim_start()
                .starts_with("runtime release-tarball-gfx942 ")
        })
        .expect("update report omitted the registered runtime");
    assert!(
        line.contains("status=error") && line.contains("timed out"),
        "expected a metadata transport timeout for this runtime: {line}"
    );
}

// ── Helpers ────────────────────────────────────────────────────────

fn ok_output(world: &E2eWorld) -> String {
    let rc = world.cli_rc.expect("no command rc recorded");
    let combined = format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    );
    assert_eq!(rc, 0, "expected success, got rc={rc}:\n{combined}");
    world.cli_output.clone().unwrap_or_default()
}
