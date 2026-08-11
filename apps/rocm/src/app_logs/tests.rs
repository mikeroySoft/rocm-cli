// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use std::collections::BTreeSet;
use std::sync::LazyLock;

use rocm_core::{ManagedToolConfig, ProviderUserConfig};

use super::*;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A scratch `AppPaths` under the workspace's ignored work directory, in the
/// same shape as `main.rs`'s own `test_paths`. Deliberately *not* created: the
/// first-run tests need the directory absent, and the rest create only the one
/// subdirectory they write to.
fn test_paths(name: &str) -> (PathBuf, AppPaths) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(".rocm-work")
        .join("tests")
        .join("app-logs")
        .join(format!(
            "rocm-cli-app-logs-{name}-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
    let _ = fs::remove_dir_all(&root);
    (
        root.clone(),
        AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        },
    )
}

/// Fixed identity, so every redaction assertion holds on any machine and in CI.
fn planted_redactor() -> Redactor {
    Redactor::with(
        &["/home/plantedhomedir"],
        Some("plantedusername"),
        Some("plantedhostname"),
    )
}

/// A redactor that substitutes nothing, for tests about parsing rather than
/// redaction.
fn bare_redactor() -> Redactor {
    Redactor::with(&[], None, None)
}

fn write_file(path: &Path, contents: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, contents).unwrap();
}

fn lifecycle_line(at_unix_ms: u64, level: &str, action: &str, message: &str) -> String {
    crate::render_cli_lifecycle_log_line(&rocm_core::AuditEventRecord {
        at_unix_ms: u128::from(at_unix_ms),
        source: "rocm".to_owned(),
        category: "runtime".to_owned(),
        actor: "cli".to_owned(),
        level: level.to_owned(),
        action: action.to_owned(),
        message: message.to_owned(),
        watcher_id: None,
        service_id: None,
    })
}

fn record(source: SourceId, index: usize, at_unix_ms: u64, severity: Severity) -> LogRecord {
    LogRecord {
        id: format!("{}:{index}", source.as_str()),
        source,
        at_unix_ms,
        severity,
        category: None,
        action: None,
        summary: format!("record {index}"),
        detail: None,
    }
}

fn scan_of(source: SourceId, records: Vec<LogRecord>) -> SourceScan {
    SourceScan {
        id: source,
        available: true,
        truncated: false,
        path: Some(format!("~/.rocm/{}", source.as_str())),
        records,
    }
}

fn inputs_of(scans: Vec<SourceScan>) -> LogsInputs {
    LogsInputs {
        generated_at_unix_ms: 1_785_152_700_123,
        first_run: false,
        scans,
    }
}

fn scan_named(inputs: &LogsInputs, id: SourceId) -> &SourceScan {
    inputs
        .scans
        .iter()
        .find(|scan| scan.id == id)
        .expect("every source is scanned")
}

// ---------------------------------------------------------------------------
// Creates nothing
// ---------------------------------------------------------------------------

#[test]
fn app_logs_reading_a_missing_data_dir_creates_nothing() {
    let (root, paths) = test_paths("first-run");

    let response = build_logs(gather_logs(&paths, &bare_redactor()), &LogsQuery::default());

    assert!(response.first_run, "no data directory means first run");
    assert!(response.records.is_empty());
    assert_eq!(response.sources.len(), SourceId::ALL.len());
    assert!(
        response.sources.iter().all(|source| !source.available),
        "nothing can be available when nothing exists: {:?}",
        response.sources
    );
    assert!(response.bounds.truncated.is_empty());
    assert!(
        !root.exists(),
        "reading logs must not create {}",
        root.display()
    );
    assert!(!paths.data_dir.exists());
}

#[test]
fn app_logs_a_data_dir_without_logs_is_not_a_first_run() {
    let (root, paths) = test_paths("not-first-run");
    fs::create_dir_all(&paths.data_dir).unwrap();

    let response = build_logs(gather_logs(&paths, &bare_redactor()), &LogsQuery::default());

    assert!(!response.first_run);
    assert!(response.records.is_empty());
    let _ = fs::remove_dir_all(root);
}

// ---------------------------------------------------------------------------
// Bounds
// ---------------------------------------------------------------------------

#[test]
fn app_logs_tail_read_stops_at_the_byte_bound_and_keeps_the_newest_bytes() {
    let (root, paths) = test_paths("tail-bytes");
    let path = paths.data_dir.join("big.log");
    // Exactly 64 bytes per line ("line-" + 5 digits + 53 filler + newline), so
    // the assertions below are about the bound and not about line arithmetic.
    let filler = "x".repeat(53);
    let mut contents = String::with_capacity(10_000 * 64);
    for index in 0..10_000 {
        let _ = writeln!(contents, "line-{index:05}{filler}");
    }
    assert_eq!(contents.len(), 640_000);
    write_file(&path, &contents);

    let tail = read_tail(&path, MAX_BYTES_PER_FILE).unwrap();

    assert_eq!(
        tail.bytes, MAX_BYTES_PER_FILE,
        "the read itself must be bounded, not a whole-file read trimmed afterwards"
    );
    assert_eq!(tail.file_bytes, 640_000);
    assert!(tail.file_bytes > tail.bytes);
    assert!(u64::try_from(tail.text.len()).unwrap() <= MAX_BYTES_PER_FILE);
    assert!(
        !tail.text.contains("line-00000"),
        "the oldest line must not survive a tail read"
    );
    assert!(
        tail.text.contains("line-09999"),
        "the newest line must survive a tail read"
    );
    assert!(
        tail.text.starts_with("line-"),
        "the partial first line must be discarded"
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn app_logs_a_file_over_the_byte_bound_marks_its_source_truncated() {
    let (root, paths) = test_paths("truncated-bytes");
    let filler = "y".repeat(40);
    let mut contents = String::new();
    for index in 0..8_000u64 {
        contents.push_str(&lifecycle_line(
            1_700_000_000_000 + index,
            "info",
            "activate",
            &format!("message {index} {filler}"),
        ));
    }
    assert!(contents.len() > MAX_BYTES_PER_FILE as usize);
    write_file(&crate::cli_lifecycle_log_path(&paths), &contents);

    let response = build_logs(gather_logs(&paths, &bare_redactor()), &LogsQuery::default());

    assert_eq!(response.bounds.truncated, vec![SourceId::CliLifecycle]);
    assert_eq!(response.bounds.max_bytes_per_file, MAX_BYTES_PER_FILE);
    assert_eq!(response.bounds.max_lines_per_file, MAX_LINES_PER_FILE);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn app_logs_a_file_over_the_line_bound_keeps_only_the_newest_lines() {
    let (root, paths) = test_paths("truncated-lines");
    // Short lines: comfortably inside the byte bound, past the line bound.
    let mut contents = String::new();
    for index in 0..(MAX_LINES_PER_FILE as u64 + 500) {
        contents.push_str(&lifecycle_line(
            1_700_000_000_000 + index,
            "info",
            "activate",
            &format!("m{index}"),
        ));
    }
    assert!(contents.len() < MAX_BYTES_PER_FILE as usize);
    write_file(&crate::cli_lifecycle_log_path(&paths), &contents);

    let inputs = gather_logs(&paths, &bare_redactor());
    let scan = scan_named(&inputs, SourceId::CliLifecycle);

    assert!(scan.truncated);
    assert_eq!(scan.records.len(), MAX_LINES_PER_FILE);
    assert!(
        scan.records.iter().all(|record| record.summary != "m0"),
        "the oldest lines are the ones dropped"
    );
    assert!(scan.records.iter().any(|record| record.summary == "m2499"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn app_logs_never_returns_more_than_the_per_request_cap() {
    let records = (0..500_usize)
        .map(|index| {
            record(
                SourceId::CliAudit,
                index,
                1_700_000_000_000 + index as u64,
                Severity::Info,
            )
        })
        .collect();

    let response = build_logs(
        inputs_of(vec![scan_of(SourceId::CliAudit, records)]),
        &LogsQuery {
            page_size: Some(10_000),
            ..LogsQuery::default()
        },
    );

    assert_eq!(response.page.size, MAX_RECORDS_PER_REQUEST);
    assert_eq!(response.records.len(), MAX_RECORDS_PER_REQUEST);
    assert!(response.page.has_more);
    assert_eq!(response.sources[0].matched, 500);
}

#[test]
fn app_logs_pages_newest_first_and_reports_the_last_page() {
    let records = (0..500_usize)
        .map(|index| {
            record(
                SourceId::CliAudit,
                index,
                1_700_000_000_000 + index as u64,
                Severity::Info,
            )
        })
        .collect();
    let inputs = inputs_of(vec![scan_of(SourceId::CliAudit, records)]);

    let first = build_logs(inputs.clone(), &LogsQuery::default());
    assert_eq!(first.records[0].id, "cli-audit:499", "newest first");
    assert!(first.page.has_more);

    let last = build_logs(
        inputs,
        &LogsQuery {
            page: 2,
            ..LogsQuery::default()
        },
    );
    assert_eq!(last.page.index, 2);
    assert_eq!(last.page.returned, 100);
    assert!(!last.page.has_more);
    assert_eq!(last.records[0].id, "cli-audit:99");
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

#[test]
fn app_logs_parses_the_lifecycle_line_this_binary_actually_writes() {
    let (root, paths) = test_paths("lifecycle-format");
    // Rendered by the production writer, not hand-typed: the parser is pinned
    // to the format, so changing one without the other fails here.
    let line = crate::render_cli_lifecycle_log_line(&rocm_core::AuditEventRecord {
        at_unix_ms: 1_785_177_626_897,
        source: "rocm".to_owned(),
        category: "runtime".to_owned(),
        actor: "cli".to_owned(),
        level: "warn".to_owned(),
        action: "activate".to_owned(),
        message: "activated nightly-wheel-gfx120x-all-7-14-0 with x=1".to_owned(),
        watcher_id: None,
        service_id: Some("svc-1".to_owned()),
    });
    write_file(&crate::cli_lifecycle_log_path(&paths), &line);

    let response = build_logs(gather_logs(&paths, &bare_redactor()), &LogsQuery::default());

    assert_eq!(response.records.len(), 1);
    let parsed = &response.records[0];
    assert_eq!(parsed.id, "cli-lifecycle:0");
    assert_eq!(parsed.source, SourceId::CliLifecycle);
    assert_eq!(parsed.at_unix_ms, 1_785_177_626_897);
    assert_eq!(parsed.severity, Severity::Warn);
    assert_eq!(parsed.category.as_deref(), Some("runtime"));
    assert_eq!(parsed.action.as_deref(), Some("activate"));
    assert_eq!(
        parsed.summary, "activated nightly-wheel-gfx120x-all-7-14-0 with x=1",
        "the message runs to end of line, equals signs and all"
    );
    assert_eq!(parsed.detail.as_deref(), Some("service svc-1"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn app_logs_an_unknown_severity_is_info_not_an_error() {
    assert_eq!(Severity::lenient("NOTICE"), Severity::Info);
    assert_eq!(Severity::lenient(""), Severity::Info);
    assert_eq!(Severity::lenient("WARNING"), Severity::Warn);
    assert_eq!(Severity::lenient("Error"), Severity::Error);
    assert_eq!(Severity::from_token("notice"), None);

    let parsed = parse_lifecycle(
        "1700000000000 level=notice category=runtime action=activate \
         service_id=<none> message=hello",
    )
    .unwrap();
    assert_eq!(parsed.severity, Severity::Info);
    assert_eq!(parsed.summary, "hello");
    assert!(parsed.detail.is_none(), "<none> is not a service id");
}

#[test]
fn app_logs_parses_tracing_lines_into_level_target_and_message() {
    let parsed =
        parse_tracing("2026-07-27T11:45:00.123456Z  WARN rocm::therock: runtime is stale").unwrap();

    assert_eq!(parsed.at_unix_ms, 1_785_152_700_123);
    assert_eq!(parsed.severity, Severity::Warn);
    assert_eq!(parsed.category.as_deref(), Some("rocm::therock"));
    assert_eq!(parsed.summary, "runtime is stale");
    assert!(
        parse_tracing("    at src/main.rs:12").is_none(),
        "a continuation line is not a record of its own"
    );
}

#[test]
fn app_logs_iso8601_stamps_become_epoch_milliseconds() {
    assert_eq!(parse_iso8601_unix_ms("1970-01-01T00:00:00.000Z"), Some(0));
    assert_eq!(
        parse_iso8601_unix_ms("2026-07-27T11:45:00.123456Z"),
        Some(1_785_152_700_123)
    );
    // Leap-year handling is the whole reason this is not a multiplication.
    assert_eq!(
        parse_iso8601_unix_ms("2024-02-29T00:00:00Z"),
        Some(1_709_164_800_000)
    );
    assert_eq!(parse_iso8601_unix_ms("not-a-timestamp"), None);
    assert_eq!(parse_iso8601_unix_ms("2026-13-27T11:45:00Z"), None);
}

#[test]
fn app_logs_an_unstamped_line_inherits_the_stamp_above_it() {
    let (root, paths) = test_paths("continuation");
    write_file(
        &paths
            .client_log_dir()
            .join(format!("{}.2026-07-27", crate::logging::LOG_FILE_PREFIX)),
        "2026-07-27T11:45:00.123456Z ERROR rocm::serve: engine crashed\n\
         Caused by: exit status 1\n",
    );

    let inputs = gather_logs(&paths, &bare_redactor());
    let scan = scan_named(&inputs, SourceId::CliClient);

    assert_eq!(scan.records.len(), 2);
    assert_eq!(scan.records[1].summary, "Caused by: exit status 1");
    assert_eq!(
        scan.records[1].at_unix_ms, scan.records[0].at_unix_ms,
        "a continuation line belongs to the event above it"
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn app_logs_reads_only_the_newest_rotated_client_log() {
    let (root, paths) = test_paths("newest-rotation");
    let log_dir = paths.client_log_dir();
    let prefix = crate::logging::LOG_FILE_PREFIX;
    write_file(
        &log_dir.join(format!("{prefix}.2026-07-25")),
        "2026-07-25T01:00:00.000Z  INFO rocm: oldest\n",
    );
    write_file(
        &log_dir.join(format!("{prefix}.2026-07-27")),
        "2026-07-27T01:00:00.000Z  INFO rocm: newest\n",
    );

    let inputs = gather_logs(&paths, &bare_redactor());
    let scan = scan_named(&inputs, SourceId::CliClient);

    assert_eq!(scan.records.len(), 1);
    assert_eq!(scan.records[0].summary, "newest");
    let _ = fs::remove_dir_all(root);
}

#[test]
fn app_logs_parses_jsonl_audit_and_automation_events() {
    let (root, paths) = test_paths("jsonl");
    write_file(
        &paths.audit_events_path(),
        &format!(
            "{}\nnot json at all\n",
            serde_json::to_string(&rocm_core::AuditEventRecord {
                at_unix_ms: 1_785_177_626_897,
                source: "rocm".to_owned(),
                category: "service".to_owned(),
                actor: "cli".to_owned(),
                level: "error".to_owned(),
                action: "serve".to_owned(),
                message: "engine refused to start".to_owned(),
                watcher_id: None,
                service_id: Some("svc-7".to_owned()),
            })
            .unwrap()
        ),
    );
    write_file(
        &paths.automation_events_path(),
        &format!(
            "{}\n",
            serde_json::to_string(&rocm_core::AutomationEventRecord {
                at_unix_ms: 1_785_177_626_898,
                watcher_id: "watch-1".to_owned(),
                level: "warn".to_owned(),
                action: "propose".to_owned(),
                message: "proposal raised".to_owned(),
                service_id: None,
            })
            .unwrap()
        ),
    );

    let inputs = gather_logs(&paths, &bare_redactor());

    let audit = scan_named(&inputs, SourceId::CliAudit);
    assert_eq!(audit.records.len(), 2, "the unparseable line is kept");
    assert_eq!(audit.records[0].severity, Severity::Error);
    assert_eq!(audit.records[0].category.as_deref(), Some("service"));
    assert_eq!(audit.records[0].action.as_deref(), Some("serve"));
    assert_eq!(audit.records[0].detail.as_deref(), Some("service svc-7"));
    assert_eq!(audit.records[1].summary, "not json at all");
    assert_eq!(
        audit.records[1].at_unix_ms, audit.records[0].at_unix_ms,
        "an unparseable line inherits the stamp above it"
    );

    let automation = scan_named(&inputs, SourceId::Automation);
    assert_eq!(automation.records.len(), 1);
    assert_eq!(automation.records[0].severity, Severity::Warn);
    assert_eq!(
        automation.records[0].detail.as_deref(),
        Some("watcher watch-1")
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn app_logs_merges_every_per_command_action_log() {
    let (root, paths) = test_paths("action-logs");
    write_file(
        &crate::cli_action_log_path(&paths, "runtime", "activate"),
        &lifecycle_line(1_700_000_000_001, "info", "activate", "one"),
    );
    write_file(
        &crate::cli_action_log_path(&paths, "service", "serve"),
        &lifecycle_line(1_700_000_000_002, "info", "serve", "two"),
    );

    let inputs = gather_logs(&paths, &bare_redactor());
    let scan = scan_named(&inputs, SourceId::CliAction);

    assert!(scan.available);
    assert_eq!(scan.records.len(), 2);
    let ids: Vec<&str> = scan
        .records
        .iter()
        .map(|record| record.id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec!["cli-action:0", "cli-action:1"],
        "ids stay unique across a source's files"
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn app_logs_reads_every_service_log() {
    let (root, paths) = test_paths("service-logs");
    write_file(
        &paths.service_log_path("svc-a"),
        "2026-07-27T01:00:00.000Z  INFO vllm: listening\n",
    );
    write_file(&paths.service_log_path("svc-b"), "raw engine chatter\n");

    let inputs = gather_logs(&paths, &bare_redactor());
    let scan = scan_named(&inputs, SourceId::Service);

    assert!(scan.available);
    assert_eq!(scan.records.len(), 2);
    assert!(
        scan.records
            .iter()
            .any(|record| record.summary == "raw engine chatter"),
        "unstructured engine output is still a record"
    );
    let _ = fs::remove_dir_all(root);
}

// ---------------------------------------------------------------------------
// Bad input is a state, not a failure
// ---------------------------------------------------------------------------

#[test]
fn app_logs_invalid_utf8_is_decoded_lossily_not_refused() {
    let (root, paths) = test_paths("invalid-utf8");
    let path = crate::cli_lifecycle_log_path(&paths);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut bytes = b"1700000000000 level=info category=runtime action=activate \
                      service_id=<none> message=caf"
        .to_vec();
    bytes.extend_from_slice(&[0xC3, 0x28]);
    bytes.extend_from_slice(b" tail\n");
    fs::write(&path, &bytes).unwrap();

    let response = build_logs(gather_logs(&paths, &bare_redactor()), &LogsQuery::default());

    assert_eq!(response.records.len(), 1);
    assert!(
        response.records[0].summary.contains('\u{FFFD}'),
        "invalid bytes become replacement characters: {:?}",
        response.records[0].summary
    );
    assert!(response.records[0].summary.ends_with(" tail"));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn app_logs_an_over_long_line_is_clipped_with_a_marker() {
    let (root, paths) = test_paths("long-line");
    write_file(
        &crate::cli_lifecycle_log_path(&paths),
        &lifecycle_line(
            1_700_000_000_000,
            "info",
            "activate",
            &"z".repeat(MAX_LINE_BYTES * 2),
        ),
    );

    let response = build_logs(gather_logs(&paths, &bare_redactor()), &LogsQuery::default());

    let summary = &response.records[0].summary;
    assert!(
        summary.ends_with(LINE_TRUNCATION_MARKER),
        "a clipped line must say so"
    );
    assert!(
        summary.len() <= MAX_LINE_BYTES + LINE_TRUNCATION_MARKER.len(),
        "clipped summary was {} bytes",
        summary.len()
    );

    // A multi-byte character straddling the limit must not panic or be split.
    // 8192 is not a multiple of 3, so the boundary walk is exercised.
    let wide = "€".repeat(MAX_LINE_BYTES);
    let clipped = clamp_line(&wide);
    assert!(clipped.ends_with(LINE_TRUNCATION_MARKER));
    assert!(clipped.len() < MAX_LINE_BYTES + LINE_TRUNCATION_MARKER.len());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn app_logs_a_file_that_vanished_leaves_its_source_unavailable() {
    let (root, paths) = test_paths("vanished");
    let present = paths.service_log_path("svc-present");
    let vanished = paths.service_log_path("svc-vanished");
    write_file(&present, "still here\n");
    write_file(&vanished, "about to disappear\n");

    // Exactly the race the contract names: the listing is taken, then the file
    // is rotated away before the read.
    let listing = vec![vanished.clone(), present];
    fs::remove_file(&vanished).unwrap();

    let mixed = scan_files(
        SourceId::Service,
        &listing,
        &paths.services_dir(),
        &bare_redactor(),
    );
    assert!(
        mixed.available,
        "one readable file still makes the source available"
    );
    assert_eq!(mixed.records.len(), 1);
    assert_eq!(mixed.records[0].summary, "still here");

    let all_gone = scan_files(
        SourceId::Service,
        &[vanished],
        &paths.services_dir(),
        &bare_redactor(),
    );
    assert!(!all_gone.available, "nothing read means nothing available");
    assert!(all_gone.records.is_empty());
    let _ = fs::remove_dir_all(root);
}

// ---------------------------------------------------------------------------
// Redaction and disclosure
// ---------------------------------------------------------------------------

#[test]
fn app_logs_redacts_identity_and_credentials_from_every_summary() {
    let (root, paths) = test_paths("redaction");
    write_file(
        &crate::cli_lifecycle_log_path(&paths),
        &lifecycle_line(
            1_700_000_000_000,
            "info",
            "install",
            "plantedusername installed to /home/plantedhomedir/.rocm \
             using api_key=SUPERSECRETPLANTEDKEY",
        ),
    );

    let response = build_logs(
        gather_logs(&paths, &planted_redactor()),
        &LogsQuery {
            reveal_locations: true,
            ..LogsQuery::default()
        },
    );

    let summary = &response.records[0].summary;
    assert!(!summary.contains("plantedusername"), "{summary}");
    assert!(!summary.contains("/home/plantedhomedir"), "{summary}");
    assert!(!summary.contains("SUPERSECRETPLANTEDKEY"), "{summary}");
    assert!(summary.contains("[user]"));
    assert!(summary.contains("~/.rocm"));
    assert!(summary.contains(rocm_core::redact::PLACEHOLDER));

    let locations = response.locations.unwrap();
    assert!(
        locations
            .iter()
            .all(|location| !location.path.contains("plantedhomedir")),
        "disclosed paths are redacted too: {locations:?}"
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn app_logs_withholds_locations_unless_they_are_asked_for() {
    let inputs = inputs_of(vec![scan_of(SourceId::CliAudit, vec![])]);

    assert!(
        build_logs(inputs.clone(), &LogsQuery::default())
            .locations
            .is_none()
    );

    let revealed = build_logs(
        inputs,
        &LogsQuery {
            reveal_locations: true,
            ..LogsQuery::default()
        },
    );
    let locations = revealed.locations.unwrap();
    assert_eq!(locations.len(), 1);
    assert_eq!(locations[0].source, SourceId::CliAudit);
}

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

#[test]
fn app_logs_filters_by_source_severity_time_and_text() {
    let audit = scan_of(
        SourceId::CliAudit,
        vec![
            LogRecord {
                summary: "disk is full".to_owned(),
                ..record(SourceId::CliAudit, 0, 2_000, Severity::Error)
            },
            LogRecord {
                summary: "routine tick".to_owned(),
                ..record(SourceId::CliAudit, 1, 3_000, Severity::Debug)
            },
            LogRecord {
                summary: "old failure".to_owned(),
                ..record(SourceId::CliAudit, 2, 1_000, Severity::Error)
            },
        ],
    );
    let service = scan_of(
        SourceId::Service,
        vec![LogRecord {
            summary: "disk is full here too".to_owned(),
            ..record(SourceId::Service, 0, 2_500, Severity::Error)
        }],
    );
    let inputs = inputs_of(vec![audit, service]);

    let by_severity = build_logs(
        inputs.clone(),
        &LogsQuery {
            min_severity: Some(Severity::Warn),
            ..LogsQuery::default()
        },
    );
    assert_eq!(by_severity.records.len(), 3);
    assert_eq!(by_severity.sources[0].matched, 2);

    let by_since = build_logs(
        inputs.clone(),
        &LogsQuery {
            since_unix_ms: Some(2_000),
            ..LogsQuery::default()
        },
    );
    assert_eq!(by_since.records.len(), 3);
    assert!(
        by_since
            .records
            .iter()
            .all(|record| record.at_unix_ms >= 2_000)
    );

    let by_search = build_logs(
        inputs.clone(),
        &LogsQuery {
            search: Some("DISK".to_owned()),
            ..LogsQuery::default()
        },
    );
    assert_eq!(by_search.records.len(), 2, "search is case-insensitive");

    let by_source = build_logs(
        inputs,
        &LogsQuery {
            sources: vec![SourceId::Service],
            ..LogsQuery::default()
        },
    );
    assert_eq!(by_source.records.len(), 1);
    assert_eq!(
        by_source
            .sources
            .iter()
            .find(|source| source.id == SourceId::CliAudit)
            .unwrap()
            .matched,
        0,
        "an unselected source contributes nothing"
    );
}

#[test]
fn app_logs_names_every_source_in_display_order_even_when_absent() {
    let response = build_logs(inputs_of(vec![]), &LogsQuery::default());
    let ids: Vec<SourceId> = response.sources.iter().map(|source| source.id).collect();

    assert_eq!(ids, SourceId::ALL.to_vec());
    assert!(
        response
            .sources
            .iter()
            .all(|source| !source.label.is_empty())
    );
    assert_eq!(SourceId::parse("cli-client"), Some(SourceId::CliClient));
    assert_eq!(
        SourceId::parse("app-audit"),
        None,
        "a consumer-side source is not one the producer answers for"
    );
}

// ---------------------------------------------------------------------------
// §2 Diagnosis
// ---------------------------------------------------------------------------

fn planted_report() -> rocm_core::DiagnoseReport {
    rocm_core::DiagnoseReport {
        matched: vec![
            rocm_core::Diagnosis {
                id: "fix-4-render-group".to_owned(),
                title: "User not in the render group".to_owned(),
                score: 80,
                evidence: vec!["user plantedusername is not in group render".to_owned()],
                fix: Some(rocm_core::Fix {
                    summary: "Add your user to the render and video groups".to_owned(),
                    commands: vec![
                        "sudo usermod -aG render,video plantedusername".to_owned(),
                        "setx PATH \"%PATH%\"".to_owned(),
                    ],
                    needs_sudo: true,
                    needs_reboot: false,
                    needs_relogin: true,
                    fix_id: "fix-4-render-group".to_owned(),
                    auto_applicable: true,
                    notes: vec!["log out and back in".to_owned()],
                    verify: "id -nG | grep render".to_owned(),
                }),
            },
            rocm_core::Diagnosis {
                id: "fix-6-path-missing".to_owned(),
                title: "ROCm is not on PATH".to_owned(),
                score: 20,
                evidence: vec!["/home/plantedhomedir/.rocm/bin missing".to_owned()],
                fix: None,
            },
        ],
        min_score_for_match: 50,
        high_confidence_threshold: 75,
        route_when_no_match: rocm_core::diagnose::Route {
            target: "rocm-core".to_owned(),
            url: "https://github.com/ROCm/ROCm/issues".to_owned(),
        },
        out_of_scope: None,
    }
}

#[test]
fn app_logs_diagnosis_never_puts_commands_on_the_wire() {
    let response = build_diagnosis(&planted_report(), 1_785_177_626_897, &planted_redactor());
    let json = serde_json::to_string_pretty(&response).unwrap();

    assert!(
        !json.contains("commands"),
        "the app must never receive argv:\n{json}"
    );
    for forbidden in ["sudo", "usermod", "setx"] {
        assert!(
            !json.contains(forbidden),
            "{forbidden:?} leaked into the diagnosis payload:\n{json}"
        );
    }
    // The human-readable half of the fix survives, or the omission would have
    // taken the useful part with it.
    assert!(json.contains("Add your user to the render and video groups"));
    assert!(json.contains("id -nG | grep render"));
}

#[test]
fn app_logs_diagnosis_precomputes_cleared_and_the_match_state() {
    let response = build_diagnosis(&planted_report(), 1_785_177_626_897, &planted_redactor());

    assert_eq!(response.schema_version, SCHEMA_VERSION);
    assert_eq!(response.thresholds.match_score, 50);
    assert_eq!(response.thresholds.high_confidence, 75);
    assert_eq!(response.findings.len(), 2);
    assert!(response.findings[0].cleared, "80 >= 50");
    assert!(!response.findings[1].cleared, "20 < 50");
    assert_eq!(
        response.match_state,
        rocm_core::MatchState::Matched {
            top: "fix-4-render-group".to_owned(),
            score: 80,
            high_confidence: true,
            count: 1,
        }
    );

    let value = serde_json::to_value(&response).unwrap();
    assert_eq!(value["matchState"]["state"], "matched");
    assert_eq!(value["matchState"]["highConfidence"], true);
    assert_eq!(value["thresholds"]["match"], 50);
    assert_eq!(value["findings"][0]["fix"]["fixId"], "fix-4-render-group");
}

#[test]
fn app_logs_diagnosis_redacts_evidence_and_titles() {
    let response = build_diagnosis(&planted_report(), 1_785_177_626_897, &planted_redactor());
    let json = serde_json::to_string(&response).unwrap();

    assert!(!json.contains("plantedusername"), "{json}");
    assert!(!json.contains("/home/plantedhomedir"), "{json}");
    assert!(json.contains("[user]"));
}

#[test]
fn app_logs_diagnosis_reports_an_out_of_scope_host_as_such() {
    let report = rocm_core::DiagnoseReport {
        matched: vec![],
        out_of_scope: Some("running under WSL2".to_owned()),
        ..planted_report()
    };

    let response = build_diagnosis(&report, 1, &bare_redactor());

    assert_eq!(
        response.match_state,
        rocm_core::MatchState::OutOfScope {
            reason: "running under WSL2".to_owned()
        }
    );
    assert_eq!(
        serde_json::to_value(&response).unwrap()["matchState"]["state"],
        "out-of-scope"
    );
}

// ---------------------------------------------------------------------------
// §3 Support bundle
// ---------------------------------------------------------------------------

/// Secrets planted in the inputs. None may appear anywhere in the archive.
const PLANTED_SECRETS: &[&str] = &[
    "SUPERSECRETPLANTEDKEY",
    "sk-PLANTEDABCDEFGHIJKLMNOP",
    "PLANTEDDAEMONTOKEN",
    "PLANTEDAUTHHEADERVALUE",
    "/opt/plantedtoolpath",
    "/home/plantedhomedir",
    "plantedusername",
];

struct BuiltBundle {
    response: BundleResponse,
    /// The *uncompressed* tar stream. A literal search over the gzip bytes
    /// would pass on any input, which is worse than no test at all.
    tar_bytes: Vec<u8>,
    scans: Vec<SourceScan>,
}

/// One real bundle, shared by every bundle test.
///
/// `write_support_bundle` probes the host — including a Python framework probe
/// — and doing that once per assertion would make this file the slowest thing
/// in the suite for no extra coverage.
static BUNDLE: LazyLock<BuiltBundle> = LazyLock::new(|| {
    let (_, paths) = test_paths("bundle");
    write_file(
        &crate::cli_lifecycle_log_path(&paths),
        &lifecycle_line(
            1_700_000_000_000,
            "error",
            "install",
            "plantedusername failed under /home/plantedhomedir \
             with api_key=SUPERSECRETPLANTEDKEY and sk-PLANTEDABCDEFGHIJKLMNOP",
        ),
    );
    write_file(
        &paths.service_log_path("svc-1"),
        "2026-07-27T01:00:00.000Z ERROR vllm: refused to start\n",
    );

    let mut config = RocmCliConfig {
        default_engine: Some("vllm".to_owned()),
        active_runtime_key: Some("nightly-wheel-gfx120x-all-7-14-0".to_owned()),
        onboarding_dismissed: true,
        ..RocmCliConfig::default()
    };
    config.setup.completed = true;
    config.setup.cli_install_dir = Some(PathBuf::from("/home/plantedhomedir/bin"));
    config.dashboard.daemon.token = Some("PLANTEDDAEMONTOKEN".to_owned());
    config.dashboard.tui.chat_auth_header = Some("PLANTEDAUTHHEADERVALUE".to_owned());
    config.tools.insert(
        "uv".to_owned(),
        ManagedToolConfig {
            path: Some(PathBuf::from("/opt/plantedtoolpath")),
            managed: true,
        },
    );
    config
        .providers
        .insert("openai".to_owned(), ProviderUserConfig { enabled: true });

    let redactor = planted_redactor();
    let out = paths.data_dir.join("bundle").join("support.tar.gz");
    let response = write_support_bundle(&paths, &config, &out, "gpu not found", &redactor)
        .expect("support bundle");
    let scans = gather_logs(&paths, &redactor).scans;

    let mut tar_bytes = Vec::new();
    flate2::read::GzDecoder::new(File::open(&out).expect("open bundle"))
        .read_to_end(&mut tar_bytes)
        .expect("gunzip bundle");

    BuiltBundle {
        response,
        tar_bytes,
        scans,
    }
});

fn archive_members(tar_bytes: &[u8]) -> Vec<(String, Vec<u8>)> {
    let mut archive = tar::Archive::new(tar_bytes);
    archive
        .entries()
        .expect("archive entries")
        .map(|entry| {
            let mut entry = entry.expect("archive entry");
            let name = entry.path().expect("entry path").display().to_string();
            let mut data = Vec::new();
            entry.read_to_end(&mut data).expect("entry bytes");
            (name, data)
        })
        .collect()
}

fn member<'a>(members: &'a [(String, Vec<u8>)], name: &str) -> &'a [u8] {
    let Some((_, data)) = members.iter().find(|(entry, _)| entry == name) else {
        panic!("{name} is missing from the bundle");
    };
    data
}

#[test]
fn app_logs_support_bundle_contains_exactly_the_declared_allowlist() {
    let found: BTreeSet<String> = archive_members(&BUNDLE.tar_bytes)
        .into_iter()
        .map(|(name, _)| name)
        .collect();

    assert_eq!(found, expected_bundle_names(&BUNDLE.scans));
    for required in BUNDLE_ENTRIES {
        assert!(found.contains(*required), "missing {required}");
    }
    assert!(found.contains("logs/cli-lifecycle.log"));
    assert!(found.contains("logs/service.log"));
    assert!(
        !found.contains("logs/cli-audit.log"),
        "an unavailable source contributes no excerpt"
    );
}

#[test]
fn app_logs_support_bundle_member_names_cannot_escape_the_bundle_root() {
    for (name, _) in archive_members(&BUNDLE.tar_bytes) {
        assert!(!name.contains(".."), "{name} escapes the bundle root");
        assert!(!name.starts_with('/'), "{name} is absolute");
        assert!(!name.starts_with('\\'), "{name} is absolute");
        assert!(!name.contains(':'), "{name} carries a drive letter");
    }
}

#[test]
fn app_logs_support_bundle_manifest_hashes_match_the_real_member_bytes() {
    let members = archive_members(&BUNDLE.tar_bytes);
    let manifest = &BUNDLE.response.manifest;

    assert!(
        !manifest
            .entries
            .iter()
            .any(|entry| entry.name == "manifest.json"),
        "manifest.json cannot hash itself"
    );
    assert_eq!(manifest.entries.len(), members.len() - 1);

    for entry in &manifest.entries {
        let data = member(&members, &entry.name);
        assert_eq!(entry.sha256, sha256_hex(data), "digest for {}", entry.name);
        assert_eq!(
            entry.bytes,
            u64::try_from(data.len()).unwrap(),
            "size of {}",
            entry.name
        );
    }

    // The archive's own digest covers the file on disk, manifest included.
    assert_eq!(BUNDLE.response.bundle.sha256.len(), 64);
    assert_eq!(BUNDLE.response.schema_version, SCHEMA_VERSION);
    assert_eq!(BUNDLE.response.manifest.schema_version, SCHEMA_VERSION);
    assert!(BUNDLE.response.bundle.bytes > 0);
    assert_eq!(
        BUNDLE.response.manifest.redaction.placeholder,
        rocm_core::redact::PLACEHOLDER
    );
}

#[test]
fn app_logs_support_bundle_contains_none_of_the_planted_secrets() {
    let text = String::from_utf8_lossy(&BUNDLE.tar_bytes);
    for secret in PLANTED_SECRETS {
        assert!(
            !text.contains(secret),
            "{secret} survived into the support bundle"
        );
    }
    assert!(
        text.contains(rocm_core::redact::PLACEHOLDER),
        "something should have been redacted"
    );
}

#[test]
fn app_logs_support_bundle_config_is_an_allowlist_with_a_declared_remainder() {
    let members = archive_members(&BUNDLE.tar_bytes);
    let config: serde_json::Value =
        serde_json::from_slice(member(&members, "config.json")).unwrap();

    let exported: BTreeSet<String> = config.as_object().unwrap().keys().cloned().collect();
    let declared: BTreeSet<String> = serde_json::to_value(SafeConfig {
        default_engine: None,
        default_runtime_id: None,
        active_runtime_key: None,
        previous_runtime_key: None,
        planner_provider: None,
        onboarding_dismissed: false,
        telemetry_mode: String::new(),
        permissions_mode: String::new(),
        setup_completed: false,
        automations_daemon_enabled: false,
        enabled_providers: vec![],
        dashboard_theme: String::new(),
    })
    .unwrap()
    .as_object()
    .unwrap()
    .keys()
    .cloned()
    .collect();
    assert_eq!(exported, declared, "config.json grew a field");

    assert_eq!(config["defaultEngine"], "vllm");
    assert_eq!(config["enabledProviders"][0], "openai");
    assert!(config.get("tools").is_none());
    assert!(config.get("dashboard").is_none());
    assert!(config.get("setup").is_none());

    // Every excluded field is named in the manifest, with a reason.
    let omitted: BTreeSet<&str> = BUNDLE
        .response
        .manifest
        .omitted
        .iter()
        .map(|entry| entry.field.as_str())
        .collect();
    for expected in [
        "tools",
        "engines",
        "dashboard.daemon.token",
        "dashboard.tui.chatAuthHeader",
        "setup.cliInstallDir",
    ] {
        assert!(omitted.contains(expected), "{expected} is not declared");
    }
    assert!(
        BUNDLE
            .response
            .manifest
            .omitted
            .iter()
            .all(|entry| entry.name == "config.json" && !entry.reason.is_empty())
    );
}

#[test]
fn app_logs_support_bundle_accounts_for_every_config_field() {
    let all: BTreeSet<String> = serde_json::to_value(RocmCliConfig::default())
        .unwrap()
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    let accounted: BTreeSet<String> = CONFIG_OMITTED
        .iter()
        .map(|(field, _)| field.split('.').next().unwrap_or(field).to_owned())
        .chain(SAFE_CONFIG_FIELDS.iter().map(|field| (*field).to_owned()))
        .collect();

    let unaccounted: Vec<&String> = all.difference(&accounted).collect();
    assert!(
        unaccounted.is_empty(),
        "these config fields are neither exported nor declared omitted: {unaccounted:?}"
    );
}

#[test]
fn app_logs_support_bundle_reproduction_names_the_host_and_the_symptom() {
    let members = archive_members(&BUNDLE.tar_bytes);
    let reproduction: serde_json::Value =
        serde_json::from_slice(member(&members, "reproduction.json")).unwrap();

    assert_eq!(reproduction["os"], std::env::consts::OS);
    assert_eq!(reproduction["arch"], std::env::consts::ARCH);
    assert_eq!(reproduction["symptom"], "gpu not found");
    assert_eq!(reproduction["command"], "rocm app-support-bundle");
    assert!(reproduction["generatedAtUnixMs"].as_u64().unwrap() > 0);
    assert_eq!(
        reproduction["newestRecordUnixMs"].as_u64(),
        Some(1_785_114_000_000),
        "the newest planted record dates the bundle"
    );
}

#[test]
fn app_logs_support_bundle_log_excerpts_are_bounded_and_redacted() {
    let members = archive_members(&BUNDLE.tar_bytes);
    let excerpt = String::from_utf8_lossy(member(&members, "logs/cli-lifecycle.log"));

    assert!(excerpt.contains("[user]"), "{excerpt}");
    assert!(
        excerpt.contains("~ "),
        "the home root is rewritten: {excerpt}"
    );
    assert!(excerpt.starts_with("1700000000000 error "));
    assert!(u64::try_from(excerpt.len()).unwrap() <= MAX_BYTES_PER_FILE);
}
