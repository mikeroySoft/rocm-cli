// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Log excerpts, diagnosis, and support bundles for the ROCm desktop app.
//!
//! # Why this is not `rocm logs`
//!
//! `rocm logs` renders a human page and is free to grow, reformat, and read a
//! whole file. This surface is a machine contract with a GUI that polls it, so
//! it is bounded by construction and versioned like [`app_contract`]. The two
//! share nothing but the [`app_contract::ProducerIdentity`] stamp.
//!
//! # Reading a log must never write one
//!
//! Every earlier path into `AppPaths` in this binary calls
//! [`rocm_core::AppPaths::ensure`] first, because it is about to append. A read
//! that does the same turns "the app opened its log window" into a data
//! directory on a machine that has never run ROCm — and then reports
//! `firstRun: false` for the rest of time, hiding the onboarding screen behind
//! its own side effect. Nothing below creates a directory or a file, and
//! [`app_logs_reading_a_missing_data_dir_creates_nothing`] pins that.
//!
//! [`app_logs_reading_a_missing_data_dir_creates_nothing`]: tests
//!
//! # Bounded at the syscall, not after the read
//!
//! [`read_tail`] seeks to `len - max` before reading. Reading the file into a
//! `String` and slicing afterwards produces the same excerpt while allocating
//! the whole file — and the files this reads are exactly the ones that grow
//! without an operator noticing, which is when a diagnostics panel becomes the
//! thing that OOMs the desktop app.
//!
//! # Two layers, on purpose
//!
//! [`build_logs`] is a **pure function** from already-read [`LogsInputs`] to a
//! [`LogsResponse`]; [`gather_logs`] does the I/O and takes the
//! [`Redactor`] as a parameter. Filtering, paging, and the truncation bookkeeping
//! are therefore reachable from a test with no machine, and a redaction test
//! names its own home directory instead of asserting against whatever user
//! happens to run CI.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result};
use flate2::Compression;
use flate2::write::GzEncoder;
use rocm_core::{AppPaths, Redactor, RocmCliConfig, unix_time_millis};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::app_contract::{self, ComponentReport, ProducerIdentity};

/// Current contract version, shared by all three payloads.
///
/// One number for the surface rather than one per payload: a consumer that
/// understands `app-logs` v1 was written against the same release as the
/// `app-diagnose` v1 it decodes, and three independently drifting versions
/// would let it accept a combination nobody ever built.
pub(crate) const SCHEMA_VERSION: u32 = 1;

/// Most bytes read from any one file.
pub(crate) const MAX_BYTES_PER_FILE: u64 = 256 * 1024;
/// Most lines parsed out of any one file.
pub(crate) const MAX_LINES_PER_FILE: usize = 2000;
/// Most records any single response carries, however many matched.
pub(crate) const MAX_RECORDS_PER_REQUEST: usize = 200;

/// Longest single line kept intact.
///
/// A serialized stack trace or a base64 payload on one line is otherwise the
/// whole per-file budget spent on one unreadable record.
const MAX_LINE_BYTES: usize = 8 * 1024;

/// Appended to a line clipped at [`MAX_LINE_BYTES`], so a reader can tell a
/// clipped line from a log that genuinely ends mid-sentence.
const LINE_TRUNCATION_MARKER: &str = " …[truncated]";

// ---------------------------------------------------------------------------
// Closed vocabularies
// ---------------------------------------------------------------------------

/// How bad a record is.
///
/// Ordered lowest-first so `--severity warn` is a comparison rather than a
/// hand-kept set of "these count as at least warn".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Severity {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl Severity {
    /// The level a token names, or `None` when it names no level at all.
    ///
    /// Strict on purpose: [`parse_tracing`] uses the `None` answer to decide a
    /// line is *not* a tracing record, which is what keeps a continuation line
    /// from being re-dated and re-levelled as if it were a new event.
    pub(crate) fn from_token(token: &str) -> Option<Self> {
        Some(match token.trim().to_ascii_lowercase().as_str() {
            "trace" => Self::Trace,
            "debug" => Self::Debug,
            "info" => Self::Info,
            "warn" | "warning" => Self::Warn,
            "error" | "err" | "fatal" | "critical" => Self::Error,
            _ => return None,
        })
    }

    /// The level a `level=` field names, defaulting to `info`.
    ///
    /// An unrecognised level is a log the reader has not seen before, not a
    /// failed request: refusing the whole response because one writer spelled
    /// its level `NOTICE` would lose the other 1999 lines in the file.
    fn lenient(token: &str) -> Self {
        Self::from_token(token).unwrap_or(Self::Info)
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Trace => "trace",
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
        }
    }
}

/// The log streams this producer answers for.
///
/// Closed and ordered: the app renders the same six rows whether or not the
/// machine has ever written to them, so an empty panel is distinguishable from
/// a panel that lost a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum SourceId {
    CliAudit,
    CliLifecycle,
    CliAction,
    CliClient,
    Service,
    Automation,
}

impl SourceId {
    /// Display order. The app renders sources in this sequence.
    pub(crate) const ALL: [Self; 6] = [
        Self::CliAudit,
        Self::CliLifecycle,
        Self::CliAction,
        Self::CliClient,
        Self::Service,
        Self::Automation,
    ];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::CliAudit => "cli-audit",
            Self::CliLifecycle => "cli-lifecycle",
            Self::CliAction => "cli-action",
            Self::CliClient => "cli-client",
            Self::Service => "service",
            Self::Automation => "automation",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::CliAudit => "ROCm command history",
            Self::CliLifecycle => "CLI activity",
            Self::CliAction => "Per-command logs",
            Self::CliClient => "CLI client log",
            Self::Service => "Service logs",
            Self::Automation => "Automation events",
        }
    }

    /// Resolve a `--source` argument. `None` is a caller error worth an exit
    /// code, not a silently empty result set.
    pub(crate) fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|id| id.as_str() == value)
    }

    /// How a source's lines are shaped on disk.
    const fn format(self) -> LineFormat {
        match self {
            Self::CliAudit | Self::Automation => LineFormat::Jsonl,
            Self::CliLifecycle | Self::CliAction => LineFormat::Lifecycle,
            Self::CliClient | Self::Service => LineFormat::Tracing,
        }
    }
}

/// The three on-disk line shapes this binary and its daemon actually write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineFormat {
    /// `<ms> level=… category=… action=… service_id=… message=…`, as rendered
    /// by `render_cli_lifecycle_log_line`.
    Lifecycle,
    /// One JSON object per line, `AuditEventRecord`/`AutomationEventRecord`-shaped.
    Jsonl,
    /// `tracing_subscriber`'s default text format, plus raw engine output.
    Tracing,
}

// ---------------------------------------------------------------------------
// §1 wire types — `rocm app-logs --json`
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LogsResponse {
    /// Always checked before anything else is decoded.
    pub schema_version: u32,
    pub generated_at_unix_ms: u64,
    /// No data directory at all. The app shows onboarding rather than "no logs".
    pub first_run: bool,
    pub sources: Vec<SourceStatus>,
    pub records: Vec<LogRecord>,
    pub page: PageInfo,
    pub bounds: Bounds,
    /// `None` unless `--reveal-locations` was passed. Absent by default because
    /// a screenshot of the log panel would otherwise carry the user's home path.
    pub locations: Option<Vec<SourceLocation>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SourceStatus {
    pub id: SourceId,
    pub label: String,
    /// Whether at least one of the source's files could actually be read.
    pub available: bool,
    /// How many of this source's records passed the request's filters, before
    /// paging. The app renders per-source counts without re-scanning.
    pub matched: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LogRecord {
    /// Stable within one response; a detail view selects by it.
    pub id: String,
    pub source: SourceId,
    pub at_unix_ms: u64,
    pub severity: Severity,
    /// `None` for sources with no such field, never `""`.
    pub category: Option<String>,
    pub action: Option<String>,
    pub summary: String,
    /// `None` unless the record carries more than `summary` shows.
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PageInfo {
    pub index: usize,
    pub size: usize,
    pub returned: usize,
    pub has_more: bool,
}

/// The limits this response was produced under.
///
/// On the wire rather than assumed by the consumer: an excerpt the reader
/// believes is the whole file is how a support engineer concludes "nothing was
/// logged" from a log that was simply longer than the budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Bounds {
    pub max_bytes_per_file: u64,
    pub max_lines_per_file: usize,
    pub max_records_per_request: usize,
    /// Sources that hit a per-file limit, in display order.
    pub truncated: Vec<SourceId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SourceLocation {
    pub source: SourceId,
    /// Already redacted: `~/.rocm/audit/events.jsonl`, never `/home/someone/…`.
    pub path: String,
}

/// What the caller asked for.
#[derive(Debug, Clone, Default)]
pub(crate) struct LogsQuery {
    /// Empty means every source.
    pub sources: Vec<SourceId>,
    pub min_severity: Option<Severity>,
    pub since_unix_ms: Option<u64>,
    pub search: Option<String>,
    pub page: usize,
    pub page_size: Option<usize>,
    pub reveal_locations: bool,
}

// ---------------------------------------------------------------------------
// §1 inputs and the pure builder
// ---------------------------------------------------------------------------

/// Everything [`build_logs`] needs, already read and already redacted.
#[derive(Debug, Clone)]
pub(crate) struct LogsInputs {
    pub generated_at_unix_ms: u64,
    pub first_run: bool,
    pub scans: Vec<SourceScan>,
}

/// One source's whole contribution: what was found, and what was left behind.
#[derive(Debug, Clone)]
pub(crate) struct SourceScan {
    pub id: SourceId,
    pub available: bool,
    /// A per-file byte or line limit was hit somewhere in this source.
    pub truncated: bool,
    /// Redacted display path, revealed only on request.
    pub path: Option<String>,
    /// Every record read, unfiltered and unpaged.
    pub records: Vec<LogRecord>,
}

/// Filter, count, sort, and page. Pure: no clock, no filesystem.
pub(crate) fn build_logs(inputs: LogsInputs, query: &LogsQuery) -> LogsResponse {
    let selected = |id: SourceId| query.sources.is_empty() || query.sources.contains(&id);

    let mut sources = Vec::with_capacity(SourceId::ALL.len());
    let mut truncated = Vec::new();
    let mut matched = Vec::new();
    let mut locations = Vec::new();

    for id in SourceId::ALL {
        let scan = inputs.scans.iter().find(|scan| scan.id == id);
        let hits: Vec<&LogRecord> = scan
            .filter(|_| selected(id))
            .map(|scan| {
                scan.records
                    .iter()
                    .filter(|record| record_matches(record, query))
                    .collect()
            })
            .unwrap_or_default();

        sources.push(SourceStatus {
            id,
            label: id.label().to_owned(),
            available: scan.is_some_and(|scan| scan.available),
            matched: hits.len(),
        });
        if scan.is_some_and(|scan| scan.truncated) && selected(id) {
            truncated.push(id);
        }
        if let Some(path) = scan.and_then(|scan| scan.path.clone()) {
            locations.push(SourceLocation { source: id, path });
        }
        matched.extend(hits);
    }

    // Newest first: a log panel opens on what just happened, and the
    // per-request cap must therefore drop the oldest records, not the newest.
    matched.sort_by_key(|record| std::cmp::Reverse(record.at_unix_ms));

    let size = query
        .page_size
        .unwrap_or(MAX_RECORDS_PER_REQUEST)
        .clamp(1, MAX_RECORDS_PER_REQUEST);
    let start = query.page.saturating_mul(size);
    let records: Vec<LogRecord> = matched
        .iter()
        .skip(start)
        .take(size)
        .map(|record| (*record).clone())
        .collect();

    LogsResponse {
        schema_version: SCHEMA_VERSION,
        generated_at_unix_ms: inputs.generated_at_unix_ms,
        first_run: inputs.first_run,
        sources,
        page: PageInfo {
            index: query.page,
            size,
            returned: records.len(),
            has_more: start.saturating_add(records.len()) < matched.len(),
        },
        records,
        bounds: Bounds {
            max_bytes_per_file: MAX_BYTES_PER_FILE,
            max_lines_per_file: MAX_LINES_PER_FILE,
            max_records_per_request: MAX_RECORDS_PER_REQUEST,
            truncated,
        },
        locations: query.reveal_locations.then_some(locations),
    }
}

fn record_matches(record: &LogRecord, query: &LogsQuery) -> bool {
    if query.min_severity.is_some_and(|min| record.severity < min) {
        return false;
    }
    if query
        .since_unix_ms
        .is_some_and(|since| record.at_unix_ms < since)
    {
        return false;
    }
    match &query.search {
        None => true,
        Some(needle) => {
            let needle = needle.to_lowercase();
            record.summary.to_lowercase().contains(&needle)
                || record
                    .detail
                    .as_ref()
                    .is_some_and(|detail| detail.to_lowercase().contains(&needle))
        }
    }
}

// ---------------------------------------------------------------------------
// §1 gathering
// ---------------------------------------------------------------------------

/// Read every source off this machine. Never creates, never fails.
///
/// A diagnostics panel that can return an error has a state the UI must render
/// instead of logs, and the honest answer for every failure here is already
/// expressible: the source is unavailable.
pub(crate) fn gather_logs(paths: &AppPaths, redactor: &Redactor) -> LogsInputs {
    LogsInputs {
        generated_at_unix_ms: now_unix_ms(),
        first_run: !paths.data_dir.exists(),
        scans: SourceId::ALL
            .into_iter()
            .map(|id| {
                let (files, location) = source_files(id, paths);
                scan_files(id, &files, &location, redactor)
            })
            .collect(),
    }
}

/// Where a source lives: the files to read, and the location to disclose.
///
/// Listed here rather than inside the scan so the two stay honest about each
/// other — a source whose files all vanished still reports the directory it
/// would have read.
fn source_files(id: SourceId, paths: &AppPaths) -> (Vec<PathBuf>, PathBuf) {
    match id {
        SourceId::CliAudit => {
            let path = paths.audit_events_path();
            (vec![path.clone()], path)
        }
        SourceId::Automation => {
            let path = paths.automation_events_path();
            (vec![path.clone()], path)
        }
        SourceId::CliLifecycle => {
            let path = crate::cli_lifecycle_log_path(paths);
            (vec![path.clone()], path)
        }
        SourceId::CliAction => {
            let dir = paths.client_log_dir().join("cli");
            (files_with_extension(&dir, "log"), dir)
        }
        SourceId::Service => {
            let dir = paths.services_dir();
            (files_with_extension(&dir, "log"), dir)
        }
        SourceId::CliClient => {
            let dir = paths.client_log_dir();
            // Newest only. The rotated siblings are whole days older, and six
            // of them would spend the entire record budget re-reporting a
            // problem the user is looking at right now.
            (newest_rotated_log(&dir).into_iter().collect(), dir)
        }
    }
}

/// Every `*.<extension>` in `dir`, sorted by name for a reproducible response.
fn files_with_extension(dir: &Path, extension: &str) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == extension))
        .collect();
    files.sort();
    files
}

/// The most recent `rocm-cli.log.<date>`.
///
/// Chosen by name, not mtime: the suffix is an ISO date, so lexicographic order
/// *is* chronological order, and a bundle copied between machines does not
/// suddenly pick a different file because `cp` rewrote the timestamps.
fn newest_rotated_log(dir: &Path) -> Option<PathBuf> {
    let prefix = format!("{}.", crate::logging::LOG_FILE_PREFIX);
    fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(&prefix))
        })
        .max()
}

/// Read and parse one source's files.
///
/// `available` turns on only after a file is actually read, so a log that was
/// rotated away between the directory listing and the open reports the source
/// unavailable instead of failing the whole request.
fn scan_files(id: SourceId, files: &[PathBuf], location: &Path, redactor: &Redactor) -> SourceScan {
    let mut records = Vec::new();
    let mut available = false;
    let mut truncated = false;

    for path in files {
        let Ok(tail) = read_tail(path, MAX_BYTES_PER_FILE) else {
            continue;
        };
        available = true;
        truncated |= tail.bytes < tail.file_bytes;

        let total_lines = tail.text.lines().count();
        truncated |= total_lines > MAX_LINES_PER_FILE;
        let skipped = total_lines.saturating_sub(MAX_LINES_PER_FILE);

        push_records(
            id,
            &tail.text.lines().skip(skipped).collect::<Vec<_>>(),
            tail.modified_unix_ms,
            redactor,
            &mut records,
        );
    }

    SourceScan {
        id,
        available,
        truncated,
        path: Some(redactor.text(&location.display().to_string())),
        records,
    }
}

/// The tail of a file, and the two numbers that prove it is a tail.
struct TailRead {
    /// How large the file was when it was opened.
    file_bytes: u64,
    /// How many bytes were actually pulled off disk. Bounded by the caller's
    /// limit whatever `file_bytes` says — that difference is the whole point of
    /// seeking, and it is also what marks the source truncated.
    bytes: u64,
    /// Whole lines only; a partial first line from mid-file is dropped.
    text: String,
    /// The file's mtime, used to date lines that carry no timestamp of their own.
    modified_unix_ms: u64,
}

/// Read at most `max_bytes` from the **end** of `path`.
fn read_tail(path: &Path, max_bytes: u64) -> io::Result<TailRead> {
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    let file_bytes = metadata.len();
    let modified_unix_ms = metadata
        .modified()
        .ok()
        .and_then(|at| at.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |since| {
            u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
        });

    let start = file_bytes.saturating_sub(max_bytes);
    if start > 0 {
        file.seek(SeekFrom::Start(start))?;
    }
    let mut buffer = Vec::with_capacity(usize::try_from(file_bytes.min(max_bytes)).unwrap_or(0));
    let bytes = file.take(max_bytes).read_to_end(&mut buffer)?;

    // Lossy, not strict: a log truncated by a crash mid-codepoint, or one
    // carrying a byte sequence from a subprocess' own encoding, is still the
    // log the user needs to read.
    let lossy = String::from_utf8_lossy(&buffer);
    let text = if start == 0 {
        lossy.into_owned()
    } else if let Some(newline) = lossy.find('\n') {
        // The seek landed mid-line; that fragment is not a record.
        lossy[newline + 1..].to_owned()
    } else {
        String::new()
    };

    Ok(TailRead {
        file_bytes,
        bytes: u64::try_from(bytes).unwrap_or(u64::MAX),
        text,
        modified_unix_ms,
    })
}

/// A line, decomposed but not yet identified.
struct ParsedLine {
    at_unix_ms: u64,
    severity: Severity,
    category: Option<String>,
    action: Option<String>,
    summary: String,
    detail: Option<String>,
}

impl ParsedLine {
    /// A line no parser recognised. Kept verbatim: an unparseable line is
    /// usually the panic message someone opened the log to find.
    fn plain(line: &str) -> Self {
        Self {
            at_unix_ms: 0,
            severity: Severity::Info,
            category: None,
            action: None,
            summary: line.to_owned(),
            detail: None,
        }
    }
}

fn push_records(
    id: SourceId,
    lines: &[&str],
    fallback_unix_ms: u64,
    redactor: &Redactor,
    out: &mut Vec<LogRecord>,
) {
    let format = id.format();
    // Continuation lines of a multi-line event belong to the event above them,
    // so an unstamped line inherits the last stamp rather than landing in 1970
    // and sorting to the bottom of every page.
    let mut last_unix_ms = fallback_unix_ms;

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let line = clamp_line(line);
        let parsed = match format {
            LineFormat::Lifecycle => parse_lifecycle(&line),
            LineFormat::Jsonl => parse_jsonl(&line),
            LineFormat::Tracing => parse_tracing(&line),
        }
        .unwrap_or_else(|| ParsedLine::plain(&line));

        let at_unix_ms = if parsed.at_unix_ms == 0 {
            last_unix_ms
        } else {
            last_unix_ms = parsed.at_unix_ms;
            parsed.at_unix_ms
        };

        out.push(LogRecord {
            id: format!("{}:{}", id.as_str(), out.len()),
            source: id,
            at_unix_ms,
            severity: parsed.severity,
            category: parsed.category,
            action: parsed.action,
            summary: redactor.text(&parsed.summary),
            detail: parsed.detail.map(|detail| redactor.text(&detail)),
        });
    }
}

/// Clip an over-long line at a character boundary, marking that it was clipped.
fn clamp_line(line: &str) -> Cow<'_, str> {
    if line.len() <= MAX_LINE_BYTES {
        return Cow::Borrowed(line);
    }
    let mut end = MAX_LINE_BYTES;
    while !line.is_char_boundary(end) {
        end -= 1;
    }
    Cow::Owned(format!("{}{LINE_TRUNCATION_MARKER}", &line[..end]))
}

/// Parse `render_cli_lifecycle_log_line`'s output.
///
/// Positional against the writer's literal separators rather than a generic
/// `k=v` scan, because `message=` is the rest of the line and may contain both
/// spaces and `=`. A generic scanner truncates every message at its first
/// equals sign.
fn parse_lifecycle(line: &str) -> Option<ParsedLine> {
    let (stamp, rest) = line.split_once(' ')?;
    let at_unix_ms = stamp.parse::<u64>().ok()?;
    let rest = rest.strip_prefix("level=")?;
    let (level, rest) = rest.split_once(" category=")?;
    let (category, rest) = rest.split_once(" action=")?;
    let (action, rest) = rest.split_once(" service_id=")?;
    let (service_id, message) = rest.split_once(" message=")?;

    Some(ParsedLine {
        at_unix_ms,
        severity: Severity::lenient(level),
        category: non_empty(category),
        action: non_empty(action),
        summary: message.to_owned(),
        detail: (service_id != "<none>")
            .then(|| non_empty(service_id).map(|id| format!("service {id}")))
            .flatten(),
    })
}

/// One JSONL event line.
///
/// A tolerant superset of `AuditEventRecord` and `AutomationEventRecord` rather
/// than either type: deserializing the strict types means one added required
/// field in a future release drops every line on the floor, and the two records
/// already disagree about `category` and `watcher_id`.
#[derive(Debug, Default, Deserialize)]
struct JsonlLine {
    #[serde(default)]
    at_unix_ms: u64,
    #[serde(default)]
    level: Option<String>,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    action: Option<String>,
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    service_id: Option<String>,
    #[serde(default)]
    watcher_id: Option<String>,
}

fn parse_jsonl(line: &str) -> Option<ParsedLine> {
    let raw: JsonlLine = serde_json::from_str(line).ok()?;
    let detail = match (&raw.service_id, &raw.watcher_id) {
        (Some(service), _) => Some(format!("service {service}")),
        (None, Some(watcher)) => Some(format!("watcher {watcher}")),
        (None, None) => None,
    };
    Some(ParsedLine {
        at_unix_ms: raw.at_unix_ms,
        severity: raw
            .level
            .as_deref()
            .map_or(Severity::Info, Severity::lenient),
        category: raw.category.and_then(|value| non_empty(&value)),
        action: raw.action.and_then(|value| non_empty(&value)),
        summary: raw.message.unwrap_or_default(),
        detail,
    })
}

/// Parse `tracing_subscriber`'s default text line:
/// `2026-07-27T11:45:00.123456Z  INFO rocm::module: message`.
fn parse_tracing(line: &str) -> Option<ParsedLine> {
    let (stamp, rest) = line.split_once(' ')?;
    let at_unix_ms = parse_iso8601_unix_ms(stamp)?;
    let rest = rest.trim_start();
    let (level, rest) = rest.split_once(' ')?;
    let severity = Severity::from_token(level)?;
    let rest = rest.trim_start();

    // The target is a grouping key the app already has a column for, so it maps
    // to `category` rather than being dissolved back into the message.
    let (category, summary) = match rest.split_once(": ") {
        Some((target, message)) if !target.contains(' ') && !target.is_empty() => {
            (Some(target.to_owned()), message)
        }
        _ => (None, rest),
    };

    Some(ParsedLine {
        at_unix_ms,
        severity,
        category,
        action: None,
        summary: summary.to_owned(),
        detail: None,
    })
}

/// `2026-07-27T11:45:00.123456Z` → epoch milliseconds.
///
/// Hand-rolled because this workspace has no date dependency and adding one to
/// read a timestamp the same process wrote would be the largest thing in the
/// tree. Without it every `cli-client` record dates to 1970 and
/// `--since-unix-ms` silently excludes the newest source.
fn parse_iso8601_unix_ms(stamp: &str) -> Option<u64> {
    let bytes = stamp.as_bytes();
    if bytes.len() < 19
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
    {
        return None;
    }
    let year: i64 = stamp[0..4].parse().ok()?;
    let month: i64 = stamp[5..7].parse().ok()?;
    let day: i64 = stamp[8..10].parse().ok()?;
    let hour: i64 = stamp[11..13].parse().ok()?;
    let minute: i64 = stamp[14..16].parse().ok()?;
    let second: i64 = stamp[17..19].parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    let millis = stamp
        .get(19..)
        .and_then(|rest| rest.strip_prefix('.'))
        .map_or(0, |fraction| {
            (0..3).fold(0_i64, |acc, index| {
                let digit = fraction
                    .as_bytes()
                    .get(index)
                    .copied()
                    .filter(u8::is_ascii_digit)
                    .map_or(0, |byte| i64::from(byte - b'0'));
                acc * 10 + digit
            })
        });

    let seconds = days_from_civil(year, month, day) * 86_400 + hour * 3_600 + minute * 60 + second;
    u64::try_from(seconds * 1_000 + millis).ok()
}

/// Days between `1970-01-01` and a proleptic-Gregorian date (Hinnant's
/// `days_from_civil`). Shifting the era to start in March is what makes the
/// leap day the last day of the year, so no month table is needed.
const fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let shifted_month = (month + 9) % 12;
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

fn non_empty(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

fn now_unix_ms() -> u64 {
    // u64, never u128: serde buffers internally-tagged enum fields through
    // `Content`, which cannot represent u128, and `MatchState` is such an enum.
    u64::try_from(unix_time_millis()).unwrap_or(u64::MAX)
}

// ---------------------------------------------------------------------------
// §2 — `rocm app-diagnose --json`
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DiagnosisResponse {
    pub schema_version: u32,
    pub generated_at_unix_ms: u64,
    /// The catalog's verdict as one closed value, so the app cannot derive
    /// "nothing wrong" from a host the catalog never ran on.
    pub match_state: rocm_core::MatchState,
    pub findings: Vec<Finding>,
    pub route_when_no_match: RouteInfo,
    pub thresholds: Thresholds,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Finding {
    pub id: String,
    pub title: String,
    pub score: i32,
    /// `score >= thresholds.match`, precomputed. Re-deriving a threshold
    /// comparison in each consumer is how two clients end up disagreeing about
    /// whether the same report matched.
    pub cleared: bool,
    pub evidence: Vec<String>,
    pub fix: Option<FixSummary>,
}

/// A fix as the app is allowed to see it.
///
/// **No `commands` field, deliberately.** The app never receives argv: a GUI
/// that holds a shell command eventually runs one, and the only mutation path
/// this product allows is the controller planning by `fixId`. Omitting the
/// field from the type — rather than emptying it at the edge — is what makes
/// that unrepresentable instead of merely conventional.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FixSummary {
    pub fix_id: String,
    pub summary: String,
    pub auto_applicable: bool,
    pub needs_sudo: bool,
    pub needs_reboot: bool,
    pub needs_relogin: bool,
    pub verify: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RouteInfo {
    pub target: String,
    pub url: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Thresholds {
    /// `match` is a keyword in Rust but not on the wire.
    #[serde(rename = "match")]
    pub match_score: i32,
    pub high_confidence: i32,
}

/// Map a catalog report onto the wire. Pure: no probe, no clock.
pub(crate) fn build_diagnosis(
    report: &rocm_core::DiagnoseReport,
    generated_at_unix_ms: u64,
    redactor: &Redactor,
) -> DiagnosisResponse {
    DiagnosisResponse {
        schema_version: SCHEMA_VERSION,
        generated_at_unix_ms,
        match_state: report.match_state(),
        findings: report
            .matched
            .iter()
            .map(|diagnosis| Finding {
                id: diagnosis.id.clone(),
                title: redactor.text(&diagnosis.title),
                score: diagnosis.score,
                cleared: diagnosis.score >= report.min_score_for_match,
                evidence: diagnosis
                    .evidence
                    .iter()
                    .map(|line| redactor.text(line))
                    .collect(),
                fix: diagnosis.fix.as_ref().map(|fix| FixSummary {
                    fix_id: fix.fix_id.clone(),
                    summary: redactor.text(&fix.summary),
                    auto_applicable: fix.auto_applicable,
                    needs_sudo: fix.needs_sudo,
                    needs_reboot: fix.needs_reboot,
                    needs_relogin: fix.needs_relogin,
                    verify: redactor.text(&fix.verify),
                    notes: fix.notes.iter().map(|note| redactor.text(note)).collect(),
                }),
            })
            .collect(),
        route_when_no_match: RouteInfo {
            target: report.route_when_no_match.target.clone(),
            url: redactor.text(&report.route_when_no_match.url),
        },
        thresholds: Thresholds {
            match_score: report.min_score_for_match,
            high_confidence: report.high_confidence_threshold,
        },
    }
}

/// Examine this host and diagnose it.
///
/// `FrameworkProbe::Auto`, matching `rocm diagnose`, not the snapshot's `Skip`:
/// five catalog entries key off the installed framework's ROCm build, and with
/// `Skip` they score zero on a machine that has exactly that problem — a
/// confident "no match" is worse than a slow answer.
pub(crate) fn diagnose_host(
    symptom: &str,
    redactor: &Redactor,
) -> (rocm_core::Examination, DiagnosisResponse) {
    let examination = rocm_core::Examination::probe(rocm_core::FrameworkProbe::Auto);
    let report = rocm_core::run_diagnose(&examination, symptom);
    let response = build_diagnosis(&report, now_unix_ms(), redactor);
    (examination, response)
}

// ---------------------------------------------------------------------------
// §3 — `rocm app-support-bundle`
// ---------------------------------------------------------------------------

/// Every non-log member of a bundle, and the only names allowed beside `logs/`.
///
/// An allowlist, checked against the finished archive by a test. This is the
/// line that actually holds against a leak: free-text redaction fails open on a
/// secret shape nobody anticipated, but a file that is never collected cannot
/// leak whatever it contained.
pub(crate) const BUNDLE_ENTRIES: &[&str] = &[
    "manifest.json",
    "versions.json",
    "examination.json",
    "diagnosis.json",
    "health.json",
    "config.json",
    "reproduction.json",
];

/// Config fields that never enter a bundle, and why.
///
/// Declared, not derived. A field added to [`RocmCliConfig`] next year is absent
/// from `config.json` until somebody puts it in [`SafeConfig`]; this table is
/// what tells the support engineer reading the manifest that the gap was a
/// decision rather than an oversight, so they ask for the value instead of
/// concluding it was unset.
const CONFIG_OMITTED: &[(&str, &str)] = &[
    ("setup.therockVenv", "absolute path"),
    ("setup.cliInstallDir", "absolute path"),
    ("tools", "absolute path"),
    ("engines", "identifier may embed a path"),
    ("automations.watchers", "user-named, not needed for triage"),
    ("dashboard.daemon.listen", "endpoint"),
    ("dashboard.daemon.token", "credential"),
    ("dashboard.daemon.benchResultsDir", "absolute path"),
    ("dashboard.tui.connect", "endpoint"),
    ("dashboard.tui.chatUrl", "endpoint"),
    ("dashboard.tui.chatModel", "not needed for triage"),
    ("dashboard.tui.chatAuthHeader", "credential"),
];

/// The [`RocmCliConfig`] fields [`SafeConfig`] draws from.
///
/// Together with [`CONFIG_OMITTED`] this must account for every top-level
/// config field, and a test asserts it does. Without that tripwire the
/// allowlist fails *silently* rather than closed: a field added next year is
/// correctly absent from the bundle, but nothing tells the support engineer it
/// exists, so a missing value reads as an unset one.
#[cfg(test)]
const SAFE_CONFIG_FIELDS: &[&str] = &[
    "default_engine",
    "default_runtime_id",
    "active_runtime_key",
    "previous_runtime_key",
    "planner_provider",
    "onboarding_dismissed",
    "telemetry",
    "permissions",
    "setup",
    "automations",
    "providers",
    "dashboard",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BundleResponse {
    pub schema_version: u32,
    pub bundle: BundleFile,
    pub manifest: BundleManifest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BundleFile {
    /// Redacted: the manifest is the first thing pasted into a public issue.
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BundleManifest {
    pub schema_version: u32,
    pub generated_at_unix_ms: u64,
    pub producer: ProducerIdentity,
    /// Every member except `manifest.json`, which cannot hash itself.
    pub entries: Vec<BundleEntry>,
    pub redaction: RedactionInfo,
    pub omitted: Vec<OmittedField>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BundleEntry {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RedactionInfo {
    pub placeholder: String,
    /// Identity the redactor refused to substitute because it was too short to
    /// do so without mangling ordinary text. Reported so a reviewer can look
    /// for it by hand rather than assume the bundle is clean.
    pub identity_skipped: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OmittedField {
    pub name: String,
    pub field: String,
    pub reason: String,
}

/// The subset of [`RocmCliConfig`] that cannot carry a credential, an endpoint,
/// or an absolute path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SafeConfig {
    pub default_engine: Option<String>,
    pub default_runtime_id: Option<String>,
    pub active_runtime_key: Option<String>,
    pub previous_runtime_key: Option<String>,
    pub planner_provider: Option<String>,
    pub onboarding_dismissed: bool,
    pub telemetry_mode: String,
    pub permissions_mode: String,
    pub setup_completed: bool,
    pub automations_daemon_enabled: bool,
    /// Names only. `ProviderUserConfig` is a single `enabled` flag; keys live in
    /// the OS keyring and were never in this file.
    pub enabled_providers: Vec<String>,
    pub dashboard_theme: String,
}

/// Producer identity plus what is installed, so a bundle is readable without
/// the machine that produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Versions {
    pub producer: ProducerIdentity,
    pub components: Vec<ComponentReport>,
}

/// What the user was doing and where. Enough to reproduce, nothing to identify.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Reproduction {
    pub os: String,
    pub arch: String,
    pub symptom: Option<String>,
    pub command: String,
    pub generated_at_unix_ms: u64,
    /// The newest log record in the bundle. Paired with `generatedAtUnixMs`
    /// this answers "how stale is this?", which is the difference between a
    /// bundle of the failure and a bundle of the week after it.
    pub newest_record_unix_ms: Option<u64>,
}

pub(crate) fn safe_config(config: &RocmCliConfig) -> SafeConfig {
    SafeConfig {
        default_engine: config.default_engine.clone(),
        default_runtime_id: config.default_runtime_id.clone(),
        active_runtime_key: config.active_runtime_key.clone(),
        previous_runtime_key: config.previous_runtime_key.clone(),
        planner_provider: config.planner_provider.clone(),
        onboarding_dismissed: config.onboarding_dismissed,
        telemetry_mode: config.telemetry.mode_label().to_owned(),
        permissions_mode: config.permissions.mode_label().to_owned(),
        setup_completed: config.setup.completed,
        automations_daemon_enabled: config.automations.daemon_enabled,
        enabled_providers: config
            .providers
            .iter()
            .filter(|(_, provider)| provider.enabled)
            .map(|(name, _)| name.clone())
            .collect(),
        dashboard_theme: config.dashboard.tui.theme.clone(),
    }
}

/// Build the whole bundle and write it to `out`.
pub(crate) fn write_support_bundle(
    paths: &AppPaths,
    config: &RocmCliConfig,
    out: &Path,
    symptom: &str,
    redactor: &Redactor,
) -> Result<BundleResponse> {
    let generated_at_unix_ms = now_unix_ms();
    let (examination, diagnosis) = diagnose_host(symptom, redactor);
    let snapshot = app_contract::build_snapshot(app_contract::gather_inputs(paths, config)?);
    let scans = gather_logs(paths, redactor).scans;

    let mut members: Vec<(String, Vec<u8>)> = vec![
        (
            "versions.json".to_owned(),
            to_json_bytes(&Versions {
                producer: ProducerIdentity::current(),
                components: snapshot.components.clone(),
            })?,
        ),
        (
            "examination.json".to_owned(),
            to_json_bytes(&redactor.value(&examination)?)?,
        ),
        ("diagnosis.json".to_owned(), to_json_bytes(&diagnosis)?),
        (
            "health.json".to_owned(),
            to_json_bytes(&redactor.value(&snapshot)?)?,
        ),
        (
            "config.json".to_owned(),
            to_json_bytes(&redactor.value(&safe_config(config))?)?,
        ),
        (
            "reproduction.json".to_owned(),
            to_json_bytes(&Reproduction {
                os: std::env::consts::OS.to_owned(),
                arch: std::env::consts::ARCH.to_owned(),
                symptom: non_empty(symptom).map(|text| redactor.text(&text)),
                command: "rocm app-support-bundle".to_owned(),
                generated_at_unix_ms,
                newest_record_unix_ms: scans
                    .iter()
                    .flat_map(|scan| scan.records.iter())
                    .map(|record| record.at_unix_ms)
                    .max(),
            })?,
        ),
    ];
    for scan in &scans {
        if scan.available {
            members.push((
                format!("logs/{}.log", scan.id.as_str()),
                render_excerpt(scan).into_bytes(),
            ));
        }
    }

    // The allowlist is a gate, not a comment. Checked here rather than only in
    // a test, because the failure it guards against — a member added upstream
    // of this line and shipped to a public issue tracker — is not one you get
    // to notice after the fact.
    let names: BTreeSet<String> = std::iter::once("manifest.json".to_owned())
        .chain(members.iter().map(|(name, _)| name.clone()))
        .collect();
    let allowed = expected_bundle_names(&scans);
    anyhow::ensure!(
        names == allowed,
        "support bundle members do not match the declared allowlist: {:?}",
        names.symmetric_difference(&allowed).collect::<Vec<_>>()
    );

    let manifest = BundleManifest {
        schema_version: SCHEMA_VERSION,
        generated_at_unix_ms,
        producer: ProducerIdentity::current(),
        entries: members
            .iter()
            .map(|(name, data)| BundleEntry {
                name: name.clone(),
                bytes: u64::try_from(data.len()).unwrap_or(u64::MAX),
                sha256: sha256_hex(data),
            })
            .collect(),
        redaction: RedactionInfo {
            placeholder: rocm_core::redact::PLACEHOLDER.to_owned(),
            identity_skipped: redactor.skipped_identity.clone(),
        },
        omitted: CONFIG_OMITTED
            .iter()
            .map(|(field, reason)| OmittedField {
                name: "config.json".to_owned(),
                field: (*field).to_owned(),
                reason: (*reason).to_owned(),
            })
            .collect(),
    };

    write_archive(out, &manifest, &members, generated_at_unix_ms / 1000)?;

    let archive = fs::read(out).with_context(|| format!("failed to read {}", out.display()))?;
    Ok(BundleResponse {
        schema_version: SCHEMA_VERSION,
        bundle: BundleFile {
            path: redactor.text(&out.display().to_string()),
            bytes: u64::try_from(archive.len()).unwrap_or(u64::MAX),
            sha256: sha256_hex(&archive),
        },
        manifest,
    })
}

/// One bounded, already-redacted plain-text excerpt per source.
fn render_excerpt(scan: &SourceScan) -> String {
    let mut out = String::new();
    for record in &scan.records {
        let _ = writeln!(
            out,
            "{} {} {}",
            record.at_unix_ms,
            record.severity.as_str(),
            record.summary
        );
        if let Some(detail) = &record.detail {
            let _ = writeln!(out, "    {detail}");
        }
    }
    out
}

fn write_archive(
    out: &Path,
    manifest: &BundleManifest,
    members: &[(String, Vec<u8>)],
    mtime_secs: u64,
) -> Result<()> {
    if let Some(parent) = out.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let file = File::create(out).with_context(|| format!("failed to create {}", out.display()))?;
    let mut builder = tar::Builder::new(GzEncoder::new(file, Compression::default()));

    // First, so an operator reading the archive sequentially learns what is in
    // it and what was left out before seeing any of the data.
    append_member(
        &mut builder,
        "manifest.json",
        &to_json_bytes(manifest)?,
        mtime_secs,
    )?;
    for (name, data) in members {
        append_member(&mut builder, name, data, mtime_secs)?;
    }

    builder
        .into_inner()
        .context("failed to finish support bundle archive")?
        .finish()
        .context("failed to flush support bundle compression")?;
    Ok(())
}

fn append_member(
    builder: &mut tar::Builder<GzEncoder<File>>,
    name: &str,
    data: &[u8],
    mtime_secs: u64,
) -> Result<()> {
    let mut header = tar::Header::new_gnu();
    header.set_size(u64::try_from(data.len()).unwrap_or(0));
    header.set_mode(0o644);
    // A fixed mtime from the manifest, not `SystemTime::now()`: two bundles of
    // the same state should differ only where the state differs.
    header.set_mtime(mtime_secs);
    header.set_cksum();
    builder
        .append_data(&mut header, name, data)
        .with_context(|| format!("failed to add {name} to the support bundle"))?;
    Ok(())
}

fn to_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    serde_json::to_vec_pretty(value).context("failed to serialize a support bundle member")
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Print a payload for the app's reader.
///
/// `--json` is compact because the app parses it; the default is pretty because
/// the only other reader of a hidden contract command is a human debugging why
/// the app disagrees with the CLI.
pub(crate) fn print_json<T: Serialize>(value: &T, compact: bool) -> Result<()> {
    if compact {
        println!("{}", serde_json::to_string(value)?);
    } else {
        println!("{}", serde_json::to_string_pretty(value)?);
    }
    Ok(())
}

/// The set of every name a finished bundle may contain, given which sources
/// were available. Shared with the test that enumerates the real archive.
pub(crate) fn expected_bundle_names(scans: &[SourceScan]) -> BTreeSet<String> {
    BUNDLE_ENTRIES
        .iter()
        .map(|name| (*name).to_owned())
        .chain(
            scans
                .iter()
                .filter(|scan| scan.available)
                .map(|scan| format!("logs/{}.log", scan.id.as_str())),
        )
        .collect()
}

#[cfg(test)]
mod tests;
