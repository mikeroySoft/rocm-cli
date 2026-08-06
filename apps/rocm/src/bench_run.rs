// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm bench run` — the `rocm.bench.v1` measurement primitive.
//!
//! Executes exactly one configuration: launch → wait ready → warm up → drive
//! one load cell → attest → tear down verified-clean → print JSON. It holds no
//! search strategy: it never chooses a configuration, compares two, or retries
//! a different one. Sweeping belongs to the caller.
//!
//! Ephemeral on purpose — no managed-service record, no dashboard CSV. The exit
//! code reports whether the command *ran*; callers branch on `status`.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use rocm_dash_daemon::bench_load::{self, CellResult, LatencyStats, LoadSpec};

/// Share of a device's VRAM that may already be resident before the device
/// counts as occupied.
///
/// Relative, never an absolute MiB figure keyed to a GPU model: a workstation
/// with a display attached always holds one to three GiB for the compositor,
/// while a leaked engine holds most of the card. A fraction separates the two
/// on any device without a per-architecture table to get wrong.
const BUSY_VRAM_FRACTION: f64 = 0.25;

/// Slack over baseline before a post-teardown VRAM re-read counts as leaking.
const CLEAN_VRAM_MIB: u64 = 512;

/// Fraction of failed requests above which a completed cell reads `unhealthy`.
const UNHEALTHY_FAILURE_RATIO: f64 = 0.1;

/// How long the process tree gets to exit on `SIGTERM` before the forced kill.
const TEARDOWN_GRACE: Duration = Duration::from_secs(10);

/// VRAM sampling period while the engine is under load.
const VRAM_POLL: Duration = Duration::from_millis(500);

/// One readiness probe round, short so an engine crash is noticed quickly.
const READY_PROBE: Duration = Duration::from_millis(750);

/// CLI arguments for `rocm bench run`.
pub struct BenchRunArgs {
    pub model_ref: String,
    pub engine: String,
    pub engine_binary: Option<PathBuf>,
    pub engine_arg: Vec<String>,
    pub device: Option<String>,
    pub gpu: Option<String>,
    pub conc: u32,
    pub isl: u32,
    pub osl: u32,
    pub requests: u32,
    pub warmup_requests: u32,
    pub timeout_sec: u64,
}

// ---------------------------------------------------------------- response

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    config: ConfigOut,
    perf: Option<PerfOut>,
    vram: VramOut,
    attested: Option<Attested>,
    artifacts: Artifacts,
    teardown: Teardown,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ErrorOut>,
}

#[derive(Serialize)]
struct ConfigOut {
    model_ref: String,
    weights_path: Option<String>,
    engine: String,
    engine_build_id: Option<String>,
    device: Option<String>,
    engine_args: BTreeMap<String, String>,
    workload: Workload,
}

#[derive(Serialize)]
struct Workload {
    conc: u32,
    isl: u32,
    osl: u32,
    requests: u32,
    warmup_requests: u32,
}

#[derive(Serialize)]
struct PerfOut {
    output_tok_s: Option<f64>,
    prompt_tok_s: Option<f64>,
    ttft_ms: Option<LatencyOut>,
    tpot_ms: Option<LatencyOut>,
    e2e_ms: Option<LatencyOut>,
    requests_completed: u32,
    requests_failed: u32,
    wall_s: Option<f64>,
}

#[derive(Serialize)]
struct LatencyOut {
    mean: f64,
    p50: Option<f64>,
    p99: Option<f64>,
    source: &'static str,
}

impl From<LatencyStats> for LatencyOut {
    fn from(s: LatencyStats) -> Self {
        Self {
            mean: s.mean,
            p50: s.p50,
            p99: s.p99,
            source: if s.source == bench_load::SOURCE_PROMETHEUS {
                bench_load::SOURCE_PROMETHEUS
            } else {
                bench_load::SOURCE_CLIENT
            },
        }
    }
}

// The `_mib` suffixes are the serialized JSON contract, not redundant naming.
#[allow(clippy::struct_field_names)]
#[derive(Serialize)]
struct VramOut {
    peak_mib: Option<u64>,
    baseline_mib: Option<u64>,
    weights_file_mib: Option<u64>,
}

#[derive(Serialize)]
struct Artifacts {
    server_log: String,
    request_log: Option<String>,
}

#[derive(Serialize)]
struct Teardown {
    outcome: &'static str,
    gpu_clean: Option<bool>,
    residual_mib: Option<u64>,
}

#[derive(Serialize)]
struct ErrorOut {
    kind: &'static str,
    message: String,
    log_tail: Vec<String>,
}

// ---------------------------------------------------------------- attested

/// What the engine reported it actually loaded, parsed from its own startup
/// output. Never an echo of the request — see [`parse_attested`].
#[derive(Debug, Default, PartialEq, Serialize)]
struct Attested {
    device: Option<String>,
    /// MiB of model weights the attested device holds, from the buffer line
    /// that names it. `0.00` with a device present is a load that produced no
    /// weights on the accelerator.
    device_buffer_mib: Option<f64>,
    /// MiB of model weights left in host memory. Non-zero on a healthy full
    /// offload too (llama.cpp keeps the mmap'd token embeddings there), so it
    /// is the ratio against `device_buffer_mib` that carries the signal.
    host_buffer_mib: Option<f64>,
    flash_attn: Option<bool>,
    n_gpu_layers: Option<u32>,
    n_layers_total: Option<u32>,
    ctx_size: Option<u32>,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    tensor_types: Option<BTreeMap<String, u32>>,
    speculative: Option<Speculative>,
    unparsed: Vec<String>,
}

#[derive(Debug, Default, PartialEq, Serialize)]
struct Speculative {
    #[serde(rename = "type")]
    kind: Option<String>,
    n_max: Option<u32>,
    p_min: Option<f64>,
}

/// Scratch state threaded through the line matchers.
#[derive(Default)]
struct AttestScratch {
    tensor_types: BTreeMap<String, u32>,
    /// `print_info: n_layer` — the transformer block count, which excludes the
    /// output layer that `offloaded a/b` counts, so it is only a fallback.
    n_layer_hint: Option<u32>,
    n_gpu_layers_hint: Option<u32>,
    spec: Speculative,
    /// Every model-buffer line seen, in log order: `(owner, MiB)`, e.g.
    /// `("ROCm0", Some(2375.91))`. Order matters — see [`parse_attested`].
    buffers: Vec<(String, Option<f64>)>,
}

/// Parse a llama-server–family startup log into the attestation block.
///
/// Takes the log and nothing else, deliberately: the caller diffs the requested
/// config against this block to catch a flag that was accepted and then
/// ignored. A field defaulted to the requested value would make that diff
/// always pass, so anything the log does not state stays `None`.
///
/// Lines that matched no rule are kept in `unparsed`, which is what lets a
/// caller tell "the engine said nothing about flash-attn" apart from "the
/// parser is out of date".
fn parse_attested(log: &str) -> Attested {
    let mut a = Attested::default();
    let mut s = AttestScratch::default();

    for raw in log.lines() {
        let line = raw.trim_end();
        // Startup ends when the socket opens; everything after is request
        // traffic and would drown `unparsed`.
        if line.contains("server is listening") || line.contains("Uvicorn running on") {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        if !attest_line(line, &mut a, &mut s) {
            a.unparsed.push(line.to_string());
        }
    }

    a.n_gpu_layers = a.n_gpu_layers.or(s.n_gpu_layers_hint);
    a.n_layers_total = a.n_layers_total.or(s.n_layer_hint);
    a.tensor_types = (!s.tensor_types.is_empty()).then_some(s.tensor_types);
    a.speculative = (s.spec != Speculative::default()).then_some(s.spec);
    // The accelerator holding the weights is the attestation that matters; a
    // CPU buffer line only answers for what was left behind.
    //
    // Last, not first. A build that fits parameters to free device memory
    // replays the entire load block: b9752 prints one dry pass reporting
    // `0.00 MiB` on device, then the real one. Taking the first match attests
    // the rehearsal and reports zero bytes on the GPU for a perfectly healthy
    // run — the exact reading a CPU-fallback check exists to catch.
    let device = s
        .buffers
        .iter()
        .rev()
        .find(|(owner, _)| !owner.starts_with("CPU"))
        .or_else(|| s.buffers.last());
    a.device = device.map(|(owner, _)| owner.clone());
    // Scoped to the attested device's own lines: on a partial offload the
    // literal last line is the CPU one, and reporting that as the device
    // buffer would invert the signal.
    a.device_buffer_mib = device.and_then(|(owner, _)| {
        s.buffers
            .iter()
            .rev()
            .find(|(other, _)| other == owner)
            .and_then(|(_, mib)| *mib)
    });
    a.host_buffer_mib = s
        .buffers
        .iter()
        .rev()
        .find(|(owner, _)| owner.starts_with("CPU"))
        .and_then(|(_, mib)| *mib);
    a
}

/// Apply every line rule; `false` means "unrecognized" and sends the line to
/// `unparsed`.
fn attest_line(line: &str, a: &mut Attested, s: &mut AttestScratch) -> bool {
    // `llama_model_loader: - type  q4_K:  216 tensors`
    if let Some((_, rest)) = line.split_once("- type ")
        && let Some((name, tail)) = rest.split_once(':')
        && let Some(count) = tail.split_whitespace().next()
        && let Ok(n) = count.parse::<u32>()
    {
        s.tensor_types.insert(name.trim().to_string(), n);
        return true;
    }

    // `load_tensors: offloaded 37/37 layers to GPU`
    if let Some((_, rest)) = line.split_once("offloaded ")
        && let Some((got, total)) = rest
            .split_whitespace()
            .next()
            .and_then(|pair| pair.split_once('/'))
        && let Ok(g) = got.parse::<u32>()
        && let Ok(t) = total.parse::<u32>()
    {
        a.n_gpu_layers = Some(g);
        a.n_layers_total = Some(t);
        return true;
    }

    // `load_tensors:        ROCm0 model buffer size =  2375.91 MiB`
    if let Some((head, tail)) = line.split_once("model buffer size")
        && let Some(owner) = head.split_whitespace().last()
        && owner != ":"
    {
        let mib = tail
            .split_once('=')
            .and_then(|(_, v)| v.split_whitespace().next())
            .and_then(|v| v.parse().ok());
        s.buffers
            .push((owner.trim_end_matches(':').to_string(), mib));
        return true;
    }

    // `srv    load_model: loading draft model '/models/draft.gguf'`
    if line.contains("draft model") {
        let mtp = line.to_ascii_lowercase().contains("mtp");
        s.spec.kind = Some(if mtp { "draft-mtp" } else { "draft-model" }.to_string());
        return true;
    }

    // Everything else llama.cpp prints at startup is `label: key = value`.
    let Some((lhs, value)) = line.split_once('=') else {
        return false;
    };
    let value = value.trim();
    // Also split on a comma, so the one field llama.cpp states mid-sentence —
    // `slot load_model: id  0 | task -1 | new slot, n_ctx = 4096` — is read
    // like any other. A leading `0.00.225.279 I ` timestamp is left behind by
    // the same rule.
    let key = lhs.rsplit([':', ',']).next().unwrap_or(lhs).trim();
    match key {
        "n_ctx" => a.ctx_size = parse_u32(value),
        "n_batch" => a.n_batch = parse_u32(value),
        "n_ubatch" => a.n_ubatch = parse_u32(value),
        "n_layer" => s.n_layer_hint = parse_u32(value),
        "n_gpu_layers" => s.n_gpu_layers_hint = parse_u32(value),
        "flash_attn" => {
            // `flash_attn = auto` is undecided at that point in the log. Leave
            // it unknown and let the line show up in `unparsed`.
            let Some(on) = parse_bool(value) else {
                return false;
            };
            a.flash_attn = Some(on);
        }
        "n_draft" | "draft_max" | "n_draft_max" => s.spec.n_max = parse_u32(value),
        "p_draft_min" | "draft_p_min" | "p_min" => s.spec.p_min = value.parse().ok(),
        "draft_type" | "speculative_type" => s.spec.kind = Some(value.to_string()),
        _ => return false,
    }
    true
}

fn parse_u32(v: &str) -> Option<u32> {
    v.split_whitespace().next()?.parse().ok()
}

fn parse_bool(v: &str) -> Option<bool> {
    match v.to_ascii_lowercase().as_str() {
        "1" | "true" | "enabled" | "on" | "yes" => Some(true),
        "0" | "false" | "disabled" | "off" | "no" => Some(false),
        _ => None,
    }
}

/// `build: 6100 (a1b2c3d)`, `version: 6100 (…)`, or b9752's colonless
/// `common_params_print_info: build 9752 (bddfd2b11) …` → `<engine>-b6100`.
///
/// The colonless form is matched with a leading space so it cannot fire on the
/// tail of a longer word.
fn parse_build_id(log: &str, engine: &str) -> Option<String> {
    log.lines().find_map(|line| {
        let (_, rest) = line
            .split_once("build: ")
            .or_else(|| line.split_once("version: "))
            .or_else(|| line.split_once(" build "))?;
        let n: u32 = rest.split_whitespace().next()?.parse().ok()?;
        Some(format!("{engine}-b{n}"))
    })
}

/// Whether a failed launch was a device out-of-memory rather than a plain
/// startup error. Drives `status: "oom"`, which a caller treats as "this
/// configuration does not fit" instead of "the engine is broken".
fn log_says_oom(log: &str) -> bool {
    let lower = log.to_ascii_lowercase();
    ["out of memory", "failed to allocate", "outofmemory"]
        .iter()
        .any(|needle| lower.contains(needle))
}

// ---------------------------------------------------------------- launching

/// Turn `KEY=VAL` pairs into engine argv.
///
/// Delegates to [`crate::serve_recipe::engine_argv`] so that a configuration
/// measured here is spelled identically when `rocm serve --recipe` replays it.
/// Two renderers that disagree would silently break the recipe's only promise.
fn engine_arg_argv(args: &BTreeMap<String, String>) -> Vec<String> {
    crate::serve_recipe::engine_argv(args)
}

fn parse_engine_args(pairs: &[String]) -> Result<BTreeMap<String, String>> {
    pairs
        .iter()
        .map(|pair| {
            let (key, value) = pair.split_once('=').unwrap_or((pair.as_str(), ""));
            if key.is_empty() {
                bail!("--engine-arg needs a flag name, got {pair:?}");
            }
            Ok((key.to_string(), value.to_string()))
        })
        .collect()
}

/// Full argv for the engine process. vLLM takes an OpenAI-server subcommand;
/// everything else in scope is llama-server–shaped.
fn engine_argv(
    engine: &str,
    model_ref: &str,
    weights: Option<&Path>,
    port: u16,
    device: Option<&str>,
    extra: &BTreeMap<String, String>,
) -> Vec<String> {
    let mut argv: Vec<String> = if engine == "vllm" {
        vec![
            "serve".into(),
            model_ref.into(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            port.to_string(),
        ]
    } else {
        let (model_flag, model_value) = weights.map_or_else(
            || ("-hf", model_ref.to_string()),
            |p| ("-m", p.display().to_string()),
        );
        vec![
            model_flag.into(),
            model_value,
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            port.to_string(),
            // The load generator and the readiness probe both address the
            // model by this name.
            "--alias".into(),
            model_ref.into(),
        ]
    };
    if let Some(dev) = device {
        argv.push("--device".into());
        argv.push(dev.into());
    }
    argv.extend(engine_arg_argv(extra));
    argv
}

/// Default binary for an engine when `--engine-binary` is absent.
const fn default_binary(engine: &str) -> &'static str {
    match engine.as_bytes() {
        b"vllm" => "vllm",
        _ => "llama-server",
    }
}

/// Verbosity threshold benchmark engines are launched with.
///
/// At its default an engine may state none of the fields [`Attested`] exists
/// to carry: b9752 defaults to 3 and prints no offload, batching or
/// flash-attn line at all, so every field is correctly `null` and the
/// attestation diff has nothing to compare. Measured on that build, 4 states
/// all of them in 218 log lines; 5 adds 1,104 debug lines and not one further
/// field.
const LOG_VERBOSITY: &str = "4";

/// Spellings of the llama.cpp verbosity *threshold*, most specific first. Each
/// takes a level, unlike the argument-free `-v` / `--verbose`.
const VERBOSITY_FLAGS: [&str; 3] = ["-lv", "--verbosity", "--log-verbosity"];

/// Argv that raises the engine's log verbosity, or nothing.
///
/// Discovered from `--help` rather than hardcoded, and optional at every step:
/// a build documenting no such flag is launched unchanged and attests whatever
/// it volunteers. Attestation is worth a thinner block, never a failed run.
fn verbosity_argv(engine: &str, binary: &Path, extra: &BTreeMap<String, String>) -> Vec<String> {
    // vLLM has no verbosity threshold and pays a multi-second torch import to
    // say so on `--help`.
    if engine == "vllm" {
        return Vec::new();
    }
    if verbosity_requested(extra) {
        return Vec::new();
    }
    let Ok(help) = ProcessCommand::new(binary).arg("--help").output() else {
        return Vec::new();
    };
    let text =
        String::from_utf8_lossy(&help.stdout).into_owned() + &String::from_utf8_lossy(&help.stderr);
    verbosity_flag(&text)
        .map(|flag| vec![flag.to_string(), LOG_VERBOSITY.to_string()])
        .unwrap_or_default()
}

/// The verbosity-threshold flag a `--help` text documents, or `None`.
fn verbosity_flag(help: &str) -> Option<&'static str> {
    VERBOSITY_FLAGS
        .iter()
        .copied()
        .find(|flag| help.contains(flag))
}

/// Whether the caller already asked for a verbosity level themselves, in which
/// case theirs stands — they may be reproducing a captured log.
fn verbosity_requested(extra: &BTreeMap<String, String>) -> bool {
    extra.keys().any(|key| {
        matches!(
            key.trim_start_matches('-'),
            "lv" | "v" | "verbosity" | "verbose" | "log-verbosity" | "log-verbose"
        )
    })
}

/// Library search path for the engine process: its own directory, then the
/// ROCm SDK, then whatever the caller exported.
///
/// Both prefixes are load-bearing and silent when missing. A prebuilt
/// llama.cpp ships its ggml backends as separate `.so` files beside the
/// binary, and those link `libhipblas`/`librocblas`, which TheRock ships
/// inside Python wheels rather than on the system loader path. A backend that
/// cannot be loaded does not fail the process — it simply never registers, so
/// the device list comes back empty and every token is generated on the CPU
/// at roughly an eighth of the throughput, with no error and no warning.
fn library_path(
    bin_dir: Option<&Path>,
    sdk_dirs: &[PathBuf],
    inherited: Option<OsString>,
) -> Option<OsString> {
    let mut dirs: Vec<PathBuf> = bin_dir.map(PathBuf::from).into_iter().collect();
    dirs.extend_from_slice(sdk_dirs);
    rocm_core::prepend_runtime_paths(&dirs, inherited)
        .ok()
        .flatten()
}

/// ROCm SDK library directories, most specific first.
///
/// Prefers the runtime rocm-cli installed and recorded, which knows which one
/// is active; falls back to reading the managed wheel tree directly so an
/// install this process never registered still resolves.
fn rocm_sdk_lib_dirs() -> Vec<PathBuf> {
    if let Ok(paths) = rocm_core::AppPaths::discover() {
        let config = rocm_core::RocmCliConfig::load(&paths).unwrap_or_default();
        if let Ok(Some(env)) = rocm_core::active_managed_therock_environment(&paths, &config) {
            let dirs = gfx_specific_first(env.library_entries);
            if !dirs.is_empty() {
                return dirs;
            }
        }
    }
    rocm_core::runtime_home_dir()
        .map(|home| wheel_sdk_lib_dirs(&home.join("ROCm").join("runtimes").join("wheel")))
        .unwrap_or_default()
}

/// Order the per-gfx wheel ahead of the generic core wheel.
///
/// rocBLAS is arch-tuned and ships in both, so the specific one has to win the
/// lookup; the registry records them in install order, which puts the generic
/// one first. Stable, so everything else keeps the order the runtime recorded.
/// "Specific before general" — no per-architecture table to fall out of date.
fn gfx_specific_first(mut dirs: Vec<PathBuf>) -> Vec<PathBuf> {
    dirs.sort_by_key(|dir| {
        !dir.components().any(|part| {
            part.as_os_str()
                .to_str()
                .is_some_and(|name| name.starts_with("_rocm_sdk_libraries_"))
        })
    });
    dirs
}

/// `<root>/<runtime>/lib/python3.*/site-packages/{_rocm_sdk_libraries_*,_rocm_sdk_core}/lib`
/// for the newest runtime that has any.
fn wheel_sdk_lib_dirs(root: &Path) -> Vec<PathBuf> {
    let mut runtimes = subdirs(root);
    runtimes.sort();
    for runtime in runtimes.into_iter().rev() {
        let mut dirs: Vec<PathBuf> = subdirs(&runtime.join("lib"))
            .iter()
            .flat_map(|python| subdirs(&python.join("site-packages")))
            .filter(|package| {
                package
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with("_rocm_sdk_libraries_") || name == "_rocm_sdk_core"
                    })
            })
            .map(|package| package.join("lib"))
            .filter(|dir| dir.is_dir())
            .collect();
        if !dirs.is_empty() {
            dirs.sort();
            return gfx_specific_first(dirs);
        }
    }
    Vec::new()
}

/// Immediate subdirectories; an unreadable directory has none.
fn subdirs(dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect()
}

/// An ephemeral loopback port, reserved by binding and immediately releasing.
///
/// ponytail: TOCTOU race with any other process that grabs the port in the
/// microseconds before the engine binds. Upgrade path: pass the listener fd
/// through, once an engine accepts one.
fn free_port() -> Result<u16> {
    let listener =
        std::net::TcpListener::bind("127.0.0.1:0").context("reserving a loopback port")?;
    Ok(listener.local_addr()?.port())
}

/// Poll for readiness, giving up the moment the engine process dies.
///
/// The shared probe only knows about the clock, so a rejected flag that makes
/// llama-server exit in 200 ms would otherwise burn the entire
/// `--timeout-sec`. Returns `(ready, exited)`.
fn wait_ready(
    child: &mut Child,
    port: u16,
    model: &str,
    started: Instant,
    budget: Duration,
) -> (bool, bool) {
    loop {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return (false, true);
        }
        // Not the engine name: bench run drives whatever binary it is pointed
        // at, so this is the generic `/v1/models` probe. Backend attestation
        // comes from the startup log, never from a health payload.
        if crate::wait_for_service_http_ready("bench", "127.0.0.1", port, model, None, READY_PROBE)
        {
            return (true, false);
        }
        if started.elapsed() >= budget {
            return (false, false);
        }
    }
}

/// Seconds since the epoch, so run directories sort chronologically and a
/// recycled PID never appends into an older run's log.
fn run_stamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// `(used, total)` MiB for one amd-smi ordinal.
fn gpu_vram(gpu: u32) -> Option<(u64, u64)> {
    crate::gpu_vram_usage()?
        .into_iter()
        .find(|row| row.index == gpu)
        .map(|row| (row.used_mb, row.total_mb))
}

fn gpu_used_mib(gpu: u32) -> Option<u64> {
    gpu_vram(gpu).map(|(used, _)| used)
}

/// Whether resident VRAM is too high to trust a measurement on this device.
///
/// Unknown totals never block a run: refusing to measure because amd-smi is
/// missing would be a worse failure than measuring on a shared device.
const fn gpu_is_busy(vram: Option<(u64, u64)>) -> bool {
    let Some((used, total)) = vram else {
        return false;
    };
    total > 0 && (used as f64) > (total as f64) * BUSY_VRAM_FRACTION
}

/// Resolve `--gpu`: an explicit amd-smi ordinal, or the emptiest device.
fn resolve_gpu(selection: Option<&str>) -> Result<u32> {
    match selection {
        Some(sel) if sel != "auto" => sel
            .parse()
            .with_context(|| format!("--gpu expects an index or 'auto', got {sel:?}")),
        _ => Ok(crate::gpu_vram_usage()
            .and_then(|rows| {
                rows.into_iter()
                    .max_by_key(|row| row.total_mb.saturating_sub(row.used_mb))
                    .map(|row| row.index)
            })
            .unwrap_or(0)),
    }
}

/// Sentinel stored before the sampler has read anything.
const NO_VRAM_SAMPLE: u64 = u64::MAX;

/// Samples the target GPU's resident VRAM while the engine is under load.
///
/// A thread, not a task: `gpu_vram_usage` shells out to amd-smi and blocks, and
/// one sampler does not justify an async runtime.
///
/// ponytail: one amd-smi subprocess per sample. Upgrade path: the amdsmi
/// library binding, once rocm-cli links it directly.
struct VramSampler {
    peak: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    handle: std::thread::JoinHandle<()>,
}

impl VramSampler {
    fn start(gpu: u32) -> Self {
        let peak = Arc::new(AtomicU64::new(NO_VRAM_SAMPLE));
        let stop = Arc::new(AtomicBool::new(false));
        let handle = {
            let peak = Arc::clone(&peak);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    if let Some(used) = gpu_used_mib(gpu) {
                        let _ = peak.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                            (cur == NO_VRAM_SAMPLE || used > cur).then_some(used)
                        });
                    }
                    std::thread::sleep(VRAM_POLL);
                }
            })
        };
        Self { peak, stop, handle }
    }

    fn finish(self) -> Option<u64> {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.handle.join();
        let peak = self.peak.load(Ordering::Relaxed);
        (peak != NO_VRAM_SAMPLE).then_some(peak)
    }
}

fn log_tail(path: &Path, lines: usize) -> Vec<String> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let all: Vec<&str> = text.lines().collect();
    all[all.len().saturating_sub(lines)..]
        .iter()
        .map(|line| (*line).to_string())
        .collect()
}

/// Terminate the engine tree and report what actually happened.
///
/// Forced, because a survivor still holds the GPU allocation that
/// `teardown.gpu_clean` is about to be measured against.
fn teardown(child: &mut Child, baseline: Option<u64>, gpu: u32) -> Teardown {
    let identity = rocm_core::ProcessIdentity::capture(child.id());
    let outcome =
        rocm_core::terminate_verified(&identity, rocm_core::KillScope::Tree, TEARDOWN_GRACE, true);
    let _ = child.wait();

    let residual = gpu_used_mib(gpu)
        .zip(baseline)
        .map(|(after, base)| after.saturating_sub(base));
    Teardown {
        outcome: outcome.as_str(),
        gpu_clean: residual.map(|mib| mib <= CLEAN_VRAM_MIB),
        residual_mib: residual,
    }
}

// ---------------------------------------------------------------- entrypoint

/// Entry point for `rocm bench run`.
///
/// Always prints one `rocm.bench.v1` document and returns `Ok` when it ran; a
/// configuration that failed to launch is a `status`, not an error.
pub fn run(args: BenchRunArgs) -> Result<()> {
    let started = Instant::now();
    let budget = Duration::from_secs(args.timeout_sec);
    let gpu = resolve_gpu(args.gpu.as_deref())?;
    let engine_args = parse_engine_args(&args.engine_arg)?;

    let weights = Path::new(&args.model_ref)
        .is_file()
        .then(|| PathBuf::from(&args.model_ref));
    let weights_file_mib = weights
        .as_deref()
        .and_then(|p| fs::metadata(p).ok())
        .map(|m| m.len() / (1024 * 1024));

    let run_dir = rocm_core::AppPaths::discover()?
        .data_dir
        .join("bench")
        .join("runs")
        .join(format!("{}-{}", run_stamp(), std::process::id()));
    fs::create_dir_all(&run_dir)
        .with_context(|| format!("creating bench run dir {}", run_dir.display()))?;
    let server_log = run_dir.join("server.log");

    let baseline = gpu_vram(gpu);
    let baseline_mib = baseline.map(|(used, _)| used);

    let mut report = Report {
        schema: "rocm.bench.v1",
        status: "launch_failed",
        config: ConfigOut {
            model_ref: args.model_ref.clone(),
            weights_path: weights.as_ref().map(|p| p.display().to_string()),
            engine: args.engine.clone(),
            engine_build_id: None,
            device: args.device.clone(),
            engine_args: engine_args.clone(),
            workload: Workload {
                conc: args.conc,
                isl: args.isl,
                osl: args.osl,
                requests: args.requests,
                warmup_requests: args.warmup_requests,
            },
        },
        perf: None,
        vram: VramOut {
            peak_mib: None,
            baseline_mib,
            weights_file_mib,
        },
        attested: None,
        artifacts: Artifacts {
            server_log: server_log.display().to_string(),
            // ponytail: run_cell aggregates and does not surface per-request
            // records. Upgrade path: have it return them and write the JSONL.
            request_log: None,
        },
        teardown: Teardown {
            outcome: "not_started",
            gpu_clean: None,
            residual_mib: None,
        },
        error: None,
    };

    // Guarantee 2: refuse to measure on a device someone else is using — a
    // foreign allocation both skews the numbers and can force the engine OOM.
    //
    // ponytail: resident VRAM only, no compute-process identity check. Upgrade
    // path: `AmdSmiCollector::processes` once this command has a runtime.
    if gpu_is_busy(baseline) {
        let (used, total) = baseline.unwrap_or_default();
        report.error = Some(ErrorOut {
            kind: "gpu_busy",
            message: format!("gpu {gpu} already holds {used} of {total} MiB; refusing to launch"),
            log_tail: Vec::new(),
        });
        return emit(&report);
    }

    let binary = args
        .engine_binary
        .clone()
        .unwrap_or_else(|| PathBuf::from(default_binary(&args.engine)));
    let port = free_port()?;
    let mut argv = engine_argv(
        &args.engine,
        &args.model_ref,
        weights.as_deref(),
        port,
        args.device.as_deref(),
        &engine_args,
    );
    argv.extend(verbosity_argv(&args.engine, &binary, &engine_args));

    let log_file = fs::File::create(&server_log)
        .with_context(|| format!("creating {}", server_log.display()))?;
    let mut command = ProcessCommand::new(&binary);
    command
        .args(&argv)
        .env("HIP_VISIBLE_DEVICES", gpu.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file.try_clone()?))
        .stderr(Stdio::from(log_file));
    if let Some(path) = library_path(
        binary.parent(),
        &rocm_sdk_lib_dirs(),
        std::env::var_os("LD_LIBRARY_PATH"),
    ) {
        command.env("LD_LIBRARY_PATH", path);
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            report.error = Some(ErrorOut {
                kind: "spawn_failed",
                message: format!("could not start {}: {e}", binary.display()),
                log_tail: Vec::new(),
            });
            return emit(&report);
        }
    };

    let sampler = VramSampler::start(gpu);

    let (ready, exited) = wait_ready(&mut child, port, &args.model_ref, started, budget);

    let startup_log = fs::read_to_string(&server_log).unwrap_or_default();
    report.config.engine_build_id = parse_build_id(&startup_log, &args.engine);
    report.attested = Some(parse_attested(&startup_log));

    if !ready {
        let oom = log_says_oom(&startup_log);
        report.status = if oom {
            "oom"
        } else if started.elapsed() >= budget {
            "timeout"
        } else {
            "launch_failed"
        };
        report.error = Some(ErrorOut {
            kind: if oom {
                "oom"
            } else if exited {
                "engine_exited"
            } else {
                "not_ready"
            },
            message: if exited {
                format!("engine exited before serving on 127.0.0.1:{port}")
            } else {
                format!("engine did not become ready on 127.0.0.1:{port}")
            },
            log_tail: log_tail(&server_log, 40),
        });
        report.vram.peak_mib = sampler.finish();
        report.teardown = teardown(&mut child, baseline_mib, gpu);
        return emit(&report);
    }

    let spec = LoadSpec {
        endpoint: format!("http://127.0.0.1:{port}/v1"),
        model: args.model_ref.clone(),
        input_len: args.isl,
        output_len: args.osl,
        requests: args.requests,
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building tokio runtime for the load cell")?;

    // Guarantee 3: warmup traffic is issued and thrown away.
    if args.warmup_requests > 0 {
        let warmup = LoadSpec {
            requests: args.warmup_requests,
            ..spec.clone()
        };
        let _ = rt.block_on(bench_load::run_cell(&warmup, args.conc));
    }

    let cell = rt.block_on(bench_load::run_cell_measured(&spec, args.conc));
    report.vram.peak_mib = sampler.finish();

    let cell: CellResult = match cell {
        Ok(cell) => cell,
        Err(e) => {
            report.status = "unhealthy";
            report.error = Some(ErrorOut {
                kind: "load_failed",
                message: e.to_string(),
                log_tail: log_tail(&server_log, 40),
            });
            report.teardown = teardown(&mut child, baseline_mib, gpu);
            return emit(&report);
        }
    };

    let completed = cell.row.n_requests.unwrap_or(0);
    let failed = cell.requests_failed;
    report.perf = Some(PerfOut {
        output_tok_s: cell.row.gen_tps,
        prompt_tok_s: cell.row.prompt_tps,
        ttft_ms: cell.ttft.map(LatencyOut::from),
        tpot_ms: cell.tpot.map(LatencyOut::from),
        e2e_ms: cell.e2e.map(LatencyOut::from),
        requests_completed: completed,
        requests_failed: failed,
        wall_s: cell.row.wall_s,
    });
    report.status = status_for(completed, failed, started.elapsed() >= budget);
    if report.status != "ok" {
        report.error = Some(ErrorOut {
            kind: report.status,
            message: format!("{completed} of {} requests completed", completed + failed),
            log_tail: log_tail(&server_log, 40),
        });
    }

    report.teardown = teardown(&mut child, baseline_mib, gpu);
    emit(&report)
}

/// `timeout` outranks `unhealthy`: an over-budget cell's failures are usually a
/// consequence of the deadline, not an independent fault.
fn status_for(completed: u32, failed: u32, over_budget: bool) -> &'static str {
    if over_budget {
        return "timeout";
    }
    let total = completed + failed;
    if completed == 0 || f64::from(failed) / f64::from(total.max(1)) > UNHEALTHY_FAILURE_RATIO {
        return "unhealthy";
    }
    "ok"
}

fn emit(report: &Report) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Captured `llama-server` startup output (Vulkan backend, Qwen3-family
    /// GGUF), trimmed of repeated `llama_model_loader` metadata rows.
    const LLAMA_STARTUP: &str = r"
build: 6100 (a1b2c3d4) with cc (GCC) 14.2.0 for x86_64-linux-gnu
system info: n_threads = 16, n_threads_batch = 16, total_threads = 32
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon AI PRO R9700 (RADV NAVI48) (radv) | uma: 0 | fp16: 1 | warp size: 64
llama_model_loader: loaded meta data with 30 key-value pairs and 291 tensors
llama_model_loader: - type  f32:   37 tensors
llama_model_loader: - type q4_K:  216 tensors
llama_model_loader: - type q6_K:   38 tensors
print_info: arch             = qwen3
print_info: n_layer          = 36
print_info: n_ctx_train      = 40960
print_info: n_embd           = 4096
load_tensors: offloading 36 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 37/37 layers to GPU
load_tensors:      Vulkan0 model buffer size =  4703.00 MiB
load_tensors:   CPU_Mapped model buffer size =   315.30 MiB
llama_context: n_ctx         = 8192
llama_context: n_batch       = 512
llama_context: n_ubatch      = 512
llama_context: flash_attn    = 1
llama_context: freq_base     = 1000000.0
ggml_vulkan: Compiling shaders...
main: server is listening on http://127.0.0.1:8080 - starting the main loop
srv  log_server_r: request: GET /v1/models 127.0.0.1 200
";

    #[test]
    fn attested_parses_a_real_llama_server_startup_log() {
        let a = parse_attested(LLAMA_STARTUP);

        assert_eq!(a.device.as_deref(), Some("Vulkan0"));
        assert_eq!(a.flash_attn, Some(true));
        // `offloaded 37/37` wins over `n_layer = 36`: the latter excludes the
        // output layer.
        assert_eq!(a.n_gpu_layers, Some(37));
        assert_eq!(a.n_layers_total, Some(37));
        assert_eq!(a.ctx_size, Some(8192));
        assert_eq!(a.n_batch, Some(512));
        assert_eq!(a.n_ubatch, Some(512));

        let types = a.tensor_types.expect("tensor counts");
        assert_eq!(types.get("f32"), Some(&37));
        assert_eq!(types.get("q4_K"), Some(&216));
        assert_eq!(types.get("q6_K"), Some(&38));

        assert_eq!(a.speculative, None, "no draft model was loaded");

        assert!(
            a.unparsed.iter().any(|l| l.contains("Compiling shaders")),
            "unrecognized startup lines must be retained: {:?}",
            a.unparsed
        );
        assert!(
            !a.unparsed.iter().any(|l| l.contains("GET /v1/models")),
            "parsing must stop once the server starts serving"
        );

        assert_eq!(
            parse_build_id(LLAMA_STARTUP, "rocmfpx").as_deref(),
            Some("rocmfpx-b6100")
        );
    }

    #[test]
    fn absent_fields_stay_null_and_never_echo_the_request() {
        // The same run, but this build printed nothing about flash-attn,
        // batching, or offload. The request asked for all of them.
        let requested: BTreeMap<String, String> = [
            ("fa".to_string(), "on".to_string()),
            ("b".to_string(), "512".to_string()),
            ("ub".to_string(), "256".to_string()),
            ("c".to_string(), "8192".to_string()),
            ("ngl".to_string(), "99".to_string()),
        ]
        .into_iter()
        .collect();
        let silent = "\
build: 6100 (a1b2c3d4)
llama_model_loader: loaded meta data with 30 key-value pairs and 291 tensors
print_info: arch             = qwen3
ggml_vulkan: Compiling shaders...
";
        let a = parse_attested(silent);

        assert_eq!(a.flash_attn, None, "must not become the requested `fa=on`");
        assert_eq!(a.n_batch, None, "must not become the requested `b=512`");
        assert_eq!(a.n_ubatch, None, "must not become the requested `ub=256`");
        assert_eq!(a.ctx_size, None, "must not become the requested `c=8192`");
        assert_eq!(
            a.n_gpu_layers, None,
            "must not become the requested `ngl=99`"
        );
        assert_eq!(a.n_layers_total, None);
        assert_eq!(a.device, None);
        assert_eq!(a.tensor_types, None);
        assert_eq!(a.speculative, None);

        // The request is what the tripwire diffs against; nothing above may
        // have leaked from it.
        assert_eq!(requested.len(), 5);
        assert!(a.unparsed.iter().any(|l| l.contains("Compiling shaders")));
    }

    #[test]
    fn undecided_flash_attn_is_unknown_not_false() {
        let a = parse_attested("llama_context: flash_attn    = auto\n");
        assert_eq!(a.flash_attn, None);
        assert!(
            a.unparsed.iter().any(|l| l.contains("auto")),
            "an undecided line must surface in unparsed so the caller sees it"
        );
    }

    #[test]
    fn speculative_decoding_is_attested_when_present() {
        let a = parse_attested(
            "srv    load_model: loading draft model '/models/draft.gguf'\n\
             common_init: n_draft_max = 4\n\
             common_init: draft_p_min = 0.55\n",
        );
        let spec = a.speculative.expect("speculative block");
        assert_eq!(spec.kind.as_deref(), Some("draft-model"));
        assert_eq!(spec.n_max, Some(4));
        assert_eq!(spec.p_min, Some(0.55));
    }

    #[test]
    fn engine_args_become_single_dash_flags() {
        let args = parse_engine_args(&[
            "fa=on".into(),
            "ub=512".into(),
            "--ctx-size=8192".into(),
            "no-mmap".into(),
        ])
        .unwrap();
        // BTreeMap ordering: --ctx-size, fa, no-mmap, ub.
        assert_eq!(
            engine_arg_argv(&args),
            vec!["--ctx-size", "8192", "-fa", "on", "-no-mmap", "-ub", "512"]
        );
        assert!(parse_engine_args(&["=8192".into()]).is_err());
    }

    #[test]
    fn argv_uses_local_weights_when_the_ref_is_a_path() {
        let extra = parse_engine_args(&["fa=on".into()]).unwrap();
        let argv = engine_argv(
            "rocmfpx",
            "qwen3-8b",
            Some(Path::new("/models/q.gguf")),
            8081,
            Some("Vulkan0"),
            &extra,
        );
        assert_eq!(&argv[..2], &["-m", "/models/q.gguf"]);
        assert!(argv.windows(2).any(|w| w == ["--device", "Vulkan0"]));
        assert!(argv.windows(2).any(|w| w == ["-fa", "on"]));

        // No local file: fall back to the hub reference.
        let argv = engine_argv("rocmfpx", "Qwen/Qwen3-8B", None, 8081, None, &extra);
        assert_eq!(&argv[..2], &["-hf", "Qwen/Qwen3-8B"]);
        assert!(!argv.iter().any(|a| a == "--device"));
    }

    #[test]
    fn status_reflects_the_failure_ratio_and_the_deadline() {
        assert_eq!(status_for(64, 0, false), "ok");
        assert_eq!(
            status_for(63, 1, false),
            "ok",
            "1.6% failures is not unhealthy"
        );
        assert_eq!(status_for(50, 14, false), "unhealthy");
        assert_eq!(status_for(0, 64, false), "unhealthy");
        assert_eq!(status_for(64, 0, true), "timeout", "the deadline outranks");
    }

    #[test]
    fn oom_is_distinguished_from_a_plain_startup_failure() {
        assert!(log_says_oom(
            "ggml_vulkan: Device memory allocation of size ... failed\nvk::Device::allocateMemory: ErrorOutOfMemory"
        ));
        assert!(log_says_oom("rocBLAS error: failed to allocate 4096 MiB"));
        assert!(!log_says_oom(
            "error while handling argument \"-ngl\": stoi"
        ));
    }

    #[test]
    fn gpu_busy_is_relative_to_the_devices_own_capacity() {
        // A desktop compositor on a 32 GiB card is not "busy".
        assert!(!gpu_is_busy(Some((2_145, 32_768))));
        // A leaked engine holding most of the same card is.
        assert!(gpu_is_busy(Some((20_000, 32_768))));
        // The same 2 GiB on a 4 GiB card is.
        assert!(gpu_is_busy(Some((2_145, 4_096))));
        // Unknown VRAM must not block a run.
        assert!(!gpu_is_busy(None));
        assert!(!gpu_is_busy(Some((0, 0))));
    }

    /// Captured `llama-server` b9752 startup (ROCm backend, `-lv 10`), trimmed
    /// of the per-layer debug rows and the repeated `llama_model_loader`
    /// metadata but otherwise verbatim — including the timestamp-and-level
    /// prefix and, crucially, **both** load blocks: this build fits parameters
    /// to free device memory in a dry pass before loading for real.
    const B9752_DOUBLED_STARTUP: &str = r"
0.00.093.956 I common_params_print_info: build 9752 (bddfd2b11) with GNU 11.4.0 for Linux x86_64
0.00.093.959 I device_info:
0.00.093.981 I   - ROCm0   : AMD Radeon AI PRO R9700 (32624 MiB, 32492 MiB free)
0.00.095.565 I common_init_result: fitting params to device memory ...
0.00.137.090 I llama_model_loader: - type  f32:  145 tensors
0.00.137.090 I llama_model_loader: - type q4_K:  216 tensors
0.00.137.090 I llama_model_loader: - type q6_K:   37 tensors
0.00.222.460 I print_info: n_layer               = 36
0.00.224.014 I load_tensors: offloading output layer to GPU
0.00.224.015 I load_tensors: offloaded 37/37 layers to GPU
0.00.224.018 I load_tensors:          CPU model buffer size =     0.00 MiB
0.00.224.018 I load_tensors:        ROCm0 model buffer size =     0.00 MiB
0.00.225.279 I llama_context: n_ctx         = 4096
0.00.225.279 I llama_context: n_batch       = 2048
0.00.225.279 I llama_context: n_ubatch      = 512
0.00.225.280 I llama_context: flash_attn    = enabled
0.00.225.999 I llama_kv_cache:      ROCm0 KV buffer size =     0.00 MiB
0.00.255.543 I common_fit_params: fitting params to free memory took 0.16 seconds
0.00.371.345 I print_info: n_layer               = 36
0.00.393.065 I load_tensors: offloading output layer to GPU
0.00.393.067 I load_tensors: offloaded 37/37 layers to GPU
0.00.393.070 I load_tensors:   CPU_Mapped model buffer size =   304.28 MiB
0.00.393.071 I load_tensors:        ROCm0 model buffer size =  2375.91 MiB
0.00.732.494 I llama_context: n_ctx         = 4096
0.00.732.494 I llama_context: n_batch       = 2048
0.00.732.494 I llama_context: n_ubatch      = 512
0.00.732.494 I llama_context: flash_attn    = enabled
0.00.849.761 I llama_kv_cache:      ROCm0 KV buffer size =   576.00 MiB
0.00.935.797 I slot   load_model: id  0 | task -1 | new slot, n_ctx = 4096
0.00.943.535 I srv  llama_server: server is listening on http://127.0.0.1:18098
";

    #[test]
    fn attested_reads_the_real_load_block_not_the_dry_run() {
        let a = parse_attested(B9752_DOUBLED_STARTUP);

        assert_eq!(a.device.as_deref(), Some("ROCm0"));
        assert_eq!(a.n_batch, Some(2048));
        assert_eq!(a.n_ubatch, Some(512));
        assert_eq!(a.flash_attn, Some(true));
        assert_eq!(a.n_gpu_layers, Some(37));
        assert_eq!(a.n_layers_total, Some(37));
        // Stated only by `new slot, n_ctx = 4096` at this build's default
        // verbosity, and mid-sentence at any verbosity.
        assert_eq!(a.ctx_size, Some(4096));

        // The whole point: the dry pass reports `0.00 MiB` on device. Reading
        // the first match attests the rehearsal and hands a CPU-fallback check
        // zero bytes on the GPU for a run that put 2.3 GiB there.
        assert_eq!(
            a.device_buffer_mib,
            Some(2375.91),
            "the last ROCm0 buffer line, not the dry run's 0.00"
        );
        // Owner-scoped, so the CPU line that follows the device line on a
        // partial offload cannot be mistaken for it.
        assert_eq!(a.host_buffer_mib, Some(304.28));

        let types = a.tensor_types.expect("tensor counts");
        assert_eq!(types.get("q4_K"), Some(&216));
        // This build states its number without the colon older ones used.
        assert_eq!(
            parse_build_id(B9752_DOUBLED_STARTUP, "lemonade").as_deref(),
            Some("lemonade-b9752")
        );

        assert!(
            !a.unparsed.iter().any(|l| l.contains("listening")),
            "parsing must stop once the server starts serving"
        );
    }

    /// The same launch at b9752's *default* verbosity, verbatim apart from the
    /// multi-line chat-template block. It is 42 lines and states almost
    /// nothing — which is why bench raises verbosity, and why every field it
    /// does not state has to survive as `null`.
    const B9752_DEFAULT_VERBOSITY: &str = r"
0.00.100.222 I log_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.100.225 I device_info:
0.00.100.248 I   - ROCm0   : AMD Radeon AI PRO R9700 (32624 MiB, 32492 MiB free)
0.00.100.252 I   - CPU     : Intel(R) Core(TM) i9-14900KF (126327 MiB, 126327 MiB free)
0.00.101.818 I srv  llama_server: loading model
0.00.101.889 I common_init_result: fitting params to device memory ...
0.00.930.739 I common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
0.00.976.929 I srv    load_model: initializing slots, n_slots = 4
0.01.002.495 I slot   load_model: id  0 | task -1 | new slot, n_ctx = 4096
0.01.010.233 I srv  llama_server: model loaded
0.01.010.236 I srv  llama_server: server is listening on http://127.0.0.1:18099
";

    #[test]
    fn a_silent_build_attests_nothing_it_did_not_state() {
        // What was asked for. Every value below differs from what the log
        // states, so an echo would be visible rather than a coincidence.
        let requested = parse_engine_args(&[
            "fa=on".into(),
            "b=512".into(),
            "ub=256".into(),
            "c=8192".into(),
            "ngl=999".into(),
        ])
        .unwrap();
        let a = parse_attested(B9752_DEFAULT_VERBOSITY);

        assert_eq!(a.flash_attn, None, "must not become the requested `fa=on`");
        assert_eq!(a.n_batch, None, "must not become the requested `b=512`");
        assert_eq!(a.n_ubatch, None, "must not become the requested `ub=256`");
        assert_eq!(
            a.n_gpu_layers, None,
            "must not become the requested `ngl=999`"
        );
        assert_eq!(a.n_layers_total, None);
        // No `model buffer size` line at this verbosity: the log lists ROCm0
        // under `device_info` but never says the weights landed there, and a
        // device that merely exists is not an attestation.
        assert_eq!(a.device, None);
        assert_eq!(a.device_buffer_mib, None, "never 0.0 by default");
        assert_eq!(a.host_buffer_mib, None);
        assert_eq!(a.tensor_types, None);
        assert_eq!(a.speculative, None);

        // The one field this build does volunteer, and it is read rather than
        // echoed: the request said 8192, the log said 4096.
        assert_eq!(requested.get("c").map(String::as_str), Some("8192"));
        assert_eq!(a.ctx_size, Some(4096));
    }

    #[test]
    fn library_path_puts_the_engine_first_and_the_gfx_wheel_before_core() {
        let root = std::env::temp_dir().join(format!(
            "rocm-bench-libpath-{}",
            rocm_core::unix_time_millis()
        ));
        let engine = root.join("engines/llama-b9752");
        let site = root.join("wheel/nightly-7-14-0/lib/python3.12/site-packages");
        let gfx = site.join("_rocm_sdk_libraries_gfx120X_all/lib");
        let core = site.join("_rocm_sdk_core/lib");
        let inherited = root.join("elsewhere/lib");
        // An older runtime beside it, so "newest first" has something to beat.
        let stale =
            root.join("wheel/nightly-7-13-0/lib/python3.12/site-packages/_rocm_sdk_core/lib");
        for dir in [&engine, &gfx, &core, &inherited, &stale] {
            fs::create_dir_all(dir).expect("temp tree");
        }

        let sdk = wheel_sdk_lib_dirs(&root.join("wheel"));
        assert_eq!(
            sdk,
            vec![gfx.clone(), core.clone()],
            "rocBLAS is arch-tuned: the per-gfx wheel must win the lookup, and \
             the stale runtime must not appear at all"
        );
        // The registry records the generic wheel first; lookup order must not
        // inherit install order.
        assert_eq!(
            gfx_specific_first(vec![core.clone(), gfx.clone()]),
            vec![gfx.clone(), core.clone()]
        );

        let joined = library_path(
            Some(&engine),
            &sdk,
            Some(inherited.clone().into_os_string()),
        )
        .expect("a library path");
        assert_eq!(
            std::env::split_paths(&joined).collect::<Vec<_>>(),
            vec![engine, gfx, core, inherited],
            "the engine's own directory holds the ggml backends and must be \
             searched before anything the caller exported"
        );

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn verbosity_is_discovered_and_never_forced() {
        // b9752's own `--help`, verbatim.
        assert_eq!(
            verbosity_flag(
                "-lv,   --verbosity, --log-verbosity N   Set the verbosity threshold.\n"
            ),
            Some("-lv")
        );
        // A build that documents only the long spelling still gets raised.
        assert_eq!(
            verbosity_flag("      --log-verbosity N   verbosity threshold\n"),
            Some("--log-verbosity")
        );
        // One that documents neither is launched unchanged.
        assert_eq!(verbosity_flag("--log-disable   Log disable\n"), None);

        assert!(verbosity_requested(
            &parse_engine_args(&["lv=2".into()]).unwrap()
        ));
        assert!(verbosity_requested(
            &parse_engine_args(&["--log-verbose".into()]).unwrap()
        ));
        assert!(!verbosity_requested(
            &parse_engine_args(&["fa=on".into(), "ngl=999".into()]).unwrap()
        ));

        // vLLM has no such flag, and probing it costs a torch import.
        assert!(verbosity_argv("vllm", Path::new("vllm"), &BTreeMap::new()).is_empty());
        // A binary that cannot even be run must not fail the launch.
        assert!(
            verbosity_argv(
                "lemonade",
                Path::new("/nonexistent/llama-server"),
                &BTreeMap::new()
            )
            .is_empty()
        );
    }
}
