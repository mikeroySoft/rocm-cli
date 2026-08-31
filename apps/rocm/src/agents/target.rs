// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use super::transaction::unique_token;
use super::{AgentHarness, AgentProtocol, ResolvedTarget, VersionInfo};
use anyhow::{Context, Result, anyhow, bail};
use rocm_core::{
    AppPaths, DEFAULT_LOCAL_HOST, DEFAULT_LOCAL_PORT, IdentityState, KillScope,
    ManagedServiceRecord, ProcessIdentity, identity_state, terminate_verified,
};
use semver::Version;
use serde_json::json;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io::{ErrorKind, Read, Seek, SeekFrom};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const VERSION_TIMEOUT: Duration = Duration::from_secs(3);
const MODEL_LIST_TIMEOUT: Duration = Duration::from_secs(3);
const PROTOCOL_TIMEOUT: Duration = Duration::from_secs(30);
const HARNESS_TEST_TIMEOUT: Duration = Duration::from_mins(2);
const COMMAND_POLL_INTERVAL: Duration = Duration::from_millis(25);
const PROCESS_STOP_GRACE: Duration = Duration::from_secs(2);
const MAX_CAPTURE_BYTES: u64 = 256 * 1024;
const TEST_TIMEOUT_ENV: &str = "ROCM_CLI_AGENT_TEST_TIMEOUT_SECS";
const TARGET_FALLBACK_URL_ENV: &str = "ROCM_CLI_AGENT_TARGET_FALLBACK_URL";

pub(super) fn detect_version(
    harness: AgentHarness,
    override_version: Option<&str>,
) -> Result<VersionInfo> {
    let executable = find_executable(harness.executable());
    if let Some(value) = override_version {
        let version = Version::parse(value.trim())
            .with_context(|| format!("invalid semantic version override `{value}`"))?;
        return Ok(VersionInfo {
            executable,
            supported: harness.supports_version(&version),
            version: Some(version),
            source: "override",
        });
    }

    let Some(path) = executable else {
        return Ok(VersionInfo {
            executable: None,
            version: None,
            source: "not installed",
            supported: false,
        });
    };
    let version = match capture_command(&path, &["--version"], None, VERSION_TIMEOUT, &[]) {
        Ok(output) if output.status.success() => {
            first_semantic_version(&format!("{}\n{}", output.stdout, output.stderr))
        }
        Ok(_) => None,
        Err(error) if has_io_error_kind(&error, ErrorKind::NotFound) => {
            return Ok(VersionInfo {
                executable: None,
                version: None,
                source: "not installed",
                supported: false,
            });
        }
        Err(_) => None,
    };
    let supported = version
        .as_ref()
        .is_some_and(|version| harness.supports_version(version));
    Ok(VersionInfo {
        executable: Some(path),
        version,
        source: "detected",
        supported,
    })
}

pub(super) fn resolve_target(
    paths: &AppPaths,
    model: Option<&str>,
    base_url: Option<&str>,
) -> Result<ResolvedTarget> {
    let explicit_model = model.map(str::trim).filter(|value| !value.is_empty());
    if model.is_some() && explicit_model.is_none() {
        bail!("model identifier cannot be empty");
    }

    if let Some(base_url) = base_url {
        let (origin, api_base) = normalize_loopback_url(base_url)?;
        let model = match explicit_model {
            Some(model) => model.to_owned(),
            None => resolve_advertised_model(&origin).with_context(|| {
                format!("failed to resolve a model from the local endpoint {api_base}")
            })?,
        };
        return Ok(ResolvedTarget {
            api_base,
            origin,
            model,
            managed_engine: None,
            service_id: None,
        });
    }

    let mut candidates = ready_loopback_services(paths)?;
    if let Some(model) = explicit_model {
        candidates.retain(|record| record.canonical_model_id == model || record.model_ref == model);
    }
    if candidates.len() > 1 {
        candidates.sort_by(|left, right| left.service_id.cmp(&right.service_id));
        let choices = candidates
            .iter()
            .map(|record| {
                let model = if record.canonical_model_id.trim().is_empty() {
                    record.model_ref.as_str()
                } else {
                    record.canonical_model_id.as_str()
                };
                format!("{} ({model} at {})", record.service_id, record.endpoint_url)
            })
            .collect::<Vec<_>>()
            .join(", ");
        bail!(
            "more than one ready loopback ROCm service matches; use --model or --base-url to choose one: {choices}"
        );
    }

    if let Some(record) = candidates.pop() {
        let (origin, api_base) = normalize_loopback_url(&record.endpoint_url)?;
        let model = match explicit_model {
            Some(model) => model.to_owned(),
            None if !record.canonical_model_id.trim().is_empty() => {
                record.canonical_model_id.clone()
            }
            None => resolve_advertised_model(&origin).with_context(|| {
                format!(
                    "managed service {} has no canonical model and its endpoint did not identify one",
                    record.service_id
                )
            })?,
        };
        return Ok(ResolvedTarget {
            api_base,
            origin,
            model,
            managed_engine: Some(record.engine),
            service_id: Some(record.service_id),
        });
    }

    let fallback = env::var(TARGET_FALLBACK_URL_ENV)
        .unwrap_or_else(|_| format!("http://{DEFAULT_LOCAL_HOST}:{DEFAULT_LOCAL_PORT}"));
    let (origin, api_base) = normalize_loopback_url(&fallback)?;
    let model = match explicit_model {
        Some(model) => model.to_owned(),
        None => resolve_advertised_model(&origin).with_context(|| {
            format!("no ready ROCm model server was found at {api_base}; run `rocm serve <model>`")
        })?,
    };
    Ok(ResolvedTarget {
        api_base,
        origin,
        model,
        managed_engine: None,
        service_id: None,
    })
}

pub(super) fn check_target(harness: AgentHarness, target: &ResolvedTarget) -> Result<()> {
    let (url, body) = match harness.protocol() {
        AgentProtocol::ChatCompletions => (
            format!("{}/chat/completions", target.api_base),
            json!({
                "model": target.model,
                "messages": [{"role": "user", "content": "Reply with ok."}],
                "max_tokens": 2,
                "stream": false,
            }),
        ),
        AgentProtocol::Responses => (
            format!("{}/responses", target.api_base),
            json!({
                "model": target.model,
                "input": "Reply with ok.",
                "max_output_tokens": 2,
                "stream": false,
            }),
        ),
        AgentProtocol::AnthropicMessages => (
            format!("{}/v1/messages", target.origin),
            json!({
                "model": target.model,
                "messages": [{"role": "user", "content": "Reply with ok."}],
                "max_tokens": 2,
                "stream": false,
            }),
        ),
    };
    let agent = ureq::AgentBuilder::new()
        .redirects(0)
        .try_proxy_from_env(false)
        .build();
    let mut request = agent
        .post(&url)
        .timeout(PROTOCOL_TIMEOUT)
        .set("Content-Type", "application/json")
        .set("Accept", "application/json");
    if harness.protocol() == AgentProtocol::AnthropicMessages {
        request = request
            .set("anthropic-version", "2023-06-01")
            .set("x-api-key", "rocm-local");
    }
    let payload = serde_json::to_string(&body).context("failed to serialize protocol probe")?;
    match request.send_string(&payload) {
        Ok(response) if (200..300).contains(&response.status()) => Ok(()),
        Ok(response) => bail!(
            "{} protocol check for model `{}` returned HTTP {} from {url}",
            harness.canonical_name(),
            target.model,
            response.status()
        ),
        Err(ureq::Error::Status(status, _)) => bail!(
            "{} protocol check for model `{}` returned HTTP {status} from {url}",
            harness.canonical_name(),
            target.model
        ),
        Err(error) => Err(error).with_context(|| {
            format!(
                "{} protocol check for model `{}` could not reach {url}",
                harness.canonical_name(),
                target.model
            )
        }),
    }
}

pub(super) fn run_harness_test(
    harness: AgentHarness,
    version: &VersionInfo,
    configured_model: Option<&str>,
) -> Result<()> {
    let executable = version.executable.as_deref().ok_or_else(|| {
        anyhow!(
            "{} executable `{}` is not installed",
            harness.canonical_name(),
            harness.executable()
        )
    })?;
    let temp = TempRoot::create("agent-test")?;
    let workspace = temp.path.join("workspace");
    fs::create_dir(&workspace).with_context(|| {
        format!(
            "failed to create temporary workspace {}",
            workspace.display()
        )
    })?;
    let nonce = unique_token("agent-probe");
    let probe_path = workspace.join("probe.txt");
    fs::write(&probe_path, format!("{nonce}\n"))
        .with_context(|| format!("failed to write temporary probe {}", probe_path.display()))?;
    make_probe_read_only(&probe_path)?;

    let prompt = "Return exactly the contents of probe.txt using only the file access provided by this harness invocation. Do not write files, run shell commands, or contact external services.";
    let timeout = harness_test_timeout();
    let timeout_secs = timeout.as_secs().max(1).to_string();
    let workspace_text = workspace.to_string_lossy().into_owned();
    let mut owned_args = Vec::<String>::new();
    let mut env_overrides = Vec::<(&str, &str)>::new();
    match harness {
        AgentHarness::Claude => {
            owned_args.extend(strings(&[
                "-p",
                prompt,
                "--output-format",
                "text",
                "--permission-mode",
                "dontAsk",
                "--tools",
                "Read",
                "--no-session-persistence",
            ]));
        }
        AgentHarness::Hermes => {
            owned_args.extend(strings(&[
                "chat",
                "--oneshot",
                "--quiet",
                "--toolsets",
                "file",
                "--max-turns",
                "4",
                "--ignore-rules",
                "--source",
                "tool",
                "-q",
                prompt,
            ]));
        }
        AgentHarness::OpenClaw => {
            owned_args.extend(strings(&[
                "agent",
                "exec",
                "--cwd",
                &workspace_text,
                "--code-mode",
                "direct",
                "--local-model-lean",
                "--timeout",
                &timeout_secs,
                prompt,
            ]));
        }
        AgentHarness::Codex => {
            owned_args.extend(strings(&[
                "exec",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                prompt,
            ]));
        }
        AgentHarness::OpenCode => {
            owned_args.extend(strings(&[
                "run",
                "--format",
                "default",
                "--title",
                "rocm-agent-probe",
                prompt,
            ]));
            env_overrides.push(("OPENCODE_PERMISSION", r#"{"*":"deny","read":"allow"}"#));
            env_overrides.push(("OPENCODE_DISABLE_DEFAULT_PLUGINS", "true"));
        }
        AgentHarness::QwenCode => {
            owned_args.extend(strings(&[
                "--prompt",
                prompt,
                "--approval-mode",
                "plan",
                "--output-format",
                "text",
            ]));
        }
        AgentHarness::Aider => {
            owned_args.extend(strings(&[
                "--message",
                prompt,
                "--read",
                "probe.txt",
                "--no-git",
                "--no-auto-commits",
                "--no-dirty-commits",
                "--no-check-update",
            ]));
        }
        AgentHarness::Continue => {
            owned_args.extend(strings(&[
                "-p",
                prompt,
                "--exclude",
                "Bash",
                "--exclude",
                "Edit",
                "--exclude",
                "MultiEdit",
                "--exclude",
                "Write",
                "--exclude",
                "Fetch",
                "--exclude",
                "UploadArtifact",
            ]));
        }
        AgentHarness::Pi => {
            let configured_model = configured_model
                .ok_or_else(|| anyhow!("pi harness test requires a configured model"))?;
            owned_args.extend(strings(&[
                "-p",
                "--no-session",
                "--no-approve",
                "--no-context-files",
                "--no-extensions",
                "--no-skills",
                "--no-prompt-templates",
                "--no-themes",
                "--tools",
                "read",
                "--provider",
                "rocm-local",
                "--model",
                configured_model,
                "--api-key",
                "rocm-local",
                "--",
                prompt,
            ]));
            env_overrides.extend([
                ("PI_OFFLINE", "1"),
                ("PI_SKIP_VERSION_CHECK", "1"),
                ("HTTP_PROXY", ""),
                ("HTTPS_PROXY", ""),
                ("ALL_PROXY", ""),
                ("http_proxy", ""),
                ("https_proxy", ""),
                ("all_proxy", ""),
                ("NO_PROXY", "*"),
                ("no_proxy", "*"),
            ]);
        }
        AgentHarness::Omp => {
            let configured_model = configured_model
                .ok_or_else(|| anyhow!("omp harness test requires a configured model"))?;
            let model = format!("rocm-local/{configured_model}");
            owned_args.extend(strings(&[
                "-p",
                "--cwd",
                &workspace_text,
                "--no-session",
                "--no-title",
                "--no-tools",
                "--no-lsp",
                "--no-pty",
                "--no-extensions",
                "--no-skills",
                "--no-rules",
                "--model",
                &model,
                "--max-time",
                &timeout_secs,
                "@probe.txt",
                "Return exactly the contents of the attached probe.txt.",
            ]));
        }
    }
    let args = owned_args.iter().map(String::as_str).collect::<Vec<_>>();
    let output = capture_command(executable, &args, Some(&workspace), timeout, &env_overrides)
        .map_err(|error| {
            if has_io_error_kind(&error, ErrorKind::NotFound) {
                anyhow!(
                    "{} executable `{}` is not installed",
                    harness.canonical_name(),
                    harness.executable()
                )
            } else if error.to_string().contains("timed out") {
                anyhow!(
                    "{} harness test timed out after {} seconds",
                    harness.canonical_name(),
                    timeout.as_secs()
                )
            } else {
                error
            }
        })?;
    if !output.status.success() {
        let detail = concise_output(&output.stderr, &output.stdout);
        bail!(
            "{} harness exited with {}{}",
            harness.canonical_name(),
            output.status,
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        );
    }
    verify_probe_unchanged(&probe_path, &nonce)?;
    if !output.stdout.contains(&nonce) {
        bail!(
            "{} harness did not return the probe nonce in its final output",
            harness.canonical_name()
        );
    }
    Ok(())
}

fn ready_loopback_services(paths: &AppPaths) -> Result<Vec<ManagedServiceRecord>> {
    let services_dir = paths.services_dir();
    if !services_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut records = Vec::new();
    for entry in fs::read_dir(&services_dir)
        .with_context(|| format!("failed to read {}", services_dir.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let Ok(record) = serde_json::from_slice::<ManagedServiceRecord>(&bytes) else {
            continue;
        };
        let supervisor_live = record.supervisor_pid != 0
            && matches!(
                identity_state(&ProcessIdentity::new(
                    record.supervisor_pid,
                    record.supervisor_start_ticks,
                )),
                IdentityState::Matches | IdentityState::Indeterminate
            );
        let engine_live = record.engine_pid.is_some_and(|pid| {
            pid != 0
                && matches!(
                    identity_state(&ProcessIdentity::new(pid, record.engine_start_ticks)),
                    IdentityState::Matches | IdentityState::Indeterminate
                )
        });
        let live = supervisor_live || engine_live;
        if record.status == "ready" && live && normalize_loopback_url(&record.endpoint_url).is_ok()
        {
            records.push(record);
        }
    }
    Ok(records)
}

fn normalize_loopback_url(value: &str) -> Result<(String, String)> {
    let value = value.trim();
    if value.contains('?') || value.contains('#') {
        bail!("base URL must not contain a query string or fragment");
    }
    let remainder = value
        .strip_prefix("http://")
        .ok_or_else(|| anyhow!("base URL must use loopback HTTP"))?;
    let (authority, path) = remainder
        .split_once('/')
        .map_or((remainder, ""), |(authority, path)| (authority, path));
    if authority.is_empty() || authority.chars().any(char::is_whitespace) {
        bail!("base URL has an invalid authority");
    }
    if authority.contains('@') {
        bail!("base URL must not contain credentials");
    }
    if !matches!(path, "" | "v1" | "v1/") {
        bail!("base URL path must be empty or /v1");
    }
    validate_loopback_authority(authority)?;
    let origin = format!("http://{authority}");
    Ok((origin.clone(), format!("{origin}/v1")))
}

fn validate_loopback_authority(authority: &str) -> Result<()> {
    let host = if let Some(rest) = authority.strip_prefix('[') {
        let close = rest
            .find(']')
            .ok_or_else(|| anyhow!("base URL has an invalid IPv6 host"))?;
        let host = &rest[..close];
        let suffix = &rest[close + 1..];
        if !suffix.is_empty() {
            let port = suffix
                .strip_prefix(':')
                .ok_or_else(|| anyhow!("base URL has an invalid authority"))?;
            validate_port(port)?;
        }
        host
    } else {
        let colon_count = authority.matches(':').count();
        if colon_count > 1 {
            bail!("IPv6 loopback hosts must use brackets");
        }
        if let Some((host, port)) = authority.rsplit_once(':') {
            validate_port(port)?;
            host
        } else {
            authority
        }
    };
    if host.eq_ignore_ascii_case("localhost") {
        return Ok(());
    }
    let address = host
        .parse::<IpAddr>()
        .map_err(|_| anyhow!("base URL host must be localhost or a loopback IP address"))?;
    if !address.is_loopback() {
        bail!("base URL host must be loopback");
    }
    Ok(())
}

fn validate_port(value: &str) -> Result<()> {
    let port = value
        .parse::<u16>()
        .map_err(|_| anyhow!("base URL has an invalid port"))?;
    if port == 0 {
        bail!("base URL port must be greater than zero");
    }
    Ok(())
}

fn resolve_advertised_model(origin: &str) -> Result<String> {
    let body = rocm_core::http_get_text(origin, "/v1/models", MODEL_LIST_TIMEOUT)?;
    let value = serde_json::from_str::<serde_json::Value>(body.trim())
        .context("failed to parse /v1/models JSON")?;
    let models = value
        .get("data")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| item.get("id").and_then(serde_json::Value::as_str))
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    match models.len() {
        0 => bail!("/v1/models did not advertise a model id"),
        1 => Ok(models.into_iter().next().expect("length checked")),
        _ => bail!(
            "/v1/models advertised more than one model; use --model with one exact id: {}",
            models.into_iter().collect::<Vec<_>>().join(", ")
        ),
    }
}

fn first_semantic_version(output: &str) -> Option<Version> {
    output.split_whitespace().find_map(|token| {
        let token = token.trim_matches(|character: char| {
            !character.is_ascii_alphanumeric() && !matches!(character, '.' | '-' | '+')
        });
        let token = token
            .split_once('/')
            .filter(|(prefix, _)| {
                !prefix.is_empty()
                    && prefix
                        .chars()
                        .all(|character| character.is_ascii_alphanumeric() || character == '-')
            })
            .map_or(token, |(_, version)| version);
        let token = token
            .strip_prefix('v')
            .or_else(|| token.strip_prefix('V'))
            .unwrap_or(token);
        Version::parse(token).ok()
    })
}

fn find_executable(name: &str) -> Option<PathBuf> {
    let direct = Path::new(name);
    if direct.components().count() > 1 {
        return executable_file(direct).then(|| absolute_path(direct));
    }
    let path = env::var_os("PATH")?;
    for directory in env::split_paths(&path) {
        let candidate = directory.join(name);
        if executable_file(&candidate) {
            return Some(absolute_path(&candidate));
        }
        #[cfg(windows)]
        for extension in env::var_os("PATHEXT")
            .unwrap_or_else(|| ".COM;.EXE;.BAT;.CMD".into())
            .to_string_lossy()
            .split(';')
            .filter(|extension| !extension.is_empty())
        {
            let candidate = directory.join(format!("{name}{extension}"));
            if executable_file(&candidate) {
                return Some(absolute_path(&candidate));
            }
        }
    }
    None
}

fn absolute_path(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn executable_file(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

struct CapturedOutput {
    status: ExitStatus,
    stdout: String,
    stderr: String,
}

fn capture_command(
    program: &Path,
    args: &[&str],
    cwd: Option<&Path>,
    timeout: Duration,
    env_overrides: &[(&str, &str)],
) -> Result<CapturedOutput> {
    let capture = TempRoot::create("agent-output")?;
    let stdout_path = capture.path.join("stdout");
    let stderr_path = capture.path.join("stderr");
    let stdout = fs::File::create(&stdout_path)
        .with_context(|| format!("failed to create {}", stdout_path.display()))?;
    let stderr = fs::File::create(&stderr_path)
        .with_context(|| format!("failed to create {}", stderr_path.display()))?;
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    for &(key, value) in env_overrides {
        command.env(key, value);
    }
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to launch {}", program.display()))?;
    let identity = ProcessIdentity::capture(child.id());
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .with_context(|| format!("failed to poll {}", program.display()))?
        {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = terminate_verified(&identity, KillScope::Tree, PROCESS_STOP_GRACE, true);
            if child.try_wait().ok().flatten().is_none() && child.kill().is_ok() {
                let _ = child.wait();
            }
            bail!("command timed out after {} seconds", timeout.as_secs());
        }
        thread::sleep(COMMAND_POLL_INTERVAL);
    };
    Ok(CapturedOutput {
        status,
        stdout: read_bounded(&stdout_path)?,
        stderr: read_bounded(&stderr_path)?,
    })
}

fn read_bounded(path: &Path) -> Result<String> {
    let mut file =
        fs::File::open(path).with_context(|| format!("failed to read {}", path.display()))?;
    let length = file.metadata()?.len();
    let mut bytes = Vec::with_capacity(MAX_CAPTURE_BYTES as usize);
    if length <= MAX_CAPTURE_BYTES {
        file.read_to_end(&mut bytes)
            .with_context(|| format!("failed to read {}", path.display()))?;
    } else {
        let marker = b"\n[output truncated]\n";
        let payload = MAX_CAPTURE_BYTES - marker.len() as u64;
        let head = payload / 2;
        let tail = payload - head;
        (&mut file)
            .take(head)
            .read_to_end(&mut bytes)
            .with_context(|| format!("failed to read {}", path.display()))?;
        bytes.extend_from_slice(marker);
        file.seek(SeekFrom::Start(length - tail))?;
        file.take(tail)
            .read_to_end(&mut bytes)
            .with_context(|| format!("failed to read {}", path.display()))?;
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn has_io_error_kind(error: &anyhow::Error, kind: ErrorKind) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| error.kind() == kind)
    })
}

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

fn harness_test_timeout() -> Duration {
    env::var(TEST_TIMEOUT_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| (1..=600).contains(seconds))
        .map_or(HARNESS_TEST_TIMEOUT, Duration::from_secs)
}

fn verify_probe_unchanged(probe_path: &Path, nonce: &str) -> Result<()> {
    let contents = fs::read_to_string(probe_path)
        .with_context(|| format!("failed to read temporary probe {}", probe_path.display()))?;
    if contents != format!("{nonce}\n") {
        bail!("harness modified the temporary probe");
    }
    Ok(())
}

fn make_probe_read_only(probe_path: &Path) -> Result<()> {
    let mut probe_permissions = fs::metadata(probe_path)?.permissions();
    probe_permissions.set_readonly(true);
    fs::set_permissions(probe_path, probe_permissions)?;
    Ok(())
}

fn concise_output(primary: &str, fallback: &str) -> String {
    let value = if primary.trim().is_empty() {
        fallback.trim()
    } else {
        primary.trim()
    };
    let mut concise = value.chars().take(500).collect::<String>();
    if value.chars().count() > 500 {
        concise.push('…');
    }
    concise
}

struct TempRoot {
    path: PathBuf,
}

impl TempRoot {
    fn create(label: &str) -> Result<Self> {
        let parent = env::temp_dir();
        let builder = fs::DirBuilder::new();
        #[cfg(unix)]
        let builder = {
            use std::os::unix::fs::DirBuilderExt;
            let mut builder = builder;
            builder.mode(0o700);
            builder
        };
        let prefix = format!("rocm-{label}");
        for _ in 0..32 {
            let path = parent.join(unique_token(&prefix));
            match builder.create(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("failed to create {}", path.display()));
                }
            }
        }
        bail!("failed to create a unique temporary directory")
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        make_tree_writable(&self.path);
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn make_tree_writable(path: &Path) {
    let Ok(metadata) = fs::symlink_metadata(path) else {
        return;
    };
    if metadata.file_type().is_symlink() {
        return;
    }
    if metadata.is_dir() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o700));
        }
        #[cfg(not(unix))]
        {
            let mut permissions = metadata.permissions();
            permissions.set_readonly(false);
            let _ = fs::set_permissions(path, permissions);
        }
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                make_tree_writable(&entry.path());
            }
        }
    } else {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
        }
        #[cfg(not(unix))]
        {
            let mut permissions = metadata.permissions();
            permissions.set_readonly(false);
            let _ = fs::set_permissions(path, permissions);
        }
    }
}
