// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! System-installed ROCm SDK detection and probing.
//!
//! Locates a ROCm SDK installed outside the managed runtime registry (for
//! example via `amdgpu-install` or a distro package into `/opt/rocm`) and
//! validates that it is usable: the HIP runtime library is present, at least
//! one ROCm tool exists under `bin/`, and a version can be resolved. GPU
//! absence is deliberately *not* a probe failure — gfx enumeration is
//! best-effort and an empty target list is valid (health is a separate
//! concern from SDK validity).

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use crate::examine::{AMDGPU_INSTALL_MARKERS, extract_rocm_version};
use crate::{OPTIONAL_COMMAND_TIMEOUT, capture_optional_path_command_with_env};

/// Environment variables that name a system ROCm root, in precedence order.
const SYSTEM_ROCM_ENV_VARS: &[&str] = &["ROCM_PATH", "ROCM_HOME", "HIP_PATH"];

/// Default system ROCm install root probed when no env var points elsewhere.
const DEFAULT_SYSTEM_ROCM_ROOT: &str = "/opt/rocm";

/// Filename prefix of the HIP runtime shared library a usable SDK must ship.
const HIP_RUNTIME_LIB_PREFIX: &str = "libamdhip64.so";

/// Tools under `bin/` at least one of which must exist for a valid SDK.
const REQUIRED_SDK_TOOLS: &[&str] = &["rocminfo", "rocm_agent_enumerator", "hipconfig"];

/// Validated snapshot of a system-installed ROCm SDK.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemSdkProbe {
    /// Canonicalized SDK root (e.g. `/opt/rocm-6.4.1`).
    #[serde(default)]
    pub root: PathBuf,
    /// Resolved ROCm version string.
    #[serde(default)]
    pub version: String,
    /// `Some("amdgpu-install")` when amdgpu-install repo markers are present.
    #[serde(default)]
    pub install_method: Option<String>,
    /// Existing tool directories: `root/bin`, `root/llvm/bin`.
    #[serde(default)]
    pub bin_paths: Vec<PathBuf>,
    /// Existing library directories: `root/lib`, `root/lib64`.
    #[serde(default)]
    pub library_paths: Vec<PathBuf>,
    /// gfx targets from `rocm_agent_enumerator`; may be empty (never fatal).
    #[serde(default)]
    pub gfx_targets: Vec<String>,
}

/// Locate a system ROCm root: `ROCM_PATH`, `ROCM_HOME`, `HIP_PATH` (first
/// non-empty value naming an existing directory wins), then `/opt/rocm`.
pub fn detect_system_rocm_root() -> Option<PathBuf> {
    detect_system_rocm_root_from(
        |name: &str| std::env::var_os(name),
        Path::new(DEFAULT_SYSTEM_ROCM_ROOT),
    )
}

/// Pure core of [`detect_system_rocm_root`]: env lookups are injected so the
/// precedence is testable without mutating process-global env vars.
fn detect_system_rocm_root_from(
    env: impl Fn(&str) -> Option<OsString>,
    default_root: &Path,
) -> Option<PathBuf> {
    for var in SYSTEM_ROCM_ENV_VARS {
        let Some(value) = env(var) else {
            continue;
        };
        if value.is_empty() {
            continue;
        }
        let candidate = PathBuf::from(value);
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    default_root.is_dir().then(|| default_root.to_path_buf())
}

/// Validate the ROCm SDK rooted at `root`.
///
/// Hard requirements — any failure returns an `Err` naming the missing
/// artifact, never a partial probe:
/// - `root` canonicalizes to an existing directory;
/// - `libamdhip64.so*` exists under `lib/` or `lib64/`;
/// - at least one of `bin/rocminfo`, `bin/rocm_agent_enumerator`,
///   `bin/hipconfig` exists;
/// - a version resolves (`.info/version*` files, a `rocm-X.Y[.Z]` path
///   component, or `hipconfig --version` with a bounded timeout).
///
/// gfx enumeration failure or empty output is non-fatal.
pub fn probe_system_rocm_sdk(root: &Path) -> Result<SystemSdkProbe> {
    let root = root.canonicalize().with_context(|| {
        format!(
            "system ROCm root {} does not exist or cannot be canonicalized",
            root.display()
        )
    })?;
    if !root.is_dir() {
        bail!("system ROCm root {} is not a directory", root.display());
    }

    let library_paths: Vec<PathBuf> = ["lib", "lib64"]
        .iter()
        .map(|dir| root.join(dir))
        .filter(|path| path.is_dir())
        .collect();
    if !library_paths
        .iter()
        .any(|dir| dir_contains_hip_runtime(dir))
    {
        bail!(
            "HIP runtime library ({HIP_RUNTIME_LIB_PREFIX}*) not found under {}/lib or {}/lib64",
            root.display(),
            root.display()
        );
    }

    let bin_dir = root.join("bin");
    if !REQUIRED_SDK_TOOLS
        .iter()
        .any(|tool| bin_dir.join(tool).is_file())
    {
        bail!(
            "no ROCm tool found under {}: expected at least one of {}",
            bin_dir.display(),
            REQUIRED_SDK_TOOLS.join(", ")
        );
    }

    let Some(version) = resolve_system_sdk_version(&root) else {
        bail!(
            "could not resolve ROCm version for {}: no .info/version file, no rocm-X.Y path component, and hipconfig --version yielded nothing",
            root.display()
        );
    };

    let bin_paths: Vec<PathBuf> = [root.join("bin"), root.join("llvm").join("bin")]
        .into_iter()
        .filter(|path| path.is_dir())
        .collect();

    let install_method = AMDGPU_INSTALL_MARKERS
        .iter()
        .any(|marker| Path::new(marker).exists())
        .then(|| "amdgpu-install".to_owned());

    let gfx_targets = enumerate_system_gfx_targets(&root, &library_paths);

    Ok(SystemSdkProbe {
        root,
        version,
        install_method,
        bin_paths,
        library_paths,
        gfx_targets,
    })
}

/// Whether `dir` contains a `libamdhip64.so*` file.
fn dir_contains_hip_runtime(dir: &Path) -> bool {
    fs::read_dir(dir).is_ok_and(|entries| {
        entries.flatten().any(|entry| {
            entry.path().is_file()
                && entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with(HIP_RUNTIME_LIB_PREFIX))
        })
    })
}

/// Cheap re-check of a stored [`SystemSdkProbe`] without running subprocesses.
///
/// Confirms the recorded root is still a directory and a `libamdhip64.so*`
/// file is still present under the stored `library_paths` (falling back to
/// `root/lib` and `root/lib64` when the stored list is empty or stale). This
/// is deliberately weaker than [`probe_system_rocm_sdk`]: it answers "is the
/// snapshot still plausible", not "is the SDK still fully valid".
pub fn validate_system_sdk_probe(probe: &SystemSdkProbe) -> Result<()> {
    if !probe.root.is_dir() {
        bail!(
            "system ROCm root is missing or not a directory: {}",
            probe.root.display()
        );
    }
    let fallback_library_dirs = [probe.root.join("lib"), probe.root.join("lib64")];
    let has_hip_runtime = probe
        .library_paths
        .iter()
        .chain(fallback_library_dirs.iter())
        .any(|dir| dir_contains_hip_runtime(dir));
    if !has_hip_runtime {
        bail!(
            "HIP runtime library ({HIP_RUNTIME_LIB_PREFIX}*) is no longer present under {}",
            probe.root.display()
        );
    }
    Ok(())
}

/// Resolve the SDK version: `.info/version`, `.info/version-utils`,
/// `.info/version-libs` (trimmed), then a `rocm-X.Y[.Z]` component of the
/// canonicalized path, then `hipconfig --version` with a bounded timeout.
fn resolve_system_sdk_version(root: &Path) -> Option<String> {
    for name in ["version", "version-utils", "version-libs"] {
        if let Ok(text) = fs::read_to_string(root.join(".info").join(name)) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_owned());
            }
        }
    }
    if let Some(version) = root.to_str().and_then(extract_rocm_version) {
        return Some(version);
    }
    let hipconfig = root.join("bin").join("hipconfig");
    if hipconfig.is_file()
        && let Some(output) = capture_optional_path_command_with_env(
            &hipconfig,
            &["--version"],
            &[],
            OPTIONAL_COMMAND_TIMEOUT,
        )
        && let Some(token) = output.split_whitespace().next()
    {
        return Some(token.to_owned());
    }
    None
}

/// Best-effort gfx targets via `bin/rocm_agent_enumerator`.
///
/// Runs with `LD_LIBRARY_PATH` composed from the probe's library paths so the
/// enumerator finds the SDK's own runtime. All whitespace-separated `gfx*`
/// tokens are collected, deduplicated preserving order, and the CPU
/// placeholder `gfx000` dropped. Any failure or empty output yields an empty
/// vector — GPU absence is not an SDK-validity concern.
fn enumerate_system_gfx_targets(root: &Path, library_paths: &[PathBuf]) -> Vec<String> {
    let tool = root.join("bin").join("rocm_agent_enumerator");
    if !tool.is_file() {
        return Vec::new();
    }
    let mut envs: Vec<(&str, OsString)> = Vec::new();
    if !library_paths.is_empty()
        && let Ok(joined) = std::env::join_paths(library_paths.iter().cloned())
    {
        envs.push(("LD_LIBRARY_PATH", joined));
    }
    let Some(output) =
        capture_optional_path_command_with_env(&tool, &[], &envs, OPTIONAL_COMMAND_TIMEOUT)
    else {
        return Vec::new();
    };
    let mut targets: Vec<String> = Vec::new();
    for token in output.split_whitespace() {
        if token.starts_with("gfx") && token != "gfx000" && !targets.iter().any(|t| t == token) {
            targets.push(token.to_owned());
        }
    }
    targets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unix_time_millis;

    /// Unique fixture parent under the OS temp dir. The name deliberately
    /// avoids the substring `rocm-` so path-based version extraction is only
    /// exercised when a test creates a `rocm-X.Y.Z` component on purpose.
    fn temp_fixture_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sys-sdk-{name}-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Lay down a minimal valid SDK tree under `root`: `.info/version`,
    /// `lib/libamdhip64.so`, and `bin/rocminfo`.
    fn write_minimal_sdk(root: &Path, version: &str) {
        fs::create_dir_all(root.join(".info")).unwrap();
        fs::write(root.join(".info").join("version"), format!("{version}\n")).unwrap();
        fs::create_dir_all(root.join("lib")).unwrap();
        fs::write(root.join("lib").join("libamdhip64.so"), b"").unwrap();
        fs::create_dir_all(root.join("bin")).unwrap();
        fs::write(root.join("bin").join("rocminfo"), b"").unwrap();
    }

    #[cfg(unix)]
    fn write_executable_script(path: &Path, body: &str) {
        use std::os::unix::fs::PermissionsExt;
        fs::write(path, body).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[test]
    fn probe_happy_path_resolves_version_and_paths() {
        let parent = temp_fixture_dir("happy");
        let root = parent.join("sdk");
        write_minimal_sdk(&root, "6.4.1");

        let probe = probe_system_rocm_sdk(&root).unwrap();
        let canonical = root.canonicalize().unwrap();
        assert_eq!(probe.root, canonical);
        assert_eq!(probe.version, "6.4.1");
        assert_eq!(probe.bin_paths, vec![canonical.join("bin")]);
        assert_eq!(probe.library_paths, vec![canonical.join("lib")]);
        assert!(probe.gfx_targets.is_empty());

        fs::remove_dir_all(&parent).unwrap();
    }

    #[test]
    fn probe_fails_without_hip_runtime_library() {
        let parent = temp_fixture_dir("no-hip-lib");
        let root = parent.join("sdk");
        write_minimal_sdk(&root, "6.4.1");
        fs::remove_file(root.join("lib").join("libamdhip64.so")).unwrap();
        fs::create_dir(root.join("lib").join("libamdhip64.so")).unwrap();

        let error = probe_system_rocm_sdk(&root).unwrap_err();
        assert!(
            error.to_string().contains("libamdhip64"),
            "error must name the missing HIP library, got: {error}"
        );

        fs::remove_dir_all(&parent).unwrap();
    }

    #[test]
    fn probe_fails_without_any_sdk_tool() {
        let parent = temp_fixture_dir("no-tools");
        let root = parent.join("sdk");
        write_minimal_sdk(&root, "6.4.1");
        fs::remove_file(root.join("bin").join("rocminfo")).unwrap();

        let error = probe_system_rocm_sdk(&root).unwrap_err();
        assert!(
            error.to_string().contains("rocminfo"),
            "error must name the expected tools, got: {error}"
        );

        fs::remove_dir_all(&parent).unwrap();
    }

    #[test]
    fn probe_fails_without_resolvable_version() {
        let parent = temp_fixture_dir("no-version");
        let root = parent.join("sdk");
        write_minimal_sdk(&root, "6.4.1");
        fs::remove_dir_all(root.join(".info")).unwrap();

        let error = probe_system_rocm_sdk(&root).unwrap_err();
        assert!(
            error.to_string().contains("version"),
            "error must name the missing version, got: {error}"
        );

        fs::remove_dir_all(&parent).unwrap();
    }

    #[test]
    fn probe_resolves_version_from_path_component() {
        let parent = temp_fixture_dir("version-from-path");
        let root = parent.join("rocm-6.3.2");
        write_minimal_sdk(&root, "unused");
        fs::remove_dir_all(root.join(".info")).unwrap();

        let probe = probe_system_rocm_sdk(&root).unwrap();
        assert_eq!(probe.version, "6.3.2");

        fs::remove_dir_all(&parent).unwrap();
    }

    #[test]
    fn detection_order_env_beats_default_and_skips_invalid() {
        let parent = temp_fixture_dir("detect-order");
        let dir_a = parent.join("a");
        let dir_b = parent.join("b");
        let dir_c = parent.join("c");
        let default_root = parent.join("default");
        for dir in [&dir_a, &dir_b, &dir_c, &default_root] {
            fs::create_dir_all(dir).unwrap();
        }
        let missing = parent.join("missing");
        let os = |path: &Path| Some(OsString::from(path.as_os_str()));

        // ROCM_PATH beats ROCM_HOME beats HIP_PATH.
        let all_set = |var: &str| match var {
            "ROCM_PATH" => os(&dir_a),
            "ROCM_HOME" => os(&dir_b),
            "HIP_PATH" => os(&dir_c),
            _ => None,
        };
        assert_eq!(
            detect_system_rocm_root_from(all_set, &default_root).as_deref(),
            Some(dir_a.as_path())
        );

        let home_and_hip = |var: &str| match var {
            "ROCM_HOME" => os(&dir_b),
            "HIP_PATH" => os(&dir_c),
            _ => None,
        };
        assert_eq!(
            detect_system_rocm_root_from(home_and_hip, &default_root).as_deref(),
            Some(dir_b.as_path())
        );

        let hip_only = |var: &str| (var == "HIP_PATH").then(|| os(&dir_c)).flatten();
        assert_eq!(
            detect_system_rocm_root_from(hip_only, &default_root).as_deref(),
            Some(dir_c.as_path())
        );

        // Empty and missing-directory candidates are skipped in favor of the
        // next candidate in precedence order.
        let empty_then_home = |var: &str| match var {
            "ROCM_PATH" => Some(OsString::new()),
            "ROCM_HOME" => os(&dir_b),
            _ => None,
        };
        assert_eq!(
            detect_system_rocm_root_from(empty_then_home, &default_root).as_deref(),
            Some(dir_b.as_path())
        );

        let missing_dir_then_default =
            |var: &str| (var == "ROCM_PATH").then(|| os(&missing)).flatten();
        assert_eq!(
            detect_system_rocm_root_from(missing_dir_then_default, &default_root).as_deref(),
            Some(default_root.as_path())
        );

        // No env, default exists → default. Nothing at all → None.
        assert_eq!(
            detect_system_rocm_root_from(|_: &str| None, &default_root).as_deref(),
            Some(default_root.as_path())
        );
        assert_eq!(detect_system_rocm_root_from(|_: &str| None, &missing), None);

        fs::remove_dir_all(&parent).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn probe_collects_deduplicated_gfx_targets_dropping_gfx000() {
        let parent = temp_fixture_dir("gfx-targets");
        let root = parent.join("sdk");
        write_minimal_sdk(&root, "6.4.1");
        write_executable_script(
            &root.join("bin").join("rocm_agent_enumerator"),
            "#!/bin/sh\nprintf 'gfx1201\\ngfx1201\\ngfx000\\n'\n",
        );

        let probe = probe_system_rocm_sdk(&root).unwrap();
        assert_eq!(probe.gfx_targets, vec!["gfx1201".to_owned()]);

        fs::remove_dir_all(&parent).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn probe_tolerates_failing_gfx_enumerator() {
        let parent = temp_fixture_dir("gfx-fail");
        let root = parent.join("sdk");
        write_minimal_sdk(&root, "6.4.1");
        write_executable_script(
            &root.join("bin").join("rocm_agent_enumerator"),
            "#!/bin/sh\nexit 3\n",
        );

        let probe = probe_system_rocm_sdk(&root).unwrap();
        assert!(probe.gfx_targets.is_empty());
        assert_eq!(probe.version, "6.4.1");

        fs::remove_dir_all(&parent).unwrap();
    }

    #[test]
    fn validate_probe_accepts_intact_fixture() {
        let parent = temp_fixture_dir("validate-ok");
        let root = parent.join("sdk");
        write_minimal_sdk(&root, "6.4.1");

        let probe = probe_system_rocm_sdk(&root).unwrap();
        validate_system_sdk_probe(&probe).unwrap();

        fs::remove_dir_all(&parent).unwrap();
    }

    #[test]
    fn validate_probe_fails_when_hip_runtime_disappears() {
        let parent = temp_fixture_dir("validate-no-hip");
        let root = parent.join("sdk");
        write_minimal_sdk(&root, "6.4.1");

        let probe = probe_system_rocm_sdk(&root).unwrap();
        fs::remove_file(root.join("lib").join("libamdhip64.so")).unwrap();

        let error = validate_system_sdk_probe(&probe).unwrap_err();
        assert!(
            error.to_string().contains("libamdhip64"),
            "error must name the missing HIP library, got: {error}"
        );

        fs::remove_dir_all(&parent).unwrap();
    }
}
