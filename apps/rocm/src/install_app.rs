// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm install app` — the one path that installs ROCm App.
//!
//! # The asymmetry this module protects
//!
//! Installing ROCm App installs the CLI. Installing the CLI **never** installs
//! ROCm App. `install.sh`, `install.ps1`, `rocm install sdk`, `rocm update`,
//! and first-run setup must not reach this module, and
//! `install_app_is_not_reachable_from_any_other_install_path` asserts it by
//! scanning the repository rather than by convention.
//!
//! # Verification order
//!
//! Platform → manifest schema → target match → download → size → digest →
//! signature → execute. Every check that can be made without touching the
//! network happens first, so an unsupported host or a malformed manifest costs
//! nothing and leaks nothing. Nothing is executed until all of them pass.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, verify_rsa_pkcs1_sha256_signature};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Manifest schema version this build understands.
pub(crate) const APP_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// A signed description of one ROCm App release.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct AppReleaseManifest {
    pub schema_version: u32,
    pub app_version: String,
    /// Inclusive CLI version range this app build is compatible with.
    pub compatible_cli: CliRange,
    pub published_at_unix_ms: u64,
    pub release_notes_url: String,
    pub assets: Vec<AppAsset>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct CliRange {
    pub min: String,
    pub max: String,
}

/// One downloadable installer for a specific target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct AppAsset {
    /// `windows` or `linux`. A closed vocabulary; anything else is rejected.
    pub os: String,
    /// `x86_64`. Other architectures are out of scope for v1.
    pub arch: String,
    pub url: String,
    pub file_name: String,
    pub size_bytes: u64,
    pub sha256: String,
    /// Detached RSASSA-PKCS#1 v1.5 SHA-256 signature over the asset bytes,
    /// base64-encoded.
    pub signature_b64: String,
}

/// Where the trusted public key comes from.
///
/// Production trust roots are owner-controlled inputs supplied at runtime; this
/// repository contains no private key and no production key material. Tests use
/// ephemeral generated keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AppTrustPolicy {
    /// When true, an asset without a verifiable signature is refused.
    pub require_signature: bool,
    pub public_key_pem: Option<String>,
}

impl AppTrustPolicy {
    /// Read the policy from the environment.
    ///
    /// Defaults to **required**. A default of "optional" would mean a
    /// misconfigured host silently accepts unsigned installers, which is the
    /// failure mode worth being loud about.
    pub(crate) fn from_env() -> Self {
        let pem = std::env::var("ROCM_CLI_APP_PUBLIC_KEY_PEM")
            .ok()
            .or_else(|| {
                std::env::var("ROCM_CLI_APP_PUBLIC_KEY_PATH")
                    .ok()
                    .and_then(|path| std::fs::read_to_string(path).ok())
            });
        let require_signature = std::env::var("ROCM_CLI_APP_REQUIRE_SIGNATURE")
            .map_or(true, |value| {
                !matches!(value.as_str(), "0" | "false" | "no")
            });
        Self {
            require_signature,
            public_key_pem: pem,
        }
    }
}

/// The host we are installing onto.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TargetHost {
    pub os: String,
    pub arch: String,
    pub is_wsl: bool,
}

impl TargetHost {
    /// Detect the current host.
    pub(crate) fn detect() -> Self {
        let examination = rocm_core::Examination::probe(rocm_core::FrameworkProbe::Skip);
        Self {
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            is_wsl: examination.is_wsl,
        }
    }

    /// Refuse hosts this product does not support.
    ///
    /// Runs **before** any network access: an unsupported machine should not
    /// announce itself to a download server just to be told no.
    pub(crate) fn ensure_supported(&self) -> Result<()> {
        if self.is_wsl {
            bail!(
                "ROCm App is not supported under WSL.\n\
                 It manages ROCm on native Windows and native Linux. \
                 Install and run ROCm App on your Windows desktop instead."
            );
        }
        if !matches!(self.os.as_str(), "windows" | "linux") {
            bail!(
                "ROCm App supports native Windows and native Linux only; this host reports {}.",
                self.os
            );
        }
        if self.arch != "x86_64" {
            bail!(
                "ROCm App supports x86_64 only; this host reports {}.",
                self.arch
            );
        }
        Ok(())
    }
}

/// The exact change `rocm install app` would make.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AppInstallPlan {
    pub app_version: String,
    pub compatible_cli: CliRange,
    pub asset: AppAsset,
    pub release_notes_url: String,
    pub signature_required: bool,
    pub install_root: PathBuf,
}

impl AppInstallPlan {
    /// Human review text. Shown identically for `--dry-run` and before apply,
    /// so what a user reviews is what a user approves.
    pub(crate) fn render(&self) -> String {
        let signature = if self.signature_required {
            "required (verified before the installer runs)"
        } else {
            "optional (policy allows unsigned)"
        };
        format!(
            "rocm install app\n\
             \x20 app_version: {version}\n\
             \x20 compatible_cli: {cli_min} to {cli_max}\n\
             \x20 target: {os}-{arch}\n\
             \x20 asset: {file_name}\n\
             \x20 source: {url}\n\
             \x20 size_bytes: {size}\n\
             \x20 sha256: {sha}\n\
             \x20 signature: {signature}\n\
             \x20 release_notes: {notes}\n\
             \x20 install_root: {root}\n\
             \x20 changes:\n\
             \x20   - download the installer to a temporary folder\n\
             \x20   - verify its size, checksum, and signature\n\
             \x20   - run the verified installer\n\
             \x20   - install the bundled rocm command-line tool\n\
             \x20   - remove the temporary folder\n\
             \x20 note: installing ROCm App also installs the rocm CLI.\n\
             \x20 note: no driver is installed, updated, or modified.\n",
            version = self.app_version,
            cli_min = self.compatible_cli.min,
            cli_max = self.compatible_cli.max,
            os = self.asset.os,
            arch = self.asset.arch,
            file_name = self.asset.file_name,
            url = self.asset.url,
            size = self.asset.size_bytes,
            sha = self.asset.sha256,
            notes = self.release_notes_url,
            root = self.install_root.display(),
        )
    }
}

/// Parse and structurally validate a manifest.
pub(crate) fn parse_manifest(raw: &str) -> Result<AppReleaseManifest> {
    let manifest: AppReleaseManifest =
        serde_json::from_str(raw).context("app release manifest is malformed")?;

    if manifest.schema_version != APP_MANIFEST_SCHEMA_VERSION {
        bail!(
            "app release manifest schema {} is not supported by this CLI (expected {})",
            manifest.schema_version,
            APP_MANIFEST_SCHEMA_VERSION
        );
    }
    if manifest.app_version.trim().is_empty() {
        bail!("app release manifest has no app version");
    }
    if manifest.assets.is_empty() {
        bail!("app release manifest lists no assets");
    }
    for asset in &manifest.assets {
        if !matches!(asset.os.as_str(), "windows" | "linux") {
            bail!(
                "app release manifest contains an unsupported target os: {}",
                asset.os
            );
        }
        if asset.sha256.len() != 64 || !asset.sha256.chars().all(|c| c.is_ascii_hexdigit()) {
            bail!(
                "app release manifest asset {} has a malformed sha256",
                asset.file_name
            );
        }
        if asset.size_bytes == 0 {
            bail!(
                "app release manifest asset {} declares zero bytes",
                asset.file_name
            );
        }
        // A file name is later joined onto a temp directory, so it must be a
        // plain name. Rejecting here keeps the check off the download path.
        if asset.file_name.contains('/')
            || asset.file_name.contains('\\')
            || asset.file_name.contains("..")
            || asset.file_name.trim().is_empty()
        {
            bail!(
                "app release manifest asset name is unsafe: {}",
                asset.file_name
            );
        }
    }
    Ok(manifest)
}

/// Select the asset for a host, or explain why none matches.
pub(crate) fn select_asset(manifest: &AppReleaseManifest, host: &TargetHost) -> Result<AppAsset> {
    manifest
        .assets
        .iter()
        .find(|asset| asset.os == host.os && asset.arch == host.arch)
        .cloned()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no ROCm App asset published for {}-{} in release {}",
                host.os,
                host.arch,
                manifest.app_version
            )
        })
}

/// Build a reviewable plan. Performs no download and no mutation.
pub(crate) fn build_plan(
    manifest: &AppReleaseManifest,
    host: &TargetHost,
    policy: &AppTrustPolicy,
    install_root: PathBuf,
) -> Result<AppInstallPlan> {
    host.ensure_supported()?;
    let asset = select_asset(manifest, host)?;
    Ok(AppInstallPlan {
        app_version: manifest.app_version.clone(),
        compatible_cli: manifest.compatible_cli.clone(),
        asset,
        release_notes_url: manifest.release_notes_url.clone(),
        signature_required: policy.require_signature,
        install_root,
    })
}

/// Verify downloaded bytes against the manifest and the trust policy.
///
/// Order matters: size, then digest, then signature. Size is a constant-time
/// reject for a truncated or padded download, and there is no reason to hash
/// or verify bytes already known to be the wrong length.
pub(crate) fn verify_asset_bytes(
    asset: &AppAsset,
    bytes: &[u8],
    policy: &AppTrustPolicy,
) -> Result<()> {
    let actual_len = bytes.len() as u64;
    if actual_len != asset.size_bytes {
        bail!(
            "downloaded {} bytes but the manifest declares {}",
            actual_len,
            asset.size_bytes
        );
    }

    let digest = hex_lower(&Sha256::digest(bytes));
    if !digest.eq_ignore_ascii_case(&asset.sha256) {
        bail!(
            "checksum mismatch for {}: expected {}, got {digest}",
            asset.file_name,
            asset.sha256
        );
    }

    if !policy.require_signature {
        return Ok(());
    }
    let Some(public_key_pem) = policy.public_key_pem.as_deref() else {
        bail!(
            "a signature is required for {} but no trusted public key is configured; \
             set ROCM_CLI_APP_PUBLIC_KEY_PATH or ROCM_CLI_APP_PUBLIC_KEY_PEM",
            asset.file_name
        );
    };
    let signature = base64_decode(&asset.signature_b64)
        .with_context(|| format!("signature for {} is not valid base64", asset.file_name))?;

    verify_rsa_pkcs1_sha256_signature(public_key_pem, bytes, &signature, "app installer")
        .with_context(|| format!("signature verification failed for {}", asset.file_name))
}

fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
}

/// Minimal base64 decoder.
///
/// The workspace has no base64 dependency and this is the only consumer;
/// pulling a crate in for one 30-line function is not worth the supply-chain
/// surface on a security path.
fn base64_decode(input: &str) -> Result<Vec<u8>> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut lookup = [255u8; 256];
    for (index, byte) in TABLE.iter().enumerate() {
        lookup[*byte as usize] = u8::try_from(index).expect("index < 64");
    }

    let cleaned: Vec<u8> = input
        .bytes()
        .filter(|b| !b.is_ascii_whitespace() && *b != b'=')
        .collect();
    let mut out = Vec::with_capacity(cleaned.len() * 3 / 4);
    let mut accumulator: u32 = 0;
    let mut bits: u32 = 0;
    for byte in cleaned {
        let value = lookup[byte as usize];
        if value == 255 {
            bail!("invalid base64 character: {:?}", byte as char);
        }
        accumulator = (accumulator << 6) | u32::from(value);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(u8::try_from((accumulator >> bits) & 0xFF).expect("masked to a byte"));
        }
    }
    Ok(out)
}

/// A uniquely named temporary directory that removes itself on drop.
///
/// Drop-based cleanup rather than a call at each exit: the verification path
/// has many early returns, and "clean up on every failure branch" is a rule
/// that gets broken the first time someone adds a branch.
pub(crate) struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    pub(crate) fn create(parent: &Path) -> Result<Self> {
        // pid + millisecond is not unique: two scratch dirs created in the same
        // millisecond collide, and the second install then writes into the
        // first one's directory. The counter makes it unique within a process,
        // the pid across processes.
        static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = format!(
            "rocm-install-app-{}-{}-{}",
            std::process::id(),
            rocm_core::unix_time_millis(),
            SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let path = parent.join(unique);
        std::fs::create_dir_all(&path)
            .with_context(|| format!("could not create {}", path.display()))?;
        Ok(Self { path })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// How the verified installer is launched.
///
/// A seam so the platform-specific launch is testable without executing an
/// installer. `argv_for_installer` is pure and is what the tests assert.
pub(crate) trait InstallerLauncher {
    fn launch(&self, installer: &Path) -> Result<()>;
}

/// The exact argv used to run a verified installer.
///
/// Never a shell string. On Windows the NSIS installer is executed directly
/// with a silent flag; on Linux the downloaded package is handed to the
/// system package tool by absolute path. Neither composes a command line from
/// text, so a hostile file name cannot become an argument.
pub(crate) fn argv_for_installer(os: &str, installer: &Path) -> Result<Vec<String>> {
    let path = installer.to_string_lossy().into_owned();
    match os {
        "windows" => Ok(vec![path, "/S".to_owned()]),
        "linux" => Ok(vec![path]),
        other => bail!("no installer launch defined for {other}"),
    }
}

/// Runs the verified installer as a direct process.
pub(crate) struct ProcessLauncher {
    pub os: String,
}

impl InstallerLauncher for ProcessLauncher {
    fn launch(&self, installer: &Path) -> Result<()> {
        let argv = argv_for_installer(&self.os, installer)?;
        let (program, args) = argv.split_first().expect("argv is never empty");

        // Direct process execution from a resolved path. No shell, so nothing
        // in the file name can be interpreted.
        let status = std::process::Command::new(program)
            .args(args)
            .status()
            .with_context(|| format!("could not run the ROCm App installer at {program}"))?;
        if !status.success() {
            bail!("the ROCm App installer exited with {status}");
        }
        Ok(())
    }
}

/// Everything `apply` needs, so the flow is testable without a network.
pub(crate) struct ApplyInputs<'a> {
    pub plan: &'a AppInstallPlan,
    pub policy: &'a AppTrustPolicy,
    /// Fetch the asset bytes. Injected so tests supply fixtures.
    pub fetch: &'a dyn Fn(&str) -> Result<Vec<u8>>,
    pub launcher: &'a dyn InstallerLauncher,
    pub scratch_parent: &'a Path,
}

/// Download, verify, and run the installer.
///
/// Returns the path that was executed, which is what the tests assert rather
/// than trusting a boolean.
pub(crate) fn apply(inputs: &ApplyInputs<'_>) -> Result<PathBuf> {
    let scratch = ScratchDir::create(inputs.scratch_parent)?;
    let target = scratch.path().join(&inputs.plan.asset.file_name);

    let bytes = (inputs.fetch)(&inputs.plan.asset.url)
        .with_context(|| format!("could not download {}", inputs.plan.asset.url))?;

    // Verified before anything is written where it could be executed.
    verify_asset_bytes(&inputs.plan.asset, &bytes, inputs.policy)?;

    std::fs::write(&target, &bytes)
        .with_context(|| format!("could not write {}", target.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o700))?;
    }

    inputs.launcher.launch(&target)?;
    Ok(target)
}

/// Default install root for the app.
pub(crate) fn default_install_root(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("app")
}

#[cfg(test)]
mod tests;
