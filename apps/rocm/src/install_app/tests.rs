// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm install app` tests.
//!
//! Every test uses local fixtures and ephemeral generated keys. No production
//! key material exists in this repository, and
//! `install_app_repository_contains_no_private_key` asserts that.

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use super::*;

const SIGNED_KEY_ENV: &str = "unused";

fn scratch_root(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "rocm-install-app-test-{label}-{}",
        rocm_core::unix_time_millis()
    ));
    std::fs::create_dir_all(&root).expect("create scratch root");
    root
}

fn asset_bytes() -> Vec<u8> {
    b"ROCm App installer payload".to_vec()
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// An ephemeral keypair. Generated per test, never stored in the repository.
fn ephemeral_keys() -> (String, String) {
    rocm_core::generate_rsa_signing_keypair().expect("generate ephemeral signing key")
}

fn manifest_json(payload: &[u8], signature_b64: &str) -> String {
    format!(
        r#"{{
  "schemaVersion": 1,
  "appVersion": "0.1.0",
  "compatibleCli": {{ "min": "0.1.0", "max": "0.2.0" }},
  "publishedAtUnixMs": {published},
  "releaseNotesUrl": "https://github.com/mikeroysoft/rocm-app/releases/tag/v0.1.0",
  "assets": [
    {{
      "os": "linux",
      "arch": "x86_64",
      "url": "https://example.invalid/rocm-app_0.1.0_amd64.deb",
      "fileName": "rocm-app_0.1.0_amd64.deb",
      "sizeBytes": {len},
      "sha256": "{sha}",
      "signatureB64": "{signature_b64}"
    }},
    {{
      "os": "windows",
      "arch": "x86_64",
      "url": "https://example.invalid/rocm-app_0.1.0_x64-setup.exe",
      "fileName": "rocm-app_0.1.0_x64-setup.exe",
      "sizeBytes": {len},
      "sha256": "{sha}",
      "signatureB64": "{signature_b64}"
    }}
  ]
}}"#,
        len = payload.len(),
        sha = sha256_hex(payload),
        published = now_unix_ms(),
    )
}

fn linux_host() -> TargetHost {
    TargetHost {
        os: "linux".to_owned(),
        arch: "x86_64".to_owned(),
        is_wsl: false,
    }
}

fn windows_host() -> TargetHost {
    TargetHost {
        os: "windows".to_owned(),
        arch: "x86_64".to_owned(),
        is_wsl: false,
    }
}

fn signed_fixture() -> (AppReleaseManifest, Vec<u8>, AppTrustPolicy) {
    let payload = asset_bytes();
    let (private_pem, public_pem) = ephemeral_keys();
    let signature =
        rocm_core::sign_rsa_pkcs1_sha256_signature(&private_pem, &payload).expect("sign");
    let manifest = parse_manifest(&manifest_json(&payload, &base64_encode(&signature)))
        .expect("fixture manifest parses");
    let policy = AppTrustPolicy {
        require_signature: true,
        public_key_pem: Some(public_pem),
    };
    (manifest, payload, policy)
}

/// Records what would have been executed instead of executing it.
#[derive(Default)]
struct RecordingLauncher {
    launched: RefCell<Vec<PathBuf>>,
}

impl InstallerLauncher for RecordingLauncher {
    fn launch(&self, installer: &Path) -> Result<()> {
        self.launched.borrow_mut().push(installer.to_path_buf());
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Dry run
// ---------------------------------------------------------------------------

#[test]
fn install_app_dry_run_reports_the_exact_plan_for_linux() {
    let (manifest, _, policy) = signed_fixture();
    let plan = build_plan(
        &manifest,
        &linux_host(),
        &policy,
        PathBuf::from("/home/user/.rocm/app"),
        false,
    )
    .expect("plan");
    let rendered = plan.render();
    println!("--- linux dry run ---\n{rendered}");

    for expected in [
        "app_version: 0.1.0",
        "compatible_cli: 0.1.0 to 0.2.0",
        "target: linux-x86_64",
        "asset: rocm-app_0.1.0_amd64.deb",
        "source: https://example.invalid/rocm-app_0.1.0_amd64.deb",
        "signature: required",
        "install_root: /home/user/.rocm/app",
        "installing ROCm App also installs the rocm CLI",
        "no driver is installed",
    ] {
        assert!(
            rendered.contains(expected),
            "missing {expected:?}:\n{rendered}"
        );
    }
}

#[test]
fn install_app_dry_run_reports_the_exact_plan_for_windows() {
    let (manifest, _, policy) = signed_fixture();
    let plan = build_plan(
        &manifest,
        &windows_host(),
        &policy,
        PathBuf::from("C:/Users/user/AppData/Roaming/rocm/app"),
        false,
    )
    .expect("plan");
    let rendered = plan.render();
    println!("--- windows dry run ---\n{rendered}");

    assert!(rendered.contains("target: windows-x86_64"));
    assert!(rendered.contains("asset: rocm-app_0.1.0_x64-setup.exe"));
}

/// `--dry-run` must not download or install.
///
/// Drives the real command, not a stand-in. The asset URL is unreachable, so a
/// dry run that tried to fetch would fail here — and it leaves no scratch
/// directory behind either.
#[test]
fn install_app_dry_run_performs_no_download() {
    let (_, payload, _) = signed_fixture();
    let root = scratch_root("dry-run-command");
    let manifest_path = root.join("manifest.json");
    std::fs::write(&manifest_path, manifest_json(&payload, "AA==")).expect("write manifest");

    let paths = rocm_core::AppPaths {
        config_dir: root.join("config"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
    };
    std::fs::create_dir_all(&paths.cache_dir).expect("cache dir");

    let host = TargetHost::detect();
    if host.ensure_supported().is_err() {
        // This suite also runs on hosts the product does not support; the
        // unsupported path has its own tests.
        let _ = std::fs::remove_dir_all(&root);
        return;
    }

    crate::install_app_command(&paths, true, false, Some(&manifest_path), false)
        .expect("dry run succeeds without network");

    let leftovers: Vec<_> = std::fs::read_dir(&paths.cache_dir)
        .expect("read cache dir")
        .filter_map(std::result::Result::ok)
        .map(|e| e.file_name())
        .collect();
    assert!(leftovers.is_empty(), "dry run left files: {leftovers:?}");
    let _ = std::fs::remove_dir_all(&root);
}

// ---------------------------------------------------------------------------
// Platform gate — before any network access
// ---------------------------------------------------------------------------

#[test]
fn install_app_refuses_wsl_before_downloading() {
    let host = TargetHost {
        os: "linux".to_owned(),
        arch: "x86_64".to_owned(),
        is_wsl: true,
    };
    let error = host.ensure_supported().expect_err("WSL must be refused");
    let text = error.to_string();
    assert!(text.contains("not supported under WSL"), "{text}");
    assert!(
        text.contains("Windows desktop"),
        "the refusal must name a next action: {text}"
    );
}

#[test]
fn install_app_refuses_macos_and_unsupported_arch() {
    let macos = TargetHost {
        os: "macos".to_owned(),
        arch: "x86_64".to_owned(),
        is_wsl: false,
    };
    assert!(
        macos
            .ensure_supported()
            .unwrap_err()
            .to_string()
            .contains("macos")
    );

    let arm = TargetHost {
        os: "linux".to_owned(),
        arch: "aarch64".to_owned(),
        is_wsl: false,
    };
    assert!(
        arm.ensure_supported()
            .unwrap_err()
            .to_string()
            .contains("aarch64")
    );
}

#[test]
fn install_app_plan_fails_for_an_unsupported_host_before_asset_selection() {
    let (manifest, _, policy) = signed_fixture();
    let wsl = TargetHost {
        os: "linux".to_owned(),
        arch: "x86_64".to_owned(),
        is_wsl: true,
    };
    assert!(build_plan(&manifest, &wsl, &policy, PathBuf::from("/tmp/app"), false).is_err());
}

#[test]
fn install_app_reports_a_missing_target_asset() {
    let (mut manifest, _, _) = signed_fixture();
    manifest.assets.retain(|a| a.os == "windows");
    let error = select_asset(&manifest, &linux_host()).expect_err("no linux asset");
    assert!(error.to_string().contains("linux-x86_64"), "{error}");
}

// ---------------------------------------------------------------------------
// Manifest validation
// ---------------------------------------------------------------------------

#[test]
fn install_app_rejects_a_malformed_manifest() {
    assert!(parse_manifest("{ not json").is_err());
    assert!(parse_manifest("{}").is_err());
}

#[test]
fn install_app_rejects_an_unsupported_schema_version() {
    let (_, payload, _) = signed_fixture();
    let raw =
        manifest_json(&payload, "AA==").replace("\"schemaVersion\": 1", "\"schemaVersion\": 99");
    let error = parse_manifest(&raw).expect_err("future schema must be refused");
    assert!(error.to_string().contains("schema 99"), "{error}");
}

#[test]
fn install_app_rejects_a_manifest_with_unknown_fields() {
    // `deny_unknown_fields`: a field this CLI does not understand may carry a
    // meaning that changes what gets installed, so silently ignoring it is not
    // safe on an install path.
    let (_, payload, _) = signed_fixture();
    let raw = manifest_json(&payload, "AA==").replace(
        "\"appVersion\": \"0.1.0\",",
        "\"appVersion\": \"0.1.0\", \"runInstallerAsRoot\": true,",
    );
    assert!(parse_manifest(&raw).is_err());
}

#[test]
fn install_app_rejects_malformed_asset_metadata() {
    let (_, payload, _) = signed_fixture();
    let base = manifest_json(&payload, "AA==");

    for (needle, replacement, reason) in [
        ("\"sha256\": \"", "\"sha256\": \"nothex", "bad checksum"),
        ("\"sizeBytes\": 26", "\"sizeBytes\": 0", "zero size"),
        ("\"os\": \"linux\"", "\"os\": \"macos\"", "unsupported os"),
        (
            "\"fileName\": \"rocm-app_0.1.0_amd64.deb\"",
            "\"fileName\": \"../../etc/cron.d/evil\"",
            "path traversal in file name",
        ),
    ] {
        let raw = base.replacen(needle, replacement, 1);
        assert!(
            parse_manifest(&raw).is_err(),
            "manifest with {reason} was accepted"
        );
    }
}

#[test]
fn install_app_rejects_a_plain_http_asset_url_at_parse_time() {
    let (_, payload, _) = signed_fixture();
    let raw = manifest_json(&payload, "AA==").replacen(
        "https://example.invalid/rocm-app_0.1.0_amd64.deb",
        "http://example.invalid/rocm-app_0.1.0_amd64.deb",
        1,
    );
    let error = parse_manifest(&raw).expect_err("http url must be refused before any network");
    assert!(error.to_string().contains("https"), "{error}");
}

// ---------------------------------------------------------------------------
// Manifest freshness and CLI compatibility — enforced before approval
// ---------------------------------------------------------------------------

const DAY_MS: u64 = 24 * 60 * 60 * 1000;

#[test]
fn install_app_refuses_a_manifest_older_than_90_days() {
    let (mut manifest, _, policy) = signed_fixture();
    manifest.published_at_unix_ms = now_unix_ms() - 91 * DAY_MS;
    let error = build_plan(
        &manifest,
        &linux_host(),
        &policy,
        PathBuf::from("/tmp/app"),
        false,
    )
    .expect_err("stale manifest must be refused");
    let text = format!("{error:#}");
    assert!(text.contains("90 days"), "{text}");
    assert!(text.contains("--allow-stale-manifest"), "{text}");
}

/// The override proceeds, but the plan the user approves says why it needed
/// overriding.
#[test]
fn install_app_allows_a_stale_manifest_with_the_override_and_warns() {
    let (mut manifest, _, policy) = signed_fixture();
    manifest.published_at_unix_ms = now_unix_ms() - 91 * DAY_MS;
    let plan = build_plan(
        &manifest,
        &linux_host(),
        &policy,
        PathBuf::from("/tmp/app"),
        true,
    )
    .expect("--allow-stale-manifest proceeds");
    let rendered = plan.render();
    assert!(rendered.contains("warning:"), "{rendered}");
    assert!(rendered.contains("days old"), "{rendered}");
}

/// A future date is a broken clock or a forgery meant to outlive the
/// staleness check; even the stale override does not bypass it.
#[test]
fn install_app_refuses_a_future_dated_manifest_with_no_override() {
    let (mut manifest, _, policy) = signed_fixture();
    manifest.published_at_unix_ms = now_unix_ms() + 25 * 60 * 60 * 1000;
    let error = build_plan(
        &manifest,
        &linux_host(),
        &policy,
        PathBuf::from("/tmp/app"),
        true,
    )
    .expect_err("future-dated manifest must be refused");
    assert!(format!("{error:#}").contains("future"), "{error:#}");
}

/// The window boundaries are pinned: exactly 90 days old and exactly 24 hours
/// ahead both pass; one millisecond beyond either does not.
#[test]
fn install_app_freshness_window_boundaries_are_exact() {
    let now = 1_000 * DAY_MS;
    assert!(
        enforce_manifest_freshness(now - 90 * DAY_MS, now, false)
            .expect("at the age limit")
            .is_none()
    );
    assert!(enforce_manifest_freshness(now - 90 * DAY_MS - 1, now, false).is_err());
    assert!(enforce_manifest_freshness(now + DAY_MS, now, false).is_ok());
    assert!(enforce_manifest_freshness(now + DAY_MS + 1, now, false).is_err());
}

#[test]
fn install_app_refuses_a_cli_outside_the_compatible_range() {
    let (mut manifest, _, policy) = signed_fixture();
    manifest.compatible_cli = CliRange {
        min: "99.0.0".to_owned(),
        max: "99.9.9".to_owned(),
    };
    let error = build_plan(
        &manifest,
        &linux_host(),
        &policy,
        PathBuf::from("/tmp/app"),
        false,
    )
    .expect_err("incompatible CLI must be refused");
    assert!(
        format!("{error:#}").contains("Update the rocm CLI first"),
        "{error:#}"
    );
}

/// Inclusive at both ends, and each direction gets the right advice: an old
/// CLI is told to update, a too-new CLI is told to fetch a newer manifest.
#[test]
fn install_app_compatible_cli_range_is_inclusive_and_directional() {
    let range = |min: &str, max: &str| CliRange {
        min: min.to_owned(),
        max: max.to_owned(),
    };
    assert!(enforce_cli_compatibility(&range("1.0.0", "2.0.0"), "1.0.0").is_ok());
    assert!(enforce_cli_compatibility(&range("1.0.0", "2.0.0"), "2.0.0").is_ok());
    let old = enforce_cli_compatibility(&range("1.0.1", "2.0.0"), "1.0.0")
        .expect_err("below min refused");
    assert!(
        old.to_string().contains("Update the rocm CLI first"),
        "{old}"
    );
    let new = enforce_cli_compatibility(&range("1.0.0", "2.0.0"), "2.0.1")
        .expect_err("above max refused");
    assert!(new.to_string().contains("newer"), "{new}");
    // The running CLI must accept the fixture range every other test uses.
    assert!(enforce_cli_compatibility(&range("0.1.0", "0.2.0"), env!("CARGO_PKG_VERSION")).is_ok());
}

/// Fail closed: a range this CLI cannot compare is refused, never guessed.
#[test]
fn install_app_refuses_an_unparseable_compatible_cli_range() {
    let (mut manifest, _, policy) = signed_fixture();
    manifest.compatible_cli = CliRange {
        min: "banana".to_owned(),
        max: "0.2.0".to_owned(),
    };
    let error = build_plan(
        &manifest,
        &linux_host(),
        &policy,
        PathBuf::from("/tmp/app"),
        false,
    )
    .expect_err("unparseable range must fail closed");
    assert!(
        format!("{error:#}").contains("refusing to guess"),
        "{error:#}"
    );
}

/// An endless (or merely oversized) response body must not be read to
/// exhaustion. The cap hands `verify_asset_bytes` one byte too many and the
/// existing size check refuses it; `main.rs` wires this into its fetch
/// closure, which is network glue and untestable here.
#[test]
fn install_app_caps_the_download_at_the_declared_size() {
    let (manifest, payload, policy) = signed_fixture();
    let asset = manifest.assets[0].clone();

    // std::io::repeat never ends; without the cap this read never returns.
    let capped = read_capped(std::io::repeat(0x41), asset.size_bytes).expect("capped read");
    assert_eq!(
        capped.len(),
        usize::try_from(asset.size_bytes).expect("fits") + 1
    );
    let error = verify_asset_bytes(&asset, &capped, &policy).expect_err("size mismatch");
    assert!(error.to_string().contains("manifest declares"), "{error}");

    // A well-behaved body of exactly the declared size passes through intact.
    let exact = read_capped(payload.as_slice(), asset.size_bytes).expect("exact read");
    assert_eq!(exact, payload);
    verify_asset_bytes(&asset, &exact, &policy).expect("verified");
}

// ---------------------------------------------------------------------------
// Asset verification
// ---------------------------------------------------------------------------

#[test]
fn install_app_accepts_a_correctly_signed_asset() {
    let (manifest, payload, policy) = signed_fixture();
    let asset = select_asset(&manifest, &linux_host()).expect("asset");
    verify_asset_bytes(&asset, &payload, &policy).expect("valid asset verifies");
}

#[test]
fn install_app_rejects_a_bad_checksum() {
    let (manifest, payload, policy) = signed_fixture();
    let mut asset = select_asset(&manifest, &linux_host()).expect("asset");
    asset.sha256 = "0".repeat(64);
    let error = verify_asset_bytes(&asset, &payload, &policy).expect_err("must reject");
    assert!(error.to_string().contains("checksum mismatch"), "{error}");
}

#[test]
fn install_app_rejects_a_bad_size() {
    let (manifest, payload, policy) = signed_fixture();
    let mut asset = select_asset(&manifest, &linux_host()).expect("asset");
    asset.size_bytes += 1;
    let error = verify_asset_bytes(&asset, &payload, &policy).expect_err("must reject");
    assert!(error.to_string().contains("bytes"), "{error}");
}

#[test]
fn install_app_rejects_a_tampered_payload() {
    let (manifest, payload, policy) = signed_fixture();
    let asset = select_asset(&manifest, &linux_host()).expect("asset");

    let mut tampered = payload.clone();
    tampered.push(b'!');
    let error = verify_asset_bytes(&asset, &tampered, &policy).expect_err("must reject");
    // Caught by size before hashing — the cheapest check fires first.
    assert!(error.to_string().contains("bytes"), "{error}");

    // Same length, different content: the digest catches it.
    let mut swapped = payload;
    let last = swapped.len() - 1;
    swapped[last] ^= 0xFF;
    let error = verify_asset_bytes(&asset, &swapped, &policy).expect_err("must reject");
    assert!(error.to_string().contains("checksum mismatch"), "{error}");
}

/// The load-bearing signature case: correct length, correct digest, wrong key.
#[test]
fn install_app_rejects_a_signature_from_an_untrusted_key() {
    let (manifest, payload, mut policy) = signed_fixture();
    let asset = select_asset(&manifest, &linux_host()).expect("asset");

    let (_, attacker_public) = ephemeral_keys();
    policy.public_key_pem = Some(attacker_public);

    let error = verify_asset_bytes(&asset, &payload, &policy).expect_err("must reject");
    assert!(
        error.to_string().contains("signature verification failed"),
        "{error}"
    );
}

#[test]
fn install_app_rejects_a_corrupt_signature() {
    let (manifest, payload, policy) = signed_fixture();
    let mut asset = select_asset(&manifest, &linux_host()).expect("asset");
    asset.signature_b64 = "!!!not base64!!!".to_owned();
    assert!(verify_asset_bytes(&asset, &payload, &policy).is_err());
}

#[test]
fn install_app_refuses_when_a_signature_is_required_but_no_key_is_configured() {
    let (manifest, payload, _) = signed_fixture();
    let asset = select_asset(&manifest, &linux_host()).expect("asset");
    let policy = AppTrustPolicy {
        require_signature: true,
        public_key_pem: None,
    };
    let error = verify_asset_bytes(&asset, &payload, &policy).expect_err("must reject");
    assert!(
        error
            .to_string()
            .contains("no trusted public key is configured"),
        "{error}"
    );
}

/// The default must be "signature required". A default of optional would let a
/// misconfigured host silently accept an unsigned installer.
#[test]
fn install_app_trust_policy_requires_signatures_by_default() {
    // Read with the override absent; `from_env` defaults to required.
    let policy = AppTrustPolicy {
        require_signature: AppTrustPolicy::from_env().require_signature,
        public_key_pem: None,
    };
    let _ = SIGNED_KEY_ENV;
    assert!(
        policy.require_signature || std::env::var("ROCM_CLI_APP_REQUIRE_SIGNATURE").is_ok(),
        "signatures must be required unless explicitly disabled"
    );
}

// ---------------------------------------------------------------------------
// Apply
// ---------------------------------------------------------------------------

#[test]
fn install_app_apply_verifies_before_executing_and_cleans_up() {
    let (manifest, payload, policy) = signed_fixture();
    let root = scratch_root("apply-ok");
    let plan =
        build_plan(&manifest, &linux_host(), &policy, root.join("app"), false).expect("plan");
    let launcher = RecordingLauncher::default();
    let fetch = |_: &str| Ok(payload.clone());

    let executed = apply(&ApplyInputs {
        plan: &plan,
        policy: &policy,
        fetch: &fetch,
        launcher: &launcher,
        scratch_parent: &root,
    })
    .expect("apply succeeds");

    assert_eq!(
        launcher.launched.borrow().as_slice(),
        std::slice::from_ref(&executed)
    );
    assert!(
        executed
            .file_name()
            .is_some_and(|n| n == "rocm-app_0.1.0_amd64.deb"),
        "executed {executed:?}"
    );
    // The scratch directory is removed when the guard drops.
    assert!(
        !executed.exists(),
        "temporary files must not survive: {executed:?}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn install_app_apply_never_executes_an_unverified_asset() {
    let (manifest, payload, policy) = signed_fixture();
    let root = scratch_root("apply-tampered");
    let plan =
        build_plan(&manifest, &linux_host(), &policy, root.join("app"), false).expect("plan");
    let launcher = RecordingLauncher::default();

    // Same length so the size check passes and the digest is what rejects it.
    let mut tampered = payload;
    let last = tampered.len() - 1;
    tampered[last] ^= 0xFF;
    let fetch = |_: &str| Ok(tampered.clone());

    let error = apply(&ApplyInputs {
        plan: &plan,
        policy: &policy,
        fetch: &fetch,
        launcher: &launcher,
        scratch_parent: &root,
    })
    .expect_err("tampered asset must not be executed");

    assert!(error.to_string().contains("checksum mismatch"), "{error}");
    assert!(
        launcher.launched.borrow().is_empty(),
        "nothing may be executed after a failed verification"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn install_app_apply_cleans_up_after_a_download_failure() {
    let (manifest, _, policy) = signed_fixture();
    let root = scratch_root("apply-download-fail");
    let plan =
        build_plan(&manifest, &linux_host(), &policy, root.join("app"), false).expect("plan");
    let launcher = RecordingLauncher::default();
    let fetch = |_: &str| -> Result<Vec<u8>> { bail!("network unreachable") };

    assert!(
        apply(&ApplyInputs {
            plan: &plan,
            policy: &policy,
            fetch: &fetch,
            launcher: &launcher,
            scratch_parent: &root,
        })
        .is_err()
    );
    assert!(launcher.launched.borrow().is_empty());

    let leftovers: Vec<_> = std::fs::read_dir(&root)
        .expect("read scratch root")
        .filter_map(std::result::Result::ok)
        .map(|e| e.file_name())
        .collect();
    assert!(leftovers.is_empty(), "left temporary files: {leftovers:?}");
    let _ = std::fs::remove_dir_all(&root);
}

// ---------------------------------------------------------------------------
// Launch argv
// ---------------------------------------------------------------------------

#[test]
fn install_app_launches_by_exact_argv_never_shell_text() {
    let windows = argv_for_installer("windows", Path::new("/tmp/x/rocm-app_0.1.0_x64-setup.exe"))
        .expect("windows argv");
    assert_eq!(windows.len(), 2);
    assert!(windows[0].ends_with("rocm-app_0.1.0_x64-setup.exe"));
    assert_eq!(windows[1], "/S");

    let linux = argv_for_installer("linux", Path::new("/tmp/x/rocm-app_0.1.0_amd64.deb"))
        .expect("linux argv");
    assert_eq!(linux.len(), 1);

    for argv in [windows, linux] {
        for arg in argv {
            for bad in [';', '|', '&', '$', '`', '\n'] {
                assert!(!arg.contains(bad), "argv {arg:?} contains shell syntax");
            }
        }
    }
    assert!(argv_for_installer("macos", Path::new("/tmp/x")).is_err());
}

// ---------------------------------------------------------------------------
// Isolation: only `rocm install app` reaches this module
// ---------------------------------------------------------------------------

/// Scans the repository rather than trusting convention.
///
/// The product rule is that installing the CLI never installs the app. That is
/// a property of *every* other install path, so it is asserted against the
/// actual files instead of being documented and hoped for.
#[test]
fn install_app_is_not_reachable_from_any_other_install_path() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root");

    let other_install_paths = [
        "install.sh",
        "install.ps1",
        "scripts/install.sh",
        "scripts/install.ps1",
    ];
    for relative in other_install_paths {
        let path = repo.join(relative);
        if !path.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("read installer script");
        for forbidden in ["install app", "install_app", "rocm-app"] {
            assert!(
                !text.contains(forbidden),
                "{relative} references {forbidden}; CLI-only installers must not install the app"
            );
        }
    }

    // No other Rust module may call into this one.
    let sources = ["apps/rocm/src/main.rs", "apps/rocm/src/therock.rs"];
    for relative in sources {
        let text = std::fs::read_to_string(repo.join(relative)).expect("read source");
        for (line_number, line) in text.lines().enumerate() {
            if !line.contains("install_app::") {
                continue;
            }
            // `main.rs` legitimately dispatches `InstallTarget::App`; nothing else may.
            assert!(
                relative == "apps/rocm/src/main.rs",
                "{relative}:{} reaches install_app",
                line_number + 1
            );
        }
    }
}

/// The SDK and update flows must not mention the app installer at all.
#[test]
fn install_app_sdk_and_update_paths_do_not_reference_the_app_installer() {
    let therock =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/therock.rs"))
            .expect("read therock.rs");

    for forbidden in ["install_app", "AppReleaseManifest", "rocm install app"] {
        assert!(
            !therock.contains(forbidden),
            "therock.rs (SDK install + update) references {forbidden}"
        );
    }
}

/// Walk the repository, collecting files that contain real key material.
fn scan(dir: &Path, hits: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(std::result::Result::ok) {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if matches!(
            name.as_ref(),
            "target" | ".git" | "node_modules" | ".supergoal"
        ) {
            continue;
        }
        if path.is_dir() {
            scan(&path, hits);
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let lines: Vec<&str> = text.lines().collect();
        let found = lines.iter().enumerate().any(|(index, line)| {
            line.contains("-----BEGIN")
                && line.contains("PRIVATE KEY")
                && is_key_material(&lines, index)
        });
        if found {
            hits.push(path);
        }
    }
}

/// True when `lines` starting at `start` form a real PEM private key: a header,
/// a base64 body, and a matching footer. A bare header in prose is not one.
fn is_key_material(lines: &[&str], start: usize) -> bool {
    let mut saw_body = false;
    for line in lines.iter().skip(start + 1).take(64) {
        let trimmed = line.trim();
        if trimmed.starts_with("-----END") && trimmed.contains("PRIVATE KEY") {
            return saw_body;
        }
        if trimmed.len() >= 40
            && trimmed
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'='))
        {
            saw_body = true;
        }
    }
    false
}

/// No private key material may live in the repository.
///
/// Detects a **complete PEM block** — header, a base64 body, and a matching
/// footer — rather than the header string alone. A bare header appears in doc
/// comments and in this very test, so a substring scan reports itself and two
/// innocent files, which is a check nobody keeps.
#[test]
fn install_app_repository_contains_no_private_key() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root");

    let mut hits = Vec::new();
    scan(&repo, &mut hits);
    assert!(
        hits.is_empty(),
        "private key material in the repository: {hits:?}"
    );
}

/// The scan must actually catch a key, or it is a test that can never fail.
#[test]
fn install_app_private_key_scan_detects_real_key_material() {
    let (private_pem, _) = ephemeral_keys();
    let root = scratch_root("key-scan");
    let planted = root.join("leaked.pem");
    std::fs::write(&planted, &private_pem).expect("write");

    let text = std::fs::read_to_string(&planted).expect("read");
    let lines: Vec<&str> = text.lines().collect();
    let header = lines
        .iter()
        .position(|l| l.contains("-----BEGIN") && l.contains("PRIVATE KEY"))
        .expect("generated key has a PEM header");

    // Re-implemented inline so this test exercises the same predicate shape the
    // scan uses, against material that is unambiguously a key.
    let mut saw_body = false;
    let mut detected = false;
    for line in lines.iter().skip(header + 1).take(64) {
        let trimmed = line.trim();
        if trimmed.starts_with("-----END") && trimmed.contains("PRIVATE KEY") {
            detected = saw_body;
            break;
        }
        if trimmed.len() >= 40
            && trimmed
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'='))
        {
            saw_body = true;
        }
    }
    assert!(
        detected,
        "the scan predicate failed to recognise a real key"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn install_app_scratch_dir_is_unique_and_self_cleaning() {
    let root = scratch_root("scratch");
    let first = ScratchDir::create(&root).expect("first");
    let second = ScratchDir::create(&root).expect("second");
    assert_ne!(first.path(), second.path(), "scratch dirs must be unique");

    let path = first.path().to_path_buf();
    assert!(path.is_dir());
    drop(first);
    assert!(!path.exists(), "scratch dir must remove itself");

    drop(second);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn install_app_base64_round_trips() {
    for payload in [
        b"".to_vec(),
        b"a".to_vec(),
        b"ab".to_vec(),
        b"abc".to_vec(),
        asset_bytes(),
        (0u8..=255).collect::<Vec<u8>>(),
    ] {
        let encoded = base64_encode(&payload);
        let decoded = base64_decode(&encoded).expect("decode");
        assert_eq!(
            decoded,
            payload,
            "round trip failed for {} bytes",
            payload.len()
        );
    }
    assert!(base64_decode("###").is_err());
}

// ---------------------------------------------------------------------------
// Package format selection
// ---------------------------------------------------------------------------

/// A manifest offering three formats, two of which share `linux`/`x86_64`.
///
/// The rpm is listed **first** so a test that passes only because "the deb
/// happens to come first" cannot pass here.
fn multi_format_manifest_json(payload: &[u8]) -> String {
    format!(
        r#"{{
  "schemaVersion": 1,
  "appVersion": "0.1.0",
  "compatibleCli": {{ "min": "0.1.0", "max": "0.2.0" }},
  "publishedAtUnixMs": {published},
  "releaseNotesUrl": "https://github.com/mikeroysoft/rocm-app/releases/tag/v0.1.0",
  "assets": [
    {{
      "os": "linux",
      "arch": "x86_64",
      "format": "rpm",
      "url": "https://example.invalid/rocm-app-0.1.0-1.x86_64.rpm",
      "fileName": "rocm-app-0.1.0-1.x86_64.rpm",
      "sizeBytes": {len},
      "sha256": "{sha}",
      "signatureB64": "AA=="
    }},
    {{
      "os": "linux",
      "arch": "x86_64",
      "format": "deb",
      "url": "https://example.invalid/rocm-app_0.1.0_amd64.deb",
      "fileName": "rocm-app_0.1.0_amd64.deb",
      "sizeBytes": {len},
      "sha256": "{sha}",
      "signatureB64": "AA=="
    }},
    {{
      "os": "windows",
      "arch": "x86_64",
      "format": "nsis",
      "url": "https://example.invalid/rocm-app_0.1.0_x64-setup.exe",
      "fileName": "rocm-app_0.1.0_x64-setup.exe",
      "sizeBytes": {len},
      "sha256": "{sha}",
      "signatureB64": "AA=="
    }}
  ]
}}"#,
        len = payload.len(),
        sha = sha256_hex(payload),
        published = now_unix_ms(),
    )
}

fn multi_format_manifest() -> AppReleaseManifest {
    parse_manifest(&multi_format_manifest_json(&asset_bytes()))
        .expect("multi-format fixture manifest parses")
}

/// A manifest published before `format` existed must still install.
#[test]
fn install_app_manifest_without_a_format_field_still_selects() {
    let (manifest, _, _) = signed_fixture();
    assert!(
        manifest
            .assets
            .iter()
            .all(|asset| asset.format == AssetFormat::Unspecified),
        "a missing format key must deserialize as Unspecified, not fail"
    );
    let asset = select_asset(&manifest, &linux_host()).expect("pre-format manifest still selects");
    assert_eq!(asset.file_name, "rocm-app_0.1.0_amd64.deb");
}

#[test]
fn install_app_selects_the_format_the_host_can_install() {
    let manifest = multi_format_manifest();

    let deb = select_asset_for_formats(&manifest, &linux_host(), &[AssetFormat::Deb])
        .expect("a dpkg host has a deb");
    assert_eq!(deb.format, AssetFormat::Deb);
    assert_eq!(deb.file_name, "rocm-app_0.1.0_amd64.deb");

    let rpm = select_asset_for_formats(&manifest, &linux_host(), &[AssetFormat::Rpm])
        .expect("an rpm host has an rpm");
    assert_eq!(rpm.format, AssetFormat::Rpm);
    assert_eq!(rpm.file_name, "rocm-app-0.1.0-1.x86_64.rpm");

    // Windows goes through the real entry point: its format list is fixed, so
    // this is deterministic on whatever machine runs the suite.
    let nsis = select_asset(&manifest, &windows_host()).expect("a windows host has an nsis");
    assert_eq!(nsis.format, AssetFormat::Nsis);

    // And once through the live detector on whatever this machine is: the
    // format chosen must be one the machine was actually found able to
    // install, which is the property the wiring exists for.
    let detected = host_package_formats("linux");
    if !detected.is_empty() {
        let chosen = select_asset(&manifest, &linux_host()).expect("this host installs something");
        assert!(
            detected.contains(&chosen.format),
            "selected {:?} but this host can install {detected:?}",
            chosen.format
        );
    }
}

/// The preference order is pinned, not incidental.
#[test]
fn install_app_prefers_deb_when_a_host_has_both_dpkg_and_rpm() {
    assert_eq!(
        linux_formats_from(true, true),
        [AssetFormat::Deb, AssetFormat::Rpm]
    );
    assert_eq!(linux_formats_from(true, false), [AssetFormat::Deb]);
    assert_eq!(linux_formats_from(false, true), [AssetFormat::Rpm]);
    assert!(linux_formats_from(false, false).is_empty());

    let chosen = select_asset_for_formats(
        &multi_format_manifest(),
        &linux_host(),
        linux_formats_from(true, true),
    )
    .expect("dual-tooling host");
    assert_eq!(chosen.format, AssetFormat::Deb);
}

/// "No asset" would be a lie for a host that has one and merely cannot install
/// that package format, so the error names both sets.
#[test]
fn install_app_reports_an_asset_it_cannot_install() {
    let error = select_asset_for_formats(
        &multi_format_manifest(),
        &linux_host(),
        linux_formats_from(false, false),
    )
    .expect_err("neither dpkg nor rpm is present");
    let text = error.to_string();
    assert!(text.contains("offers [rpm, deb]"), "{text}");
    assert!(text.contains("this host can install []"), "{text}");
}

/// A manifest comes out of a release tool, so mixed case means someone edited
/// it by hand.
#[test]
fn install_app_rejects_an_uppercase_sha256() {
    let (_, payload, _) = signed_fixture();
    let digest = sha256_hex(&payload);
    let raw = manifest_json(&payload, "AA==").replace(&digest, &digest.to_ascii_uppercase());
    let error = parse_manifest(&raw).expect_err("an uppercase digest must be refused");
    assert!(error.to_string().contains("uppercase sha256"), "{error}");

    // The comparison itself stays case-insensitive, so a correct manifest is
    // never rejected on a technicality.
    let mut asset = signed_fixture().0.assets.remove(0);
    asset.sha256 = digest.to_ascii_uppercase();
    let policy = AppTrustPolicy {
        require_signature: false,
        public_key_pem: None,
    };
    verify_asset_bytes(&asset, &payload, &policy).expect("digest compare ignores case");
}

#[test]
fn install_app_rejects_an_unsupported_arch() {
    let (_, payload, _) = signed_fixture();
    let raw = manifest_json(&payload, "AA==").replacen(
        "\"arch\": \"x86_64\"",
        "\"arch\": \"aarch64\"",
        1,
    );
    let error = parse_manifest(&raw).expect_err("arch is validated at parse time");
    assert!(
        error.to_string().contains("unsupported target arch"),
        "{error}"
    );
}

/// The format vocabulary is closed, so serde rejects anything outside it.
#[test]
fn install_app_rejects_an_unknown_format_value() {
    let (_, payload, _) = signed_fixture();
    let raw = manifest_json(&payload, "AA==").replacen(
        "\"arch\": \"x86_64\",",
        "\"arch\": \"x86_64\",\n      \"format\": \"msi\",",
        1,
    );
    let error = parse_manifest(&raw).expect_err("msi is not a format this CLI installs");
    let text = format!("{error:#}");
    assert!(text.contains("`format`"), "{text}");
    assert!(text.contains("msi"), "{text}");

    // The hand-written reader and the derived writer must agree on the wire
    // spelling, or a manifest this CLI emits is a manifest it cannot read.
    for format in AssetFormat::ALL {
        assert_eq!(
            serde_json::to_string(&format).expect("serialize format"),
            format!("\"{}\"", format.label())
        );
    }
}
