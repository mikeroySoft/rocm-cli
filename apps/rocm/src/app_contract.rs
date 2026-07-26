// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Versioned, read-only contract for the ROCm desktop app.
//!
//! # Why this is a separate surface
//!
//! `rocm examine --json` serializes [`rocm_core::Examination`], whose 50 top-level
//! keys are a frozen wire contract (`examine_render`'s sibling test
//! `examination_top_level_keys_match_examine_py_contract`). Adding app fields
//! there would break that contract, and the human `rocm examine` text is parsed
//! by the e2e capability harness. So the app gets its own additive surface and
//! every existing output is left byte-identical.
//!
//! # Two layers, on purpose
//!
//! [`build_snapshot`] is a **pure function** from [`SnapshotInputs`] to
//! [`AppSnapshot`]. [`gather_inputs`] does the probing. Splitting them is what
//! makes every verdict — including hosts this machine can never be, like WSL —
//! reachable from a deterministic unit test with no GPU, no network, and no
//! clock. A single probe-and-decide function would leave most of the state
//! machine untestable.
//!
//! # Driver data is read-only
//!
//! [`DriverReport`] carries version and support links and nothing else. There is
//! deliberately no driver variant in [`EligibleAction`]. rocm-cli itself does
//! have one driver-mutating flow (`rocm install driver --dkms --yes`), but it is
//! not reachable through this contract and must never become so.

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::Result;
use rocm_core::{AppPaths, RocmCliConfig, unix_time_millis};
use serde::{Deserialize, Serialize};

use crate::therock::{self, InstalledRuntimeManifest};

/// Current contract version. Bump on any breaking change to the payload shape.
///
/// Consumers reject a payload whose version they do not implement rather than
/// best-effort decoding it: a partially understood health report is worse than
/// a refusal, because it renders a confident wrong answer.
pub(crate) const SCHEMA_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Envelope
// ---------------------------------------------------------------------------

/// The complete app-facing snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AppSnapshot {
    /// Always positive. Consumers must check this before decoding anything else.
    pub schema_version: u32,
    pub producer: ProducerIdentity,
    /// When the observation was made, not when it was serialized.
    pub observed_at_unix_ms: u64,
    pub platform: PlatformReport,
    pub gpu: GpuIdentity,
    pub health: HealthReport,
    pub components: Vec<ComponentReport>,
    pub runtimes: Vec<RuntimeRecord>,
    pub driver: DriverReport,
    pub update: UpdateReport,
    /// Mutations this host may be offered. Empty on any unsupported platform.
    pub eligible_actions: Vec<EligibleAction>,
}

/// Who produced this payload and from which build.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ProducerIdentity {
    pub name: String,
    pub version: String,
    /// Build identity. `"unknown"` when the binary was not stamped — explicit,
    /// never an empty string.
    pub build: String,
}

impl ProducerIdentity {
    fn current() -> Self {
        Self {
            name: "rocm-cli".to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            build: option_env!("ROCM_CLI_BUILD_SHA")
                .unwrap_or("unknown")
                .to_owned(),
        }
    }
}

// ---------------------------------------------------------------------------
// Platform and hardware
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum OsFamily {
    Windows,
    Linux,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PlatformReport {
    pub os: OsFamily,
    pub arch: String,
    pub is_wsl: bool,
    pub support: SupportStatus,
}

/// Whether the app supports managing ROCm on this host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "state",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub(crate) enum SupportStatus {
    Supported,
    Unsupported { reason: ReasonCode },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GpuIdentity {
    pub name: Option<String>,
    pub gfx_target: Option<String>,
    /// Normalised TheRock family, e.g. `gfx120X-all`. `None` when the target is
    /// unrecognised — which is not the same as absent hardware.
    pub therock_family: Option<String>,
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

/// The single answer the app leads with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum HealthVerdict {
    Healthy,
    Unknown,
    SetupRequired,
    Attention,
    Unsupported,
}

/// Closed set of reasons a verdict was reached.
///
/// The verdict is *derived* from these, never from a process exit code. A new
/// situation must add a variant here rather than widening an existing one, so
/// the UI can always map a reason to specific copy and a next action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum ReasonCode {
    /// WSL cannot reach the GPU the way this product requires.
    PlatformWsl,
    /// Not native Windows or native Linux.
    PlatformUnsupportedOs,
    /// No AMD GPU found.
    GpuAbsent,
    /// GPU found, but its gfx target maps to no known TheRock family.
    GpuUnrecognisedFamily,
    /// No managed runtime installed.
    RuntimeAbsent,
    /// The active runtime failed validation.
    RuntimeValidationFailed,
    /// Config names an active runtime that is not in the registry.
    RuntimeActiveMissing,
    /// More than one runtime matches the configured selector.
    RuntimeAmbiguousSelection,
    /// The amdgpu kernel driver was not detected.
    DriverNotDetected,
    /// A newer managed runtime is available.
    UpdateAvailable,
    /// Update metadata failed its signature or trust policy.
    UpdateMetadataUntrusted,
    /// The update catalog could not be reached.
    UpdateOffline,
    /// One or more probes did not complete.
    ProbeIncomplete,
}

impl ReasonCode {
    /// The verdict this reason implies on its own.
    ///
    /// The snapshot's verdict is the maximum over all reasons, which is why
    /// [`HealthVerdict`] derives `Ord` in increasing-severity order.
    const fn severity(self) -> HealthVerdict {
        match self {
            Self::PlatformWsl | Self::PlatformUnsupportedOs => HealthVerdict::Unsupported,
            Self::RuntimeValidationFailed
            | Self::RuntimeActiveMissing
            | Self::RuntimeAmbiguousSelection
            | Self::UpdateMetadataUntrusted
            | Self::DriverNotDetected
            | Self::GpuAbsent => HealthVerdict::Attention,
            Self::RuntimeAbsent | Self::GpuUnrecognisedFamily => HealthVerdict::SetupRequired,
            Self::ProbeIncomplete | Self::UpdateOffline | Self::UpdateAvailable => {
                HealthVerdict::Unknown
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct HealthReason {
    pub code: ReasonCode,
    /// Human-readable evidence. Never load-bearing for logic.
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct HealthReport {
    pub verdict: HealthVerdict,
    pub reasons: Vec<HealthReason>,
    /// The one thing the user should do next, if anything.
    pub next_action: Option<String>,
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum ComponentKind {
    /// Supplied by the consumer; the CLI cannot know its own GUI's version.
    App,
    Cli,
    Driver,
    /// System-wide HIP / ROCm installation, outside managed runtimes.
    SystemHipRocm,
    ManagedRuntime,
    Python,
    PyTorch,
    Engine,
}

/// A component's version state.
///
/// Every situation is a distinct variant. Earlier surfaces in this repo
/// overload `""` and `"unknown"` strings for "absent", "not probed", and
/// "probe failed"; a consumer cannot tell those apart, so it cannot choose
/// between "install this" and "we could not check".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "state",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub(crate) enum ComponentState {
    /// Present, and the newest compatible version known.
    LatestCompatible { version: String },
    /// Present at a supported version.
    Installed { version: String },
    /// Present, with a newer compatible version published.
    UpdateAvailable { installed: String, latest: String },
    /// Present but not supported by this product.
    Unsupported { version: String, reason: String },
    /// Absent, and that is a determinate answer.
    NotInstalled,
    /// Last known value, past its freshness window.
    Stale {
        version: Option<String>,
        checked_at_unix_ms: u64,
    },
    /// Could not be determined. `reason` says why — never an empty string.
    Unknown { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ComponentReport {
    pub kind: ComponentKind,
    /// Distinguishes instances within a kind, e.g. `lemonade` vs `vllm`.
    pub name: String,
    pub state: ComponentState,
}

// ---------------------------------------------------------------------------
// Runtimes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "state",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub(crate) enum RuntimeValidation {
    Ready,
    Failed {
        detail: String,
    },
    /// Not yet validated. An install is never activated from this state.
    Unvalidated,
}

/// Where a runtime came from. Provenance matters because an adopted or imported
/// tree is not ours to modify.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub(crate) enum InstallSource {
    Index {
        url: String,
    },
    Tarball {
        url: String,
        file_name: String,
    },
    /// Pre-existing tree registered read-only.
    Adopted {
        path: PathBuf,
    },
    Imported {
        path: PathBuf,
    },
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RuntimeRecord {
    /// Exact side-by-side key. Two runtimes never share one.
    pub key: String,
    /// Friendly selector, e.g. `therock-nightly:gfx120X-all`. Not unique across
    /// versions, so it must never be used as an identity.
    pub runtime_id: String,
    pub version: String,
    pub active: bool,
    pub previous: bool,
    pub validation: RuntimeValidation,
    pub channel: String,
    pub family: String,
    pub format: String,
    pub install_source: InstallSource,
    pub install_root: PathBuf,
    pub read_only: bool,
}

// ---------------------------------------------------------------------------
// Driver — read-only by construction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "state",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub(crate) enum DriverVersionState {
    /// A concrete version string was read.
    Known {
        version: String,
    },
    /// The driver is loaded but exposes no version on this platform.
    DetectedWithoutVersion {
        detail: String,
    },
    NotDetected {
        detail: String,
    },
    Unknown {
        reason: String,
    },
}

/// An official link. The app opens these; it never acts on the driver itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SupportLink {
    pub label: String,
    pub url: String,
}

/// Driver inventory.
///
/// Read-only by construction: there is no action, plan, or command field, and
/// no variant of [`EligibleAction`] targets a driver.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DriverReport {
    pub installed: DriverVersionState,
    /// Newest version the catalog knows about, when it is trusted and fresh.
    pub latest_known: Option<String>,
    pub support_links: Vec<SupportLink>,
}

// ---------------------------------------------------------------------------
// Updates
// ---------------------------------------------------------------------------

/// Trust in the metadata the update answer came from.
///
/// An update is never claimed from untrusted or expired metadata, so this is
/// part of the contract rather than an implementation detail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub(crate) enum SourceTrust {
    /// Signature verified against the configured public key.
    Signed {
        key_source: String,
    },
    /// Signature not required by policy and not present.
    UnsignedAllowed,
    Untrusted {
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "state",
    rename_all = "kebab-case",
    rename_all_fields = "camelCase"
)]
pub(crate) enum UpdateState {
    NoUpdate {
        installed: String,
    },
    Available {
        installed: String,
        latest: String,
    },
    /// Installed version is newer than anything the index offers.
    AheadOfIndex {
        installed: String,
        latest: String,
    },
    /// The catalog could not be reached.
    Offline {
        detail: String,
    },
    /// Answer comes from a cache past its freshness window.
    Stale {
        installed: String,
        checked_at_unix_ms: u64,
    },
    /// Metadata was reachable but failed the trust policy.
    UntrustedMetadata {
        detail: String,
    },
    /// No managed runtime to check.
    NotApplicable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UpdateReport {
    pub state: UpdateState,
    pub checked_at_unix_ms: Option<u64>,
    pub trust: SourceTrust,
}

// ---------------------------------------------------------------------------
// Eligible actions
// ---------------------------------------------------------------------------

/// Mutations the app may offer on this host.
///
/// There is deliberately **no driver variant**. Adding one would make driver
/// mutation reachable from the desktop app, which the product forbids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
// The shared `Runtime` postfix is the point, not noise: it is the type-level
// record that every mutation this product offers targets a managed runtime and
// never a driver. Shortening to `Install`/`Remove` would drop that, and the
// wire names stay `install-runtime` either way.
#[allow(clippy::enum_variant_names)]
pub(crate) enum EligibleAction {
    InstallRuntime,
    UpdateRuntime,
    ActivateRuntime,
    RemoveRuntime,
    ValidateRuntime,
}

// ---------------------------------------------------------------------------
// Inputs to the pure builder
// ---------------------------------------------------------------------------

/// Everything [`build_snapshot`] needs, already probed.
#[derive(Debug, Clone)]
pub(crate) struct SnapshotInputs {
    pub observed_at_unix_ms: u64,
    pub platform: PlatformReport,
    pub gpu: GpuIdentity,
    pub runtimes: Vec<RuntimeRecord>,
    pub components: Vec<ComponentReport>,
    pub driver: DriverReport,
    pub update: UpdateReport,
    /// Probes that did not complete. Non-empty yields `ProbeIncomplete`.
    pub probe_failures: Vec<String>,
}

/// Derive the snapshot. Pure: no clock, no filesystem, no network.
pub(crate) fn build_snapshot(inputs: SnapshotInputs) -> AppSnapshot {
    let reasons = derive_reasons(&inputs);
    let verdict = reasons
        .iter()
        .map(|r| r.code.severity())
        .max()
        .unwrap_or(HealthVerdict::Healthy);

    // An unsupported host is offered nothing. This is the gate, not a UI concern:
    // a client that ignored the platform field would otherwise still receive a
    // list of actions it must never present.
    let eligible_actions = if matches!(inputs.platform.support, SupportStatus::Supported) {
        derive_actions(&inputs)
    } else {
        Vec::new()
    };

    AppSnapshot {
        schema_version: SCHEMA_VERSION,
        producer: ProducerIdentity::current(),
        observed_at_unix_ms: inputs.observed_at_unix_ms,
        health: HealthReport {
            verdict,
            next_action: next_action_for(verdict, &reasons),
            reasons,
        },
        platform: inputs.platform,
        gpu: inputs.gpu,
        components: inputs.components,
        runtimes: inputs.runtimes,
        driver: inputs.driver,
        update: inputs.update,
        eligible_actions,
    }
}

fn derive_reasons(inputs: &SnapshotInputs) -> Vec<HealthReason> {
    let mut reasons = Vec::new();

    if let SupportStatus::Unsupported { reason } = inputs.platform.support {
        reasons.push(HealthReason {
            code: reason,
            detail: match reason {
                ReasonCode::PlatformWsl => {
                    "Running under WSL. ROCm App manages native Windows and native Linux hosts."
                        .to_owned()
                }
                _ => format!("Unsupported platform: {:?}.", inputs.platform.os),
            },
        });
        // Nothing below is meaningful on a host we refuse to manage, and
        // reporting it would invite a client to act on it anyway.
        return reasons;
    }

    if inputs.gpu.name.is_none() && inputs.gpu.gfx_target.is_none() {
        reasons.push(HealthReason {
            code: ReasonCode::GpuAbsent,
            detail: "No AMD GPU was detected on this computer.".to_owned(),
        });
    } else if inputs.gpu.therock_family.is_none() {
        reasons.push(HealthReason {
            code: ReasonCode::GpuUnrecognisedFamily,
            detail: format!(
                "GPU target {} does not map to a known ROCm family.",
                inputs.gpu.gfx_target.as_deref().unwrap_or("unknown")
            ),
        });
    }

    let active = inputs.runtimes.iter().find(|r| r.active);
    match active {
        None if inputs.runtimes.is_empty() => reasons.push(HealthReason {
            code: ReasonCode::RuntimeAbsent,
            detail: "No managed ROCm runtime is installed.".to_owned(),
        }),
        None => reasons.push(HealthReason {
            code: ReasonCode::RuntimeActiveMissing,
            detail: format!(
                "{} runtime(s) are installed but none is active.",
                inputs.runtimes.len()
            ),
        }),
        Some(runtime) => match &runtime.validation {
            RuntimeValidation::Failed { detail } => reasons.push(HealthReason {
                code: ReasonCode::RuntimeValidationFailed,
                detail: format!("Active runtime {} failed validation: {detail}", runtime.key),
            }),
            RuntimeValidation::Unvalidated => reasons.push(HealthReason {
                code: ReasonCode::RuntimeValidationFailed,
                detail: format!("Active runtime {} has not been validated.", runtime.key),
            }),
            RuntimeValidation::Ready => {}
        },
    }

    // A validated active managed runtime is a working ROCm install. Absence of a
    // system-wide /opt/rocm is then irrelevant and must not be reported as a
    // problem — that is the difference between "not set up" and "set up
    // differently than you expected".
    if matches!(
        &inputs.driver.installed,
        DriverVersionState::NotDetected { .. }
    ) {
        reasons.push(HealthReason {
            code: ReasonCode::DriverNotDetected,
            detail: "The amdgpu kernel driver was not detected.".to_owned(),
        });
    }

    match &inputs.update.state {
        UpdateState::Available { installed, latest } => reasons.push(HealthReason {
            code: ReasonCode::UpdateAvailable,
            detail: format!("ROCm {latest} is available; {installed} is installed."),
        }),
        UpdateState::UntrustedMetadata { detail } => reasons.push(HealthReason {
            code: ReasonCode::UpdateMetadataUntrusted,
            detail: format!("Update metadata was rejected: {detail}"),
        }),
        UpdateState::Offline { detail } => reasons.push(HealthReason {
            code: ReasonCode::UpdateOffline,
            detail: format!("Could not check for updates: {detail}"),
        }),
        UpdateState::NoUpdate { .. }
        | UpdateState::AheadOfIndex { .. }
        | UpdateState::Stale { .. }
        | UpdateState::NotApplicable => {}
    }

    if !inputs.probe_failures.is_empty() {
        reasons.push(HealthReason {
            code: ReasonCode::ProbeIncomplete,
            detail: format!("{} probe(s) did not complete.", inputs.probe_failures.len()),
        });
    }

    reasons
}

fn derive_actions(inputs: &SnapshotInputs) -> Vec<EligibleAction> {
    let mut actions = BTreeSet::new();

    if inputs.runtimes.is_empty() {
        actions.insert(EligibleAction::InstallRuntime);
    } else {
        actions.insert(EligibleAction::InstallRuntime);
        actions.insert(EligibleAction::ValidateRuntime);
        if inputs.runtimes.iter().any(|r| !r.active) {
            actions.insert(EligibleAction::ActivateRuntime);
        }
        if inputs.runtimes.iter().any(|r| !r.active && !r.read_only) {
            actions.insert(EligibleAction::RemoveRuntime);
        }
    }

    if matches!(inputs.update.state, UpdateState::Available { .. }) {
        actions.insert(EligibleAction::UpdateRuntime);
    }

    actions.into_iter().collect()
}

fn next_action_for(verdict: HealthVerdict, reasons: &[HealthReason]) -> Option<String> {
    // The next action follows the most severe reason, so the user is told about
    // the thing that actually blocks them rather than the first thing checked.
    let primary = reasons.iter().max_by_key(|r| r.code.severity())?;
    let text = match primary.code {
        ReasonCode::PlatformWsl => "Run ROCm App on your Windows desktop instead.",
        ReasonCode::PlatformUnsupportedOs => "Use native Windows or native Linux.",
        ReasonCode::GpuAbsent => "Connect a supported AMD GPU, then check again.",
        ReasonCode::GpuUnrecognisedFamily => "Check the supported hardware list.",
        ReasonCode::RuntimeAbsent => "Set up ROCm.",
        ReasonCode::RuntimeValidationFailed => "Repair or reinstall the active ROCm runtime.",
        ReasonCode::RuntimeActiveMissing => "Choose which installed ROCm version to use.",
        ReasonCode::RuntimeAmbiguousSelection => "Choose exactly one ROCm version to use.",
        ReasonCode::DriverNotDetected => "Check the AMD driver release notes for your GPU.",
        ReasonCode::UpdateAvailable => "Review the available ROCm update.",
        ReasonCode::UpdateMetadataUntrusted => "Retry later; update metadata could not be trusted.",
        ReasonCode::UpdateOffline => "Reconnect to check for ROCm updates.",
        ReasonCode::ProbeIncomplete => "Refresh to complete the remaining checks.",
    };
    debug_assert_ne!(verdict, HealthVerdict::Healthy, "healthy has no reasons");
    Some(text.to_owned())
}

// ---------------------------------------------------------------------------
// Probing
// ---------------------------------------------------------------------------

/// Probe this machine into [`SnapshotInputs`].
pub(crate) fn gather_inputs(paths: &AppPaths, config: &RocmCliConfig) -> Result<SnapshotInputs> {
    let mut probe_failures = Vec::new();

    let examination = rocm_core::Examination::probe(rocm_core::FrameworkProbe::Skip);
    probe_failures.extend(examination.probe_failures.iter().cloned());

    let manifests = therock::load_runtime_manifests(paths).unwrap_or_else(|e| {
        probe_failures.push(format!("runtime registry: {e}"));
        Vec::new()
    });

    let runtimes = manifests
        .iter()
        .map(|m| runtime_record(m, config))
        .collect::<Vec<_>>();

    Ok(SnapshotInputs {
        // `unix_time_millis` is u128, but the contract uses u64: serde's
        // internally-tagged enum buffering has no u128 support, so a u128 field
        // inside a tagged variant fails to deserialize with "u128 is not
        // supported". u64 milliseconds overflow in ~584 million years.
        observed_at_unix_ms: u64::try_from(unix_time_millis()).unwrap_or(u64::MAX),
        platform: platform_report(&examination),
        gpu: gpu_identity(
            &rocm_core::detect_host_gpu_summary(Some(paths)),
            &examination,
        ),
        components: component_reports(&examination, &runtimes),
        driver: driver_report(&examination),
        // A live update check performs network I/O and belongs to the explicit
        // update flow, not to every snapshot read. Reporting `Stale`/`NotApplicable`
        // here is honest; claiming `NoUpdate` without checking would not be.
        update: UpdateReport {
            state: if runtimes.is_empty() {
                UpdateState::NotApplicable
            } else {
                UpdateState::Stale {
                    installed: runtimes
                        .iter()
                        .find(|r| r.active)
                        .map_or_else(|| runtimes[0].version.clone(), |r| r.version.clone()),
                    checked_at_unix_ms: 0,
                }
            },
            checked_at_unix_ms: None,
            trust: SourceTrust::Untrusted {
                reason: "no update check performed for this snapshot".to_owned(),
            },
        },
        runtimes,
        probe_failures,
    })
}

fn platform_report(examination: &rocm_core::Examination) -> PlatformReport {
    let os = if cfg!(target_os = "windows") {
        OsFamily::Windows
    } else if cfg!(target_os = "linux") {
        OsFamily::Linux
    } else {
        OsFamily::Other
    };
    let support = if examination.is_wsl {
        SupportStatus::Unsupported {
            reason: ReasonCode::PlatformWsl,
        }
    } else if matches!(os, OsFamily::Other) {
        SupportStatus::Unsupported {
            reason: ReasonCode::PlatformUnsupportedOs,
        }
    } else {
        SupportStatus::Supported
    };
    PlatformReport {
        os,
        arch: std::env::consts::ARCH.to_owned(),
        is_wsl: examination.is_wsl,
        support,
    }
}

/// Merge the two GPU-identity sources this repository has.
///
/// `Examination` learns `gfx_target` only from `rocminfo`, which is absent on a
/// machine whose ROCm lives entirely in a managed runtime — the common case for
/// this product. `detect_host_gpu_summary` reads KFD sysfs and answers there.
/// Preferring only `Examination` reported a live gfx1201 host as
/// `gpu-unrecognised-family` / `setup-required` while a validated runtime was
/// active: a confidently wrong verdict, which is the exact failure this
/// contract exists to prevent.
fn gpu_identity(
    host: &rocm_core::HostGpuSummary,
    examination: &rocm_core::Examination,
) -> GpuIdentity {
    let probed = examination.gpus.iter().find(|g| g.is_amd);
    let examined_target = probed
        .map(|g| g.gfx_target.clone())
        .filter(|t| !t.is_empty());

    let gfx_target = host.gfx_target.clone().or(examined_target);
    GpuIdentity {
        name: host
            .name
            .clone()
            .or_else(|| probed.map(|g| g.name.clone()))
            .filter(|n| !n.is_empty()),
        therock_family: host.therock_family.clone().or_else(|| {
            gfx_target
                .as_deref()
                .and_then(rocm_core::normalize_therock_family)
        }),
        gfx_target,
    }
}

fn driver_report(examination: &rocm_core::Examination) -> DriverReport {
    // Windows exposes a numeric Adrenalin version; Linux exposes only module
    // presence. Reporting "unknown" on Linux is accurate — inventing a version
    // from the ROCm install would be a different fact wearing this label.
    let installed = if examination.adrenalin_version.is_empty() {
        match examination.amdgpu_loaded {
            Some(true) => DriverVersionState::DetectedWithoutVersion {
                detail: "amdgpu kernel module is loaded".to_owned(),
            },
            Some(false) => DriverVersionState::NotDetected {
                detail: "amdgpu kernel module is not loaded".to_owned(),
            },
            None => DriverVersionState::Unknown {
                reason: "driver state is not observable on this platform".to_owned(),
            },
        }
    } else {
        DriverVersionState::Known {
            version: examination.adrenalin_version.clone(),
        }
    };

    DriverReport {
        installed,
        latest_known: None,
        support_links: vec![SupportLink {
            label: "AMD driver downloads and release notes".to_owned(),
            url: "https://www.amd.com/en/support".to_owned(),
        }],
    }
}

fn runtime_record(manifest: &InstalledRuntimeManifest, config: &RocmCliConfig) -> RuntimeRecord {
    let status = crate::runtime_usability_status(manifest);
    let validation = if status == "ready" {
        RuntimeValidation::Ready
    } else {
        RuntimeValidation::Failed { detail: status }
    };

    RuntimeRecord {
        active: config.active_runtime_key.as_deref() == Some(manifest.runtime_key.as_str()),
        previous: config.previous_runtime_key.as_deref() == Some(manifest.runtime_key.as_str()),
        key: manifest.runtime_key.clone(),
        runtime_id: manifest.runtime_id.clone(),
        version: manifest.version.clone(),
        validation,
        channel: manifest.channel.clone(),
        family: manifest.family.clone(),
        format: manifest.format.clone(),
        install_source: install_source(manifest),
        install_root: manifest.install_root.clone(),
        read_only: manifest.read_only,
    }
}

fn install_source(manifest: &InstalledRuntimeManifest) -> InstallSource {
    if let Some(path) = &manifest.imported_from {
        return if manifest.read_only {
            InstallSource::Adopted { path: path.clone() }
        } else {
            InstallSource::Imported { path: path.clone() }
        };
    }
    if let Some(file_name) = &manifest.tarball_file_name {
        return InstallSource::Tarball {
            url: manifest.selected_artifact_url.clone(),
            file_name: file_name.clone(),
        };
    }
    manifest
        .index_url
        .as_ref()
        .map_or(InstallSource::Unknown, |url| InstallSource::Index {
            url: url.clone(),
        })
}

fn component_reports(
    examination: &rocm_core::Examination,
    runtimes: &[RuntimeRecord],
) -> Vec<ComponentReport> {
    let mut reports = vec![
        ComponentReport {
            kind: ComponentKind::App,
            name: "rocm-app".to_owned(),
            // The CLI cannot observe the desktop app's version; the consumer
            // fills this in. Saying so beats reporting a wrong version.
            state: ComponentState::Unknown {
                reason: "supplied by the desktop app, not the CLI".to_owned(),
            },
        },
        ComponentReport {
            kind: ComponentKind::Cli,
            name: "rocm".to_owned(),
            state: ComponentState::Installed {
                version: env!("CARGO_PKG_VERSION").to_owned(),
            },
        },
    ];

    reports.push(ComponentReport {
        kind: ComponentKind::SystemHipRocm,
        name: "system-rocm".to_owned(),
        state: if examination.rocm_version.is_empty() {
            ComponentState::NotInstalled
        } else {
            ComponentState::Installed {
                version: examination.rocm_version.clone(),
            }
        },
    });

    reports.push(ComponentReport {
        kind: ComponentKind::ManagedRuntime,
        name: "therock".to_owned(),
        state: runtimes.iter().find(|r| r.active).map_or_else(
            || {
                if runtimes.is_empty() {
                    ComponentState::NotInstalled
                } else {
                    ComponentState::Unknown {
                        reason: "runtimes are installed but none is active".to_owned(),
                    }
                }
            },
            |active| ComponentState::Installed {
                version: active.version.clone(),
            },
        ),
    });

    reports.push(ComponentReport {
        kind: ComponentKind::PyTorch,
        name: "torch".to_owned(),
        state: if examination.framework_version.is_empty() {
            ComponentState::Unknown {
                reason: "framework probe skipped for this snapshot".to_owned(),
            }
        } else {
            ComponentState::Installed {
                version: examination.framework_version.clone(),
            }
        },
    });

    reports
}

#[cfg(test)]
mod tests {
    use super::*;

    fn supported_platform() -> PlatformReport {
        PlatformReport {
            os: OsFamily::Linux,
            arch: "x86_64".to_owned(),
            is_wsl: false,
            support: SupportStatus::Supported,
        }
    }

    fn known_gpu() -> GpuIdentity {
        GpuIdentity {
            name: Some("AMD Radeon AI PRO R9700".to_owned()),
            gfx_target: Some("gfx1201".to_owned()),
            therock_family: Some("gfx120X-all".to_owned()),
        }
    }

    fn ready_runtime() -> RuntimeRecord {
        RuntimeRecord {
            key: "nightly-wheel-gfx120x-all-7-14-0".to_owned(),
            runtime_id: "therock-nightly:gfx120X-all".to_owned(),
            version: "7.14.0".to_owned(),
            active: true,
            previous: false,
            validation: RuntimeValidation::Ready,
            channel: "nightly".to_owned(),
            family: "gfx120X-all".to_owned(),
            format: "wheel".to_owned(),
            install_source: InstallSource::Index {
                url: "https://example.invalid/simple".to_owned(),
            },
            install_root: PathBuf::from("/tmp/rocm/runtime"),
            read_only: false,
        }
    }

    fn healthy_inputs() -> SnapshotInputs {
        SnapshotInputs {
            observed_at_unix_ms: 1_767_225_600_000,
            platform: supported_platform(),
            gpu: known_gpu(),
            runtimes: vec![ready_runtime()],
            components: vec![ComponentReport {
                kind: ComponentKind::Cli,
                name: "rocm".to_owned(),
                state: ComponentState::Installed {
                    version: "0.1.0".to_owned(),
                },
            }],
            driver: DriverReport {
                installed: DriverVersionState::DetectedWithoutVersion {
                    detail: "amdgpu kernel module is loaded".to_owned(),
                },
                latest_known: None,
                support_links: vec![],
            },
            update: UpdateReport {
                state: UpdateState::NoUpdate {
                    installed: "7.14.0".to_owned(),
                },
                checked_at_unix_ms: Some(1_767_225_600_000),
                trust: SourceTrust::Signed {
                    key_source: "pinned".to_owned(),
                },
            },
            probe_failures: vec![],
        }
    }

    #[test]
    fn app_contract_healthy_snapshot_has_no_reasons() {
        let snapshot = build_snapshot(healthy_inputs());
        assert_eq!(snapshot.health.verdict, HealthVerdict::Healthy);
        assert!(snapshot.health.reasons.is_empty());
        assert!(snapshot.health.next_action.is_none());
    }

    #[test]
    fn app_contract_envelope_is_always_identified() {
        let snapshot = build_snapshot(healthy_inputs());
        assert!(snapshot.schema_version > 0);
        assert_eq!(snapshot.schema_version, SCHEMA_VERSION);
        assert_eq!(snapshot.producer.name, "rocm-cli");
        assert!(!snapshot.producer.version.is_empty());
        assert!(!snapshot.producer.build.is_empty());
        assert!(snapshot.observed_at_unix_ms > 0);
    }

    /// The load-bearing platform gate. An unsupported host must receive an empty
    /// action list, not a filtered one — a client that trusts the list alone is
    /// then still correct.
    #[test]
    fn app_contract_wsl_is_unsupported_with_no_eligible_actions() {
        let mut inputs = healthy_inputs();
        inputs.platform.is_wsl = true;
        inputs.platform.support = SupportStatus::Unsupported {
            reason: ReasonCode::PlatformWsl,
        };
        let snapshot = build_snapshot(inputs);

        assert_eq!(snapshot.health.verdict, HealthVerdict::Unsupported);
        assert!(snapshot.eligible_actions.is_empty());
        assert_eq!(
            snapshot
                .health
                .reasons
                .iter()
                .map(|r| r.code)
                .collect::<Vec<_>>(),
            vec![ReasonCode::PlatformWsl]
        );
    }

    /// Even with runtimes installed and an update waiting, WSL is offered nothing.
    #[test]
    fn app_contract_wsl_suppresses_actions_that_would_otherwise_apply() {
        let mut inputs = healthy_inputs();
        inputs.platform.support = SupportStatus::Unsupported {
            reason: ReasonCode::PlatformWsl,
        };
        inputs.update.state = UpdateState::Available {
            installed: "7.14.0".to_owned(),
            latest: "7.15.0".to_owned(),
        };
        assert!(build_snapshot(inputs).eligible_actions.is_empty());
    }

    #[test]
    fn app_contract_missing_runtime_is_setup_required() {
        let mut inputs = healthy_inputs();
        inputs.runtimes.clear();
        inputs.update.state = UpdateState::NotApplicable;
        let snapshot = build_snapshot(inputs);

        assert_eq!(snapshot.health.verdict, HealthVerdict::SetupRequired);
        assert!(
            snapshot
                .health
                .reasons
                .iter()
                .any(|r| r.code == ReasonCode::RuntimeAbsent)
        );
        assert_eq!(
            snapshot.eligible_actions,
            vec![EligibleAction::InstallRuntime]
        );
        assert_eq!(snapshot.health.next_action.as_deref(), Some("Set up ROCm."));
    }

    #[test]
    fn app_contract_failed_validation_is_attention() {
        let mut inputs = healthy_inputs();
        inputs.runtimes[0].validation = RuntimeValidation::Failed {
            detail: "python launcher missing".to_owned(),
        };
        let snapshot = build_snapshot(inputs);

        assert_eq!(snapshot.health.verdict, HealthVerdict::Attention);
        assert!(
            snapshot
                .health
                .reasons
                .iter()
                .any(|r| r.code == ReasonCode::RuntimeValidationFailed)
        );
    }

    /// Severity, not check order, picks the headline. An update notice must not
    /// outrank a failed runtime.
    #[test]
    fn app_contract_verdict_takes_the_most_severe_reason() {
        let mut inputs = healthy_inputs();
        inputs.runtimes[0].validation = RuntimeValidation::Failed {
            detail: "broken".to_owned(),
        };
        inputs.update.state = UpdateState::Available {
            installed: "7.14.0".to_owned(),
            latest: "7.15.0".to_owned(),
        };
        let snapshot = build_snapshot(inputs);

        assert_eq!(snapshot.health.verdict, HealthVerdict::Attention);
        assert_eq!(
            snapshot.health.next_action.as_deref(),
            Some("Repair or reinstall the active ROCm runtime.")
        );
    }

    #[test]
    fn app_contract_incomplete_probe_is_unknown_not_healthy() {
        let mut inputs = healthy_inputs();
        inputs.probe_failures.push("amd-smi timed out".to_owned());
        let snapshot = build_snapshot(inputs);

        assert_eq!(snapshot.health.verdict, HealthVerdict::Unknown);
        assert!(
            snapshot
                .health
                .reasons
                .iter()
                .any(|r| r.code == ReasonCode::ProbeIncomplete)
        );
    }

    #[test]
    fn app_contract_update_available_adds_the_update_action() {
        let mut inputs = healthy_inputs();
        inputs.update.state = UpdateState::Available {
            installed: "7.14.0".to_owned(),
            latest: "7.15.0".to_owned(),
        };
        let snapshot = build_snapshot(inputs);
        assert!(
            snapshot
                .eligible_actions
                .contains(&EligibleAction::UpdateRuntime)
        );
    }

    #[test]
    fn app_contract_untrusted_metadata_never_claims_an_update() {
        let mut inputs = healthy_inputs();
        inputs.update.state = UpdateState::UntrustedMetadata {
            detail: "signature verification failed".to_owned(),
        };
        inputs.update.trust = SourceTrust::Untrusted {
            reason: "bad signature".to_owned(),
        };
        let snapshot = build_snapshot(inputs);

        assert!(
            !snapshot
                .eligible_actions
                .contains(&EligibleAction::UpdateRuntime)
        );
        assert_eq!(snapshot.health.verdict, HealthVerdict::Attention);
    }

    #[test]
    fn app_contract_read_only_runtime_cannot_be_removed() {
        let mut inputs = healthy_inputs();
        let mut adopted = ready_runtime();
        adopted.active = false;
        adopted.read_only = true;
        adopted.key = "adopted".to_owned();
        adopted.install_source = InstallSource::Adopted {
            path: PathBuf::from("/opt/existing"),
        };
        inputs.runtimes.push(adopted);
        let snapshot = build_snapshot(inputs);

        assert!(
            snapshot
                .eligible_actions
                .contains(&EligibleAction::ActivateRuntime)
        );
        assert!(
            !snapshot
                .eligible_actions
                .contains(&EligibleAction::RemoveRuntime)
        );
    }

    /// Component states must be distinguishable without string sniffing.
    #[test]
    fn app_contract_component_states_are_distinct_variants() {
        let states = [
            ComponentState::LatestCompatible {
                version: "1".to_owned(),
            },
            ComponentState::Installed {
                version: "1".to_owned(),
            },
            ComponentState::UpdateAvailable {
                installed: "1".to_owned(),
                latest: "2".to_owned(),
            },
            ComponentState::Unsupported {
                version: "1".to_owned(),
                reason: "too old".to_owned(),
            },
            ComponentState::NotInstalled,
            ComponentState::Stale {
                version: None,
                checked_at_unix_ms: 1,
            },
            ComponentState::Unknown {
                reason: "not probed".to_owned(),
            },
        ];
        let tags: BTreeSet<String> = states
            .iter()
            .map(|s| {
                serde_json::to_value(s).expect("serialize")["state"]
                    .as_str()
                    .expect("tagged")
                    .to_owned()
            })
            .collect();
        assert_eq!(tags.len(), states.len(), "every state needs its own tag");
        assert!(tags.contains("not-installed"));
        assert!(tags.contains("unknown"));
    }

    /// Driver data carries no mutation.
    ///
    /// Asserted as an exact key set rather than a substring scan: `installed`
    /// is a legitimate noun here (the installed *version*), so a naive search
    /// for "install" both false-positives on it and would still miss a field
    /// named `apply` or `plan`. Pinning the key set means any new driver field
    /// fails this test until someone justifies it.
    #[test]
    fn app_contract_driver_report_has_no_mutation_action() {
        let value = serde_json::to_value(DriverReport {
            installed: DriverVersionState::Known {
                version: "25.10.1".to_owned(),
            },
            latest_known: Some("25.20.0".to_owned()),
            support_links: vec![SupportLink {
                label: "release notes".to_owned(),
                url: "https://www.amd.com/en/support".to_owned(),
            }],
        })
        .expect("serialize");

        let keys: BTreeSet<&str> = value
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            BTreeSet::from(["installed", "latestKnown", "supportLinks"]),
            "driver report gained a field; it must not be an operation"
        );

        // Nothing anywhere in the payload may name an operation the app could invoke.
        let wire = value.to_string();
        for verb in [
            "installDriver",
            "updateDriver",
            "removeDriver",
            "repair",
            "reconcile",
            "apply",
            "execute",
            "command",
            "plan",
            "mutation",
        ] {
            assert!(
                !wire.contains(verb),
                "driver payload must not expose {verb}: {wire}"
            );
        }
    }

    /// No eligible action may target a driver, on any host, in any state.
    #[test]
    fn app_contract_no_eligible_action_targets_a_driver() {
        for action in [
            EligibleAction::InstallRuntime,
            EligibleAction::UpdateRuntime,
            EligibleAction::ActivateRuntime,
            EligibleAction::RemoveRuntime,
            EligibleAction::ValidateRuntime,
        ] {
            let wire = serde_json::to_string(&action).expect("serialize");
            assert!(!wire.contains("driver"), "{wire} targets a driver");
        }
    }

    #[test]
    fn app_contract_snapshot_round_trips_through_json() {
        let snapshot = build_snapshot(healthy_inputs());
        let json = serde_json::to_string_pretty(&snapshot).expect("serialize");
        let back: AppSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, snapshot);
    }

    /// Round-trip the variants the healthy scenario never reaches.
    ///
    /// Regression: timestamps were `u128`, which serializes fine but cannot be
    /// *deserialized* inside an internally-tagged enum — serde buffers those
    /// through `Content`, which has no u128. The healthy-only round-trip test
    /// missed it because no `Stale` variant was ever constructed, so the
    /// producer shipped a payload it could not read back.
    #[test]
    fn app_contract_tagged_variants_round_trip() {
        let mut inputs = healthy_inputs();
        inputs.components.push(ComponentReport {
            kind: ComponentKind::ManagedRuntime,
            name: "therock".to_owned(),
            state: ComponentState::Stale {
                version: Some("7.14.0".to_owned()),
                checked_at_unix_ms: 1_767_139_200_000,
            },
        });
        inputs.update.state = UpdateState::Stale {
            installed: "7.14.0".to_owned(),
            checked_at_unix_ms: 1_767_139_200_000,
        };
        inputs.update.checked_at_unix_ms = Some(1_767_139_200_000);
        inputs.runtimes[0].install_source = InstallSource::Tarball {
            url: "https://example.invalid/a.tar.gz".to_owned(),
            file_name: "a.tar.gz".to_owned(),
        };

        let snapshot = build_snapshot(inputs);
        let json = serde_json::to_string(&snapshot).expect("serialize");
        let back: AppSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, snapshot);
    }

    /// Regression: a real gfx1201 host with an active managed runtime reported
    /// `gpu-unrecognised-family` / `setup-required` because `Examination` gets
    /// its gfx target from `rocminfo`, which that machine does not have. KFD
    /// sysfs knew the answer, so the host summary must win.
    #[test]
    fn app_contract_gpu_identity_prefers_kfd_over_missing_rocminfo() {
        let host = rocm_core::HostGpuSummary {
            name: Some("AMD Radeon AI PRO R9700".to_owned()),
            gfx_target: Some("gfx1201".to_owned()),
            therock_family: Some("gfx120X-all".to_owned()),
        };
        let examination = rocm_core::Examination::default();
        let identity = gpu_identity(&host, &examination);

        assert_eq!(identity.gfx_target.as_deref(), Some("gfx1201"));
        assert_eq!(identity.therock_family.as_deref(), Some("gfx120X-all"));

        // With both sources blank the answer is honestly unknown, not invented.
        let blank = rocm_core::HostGpuSummary::default();
        let unknown = gpu_identity(&blank, &examination);
        assert!(unknown.gfx_target.is_none());
        assert!(unknown.therock_family.is_none());
    }

    #[test]
    fn app_contract_install_source_is_derived_from_provenance() {
        let base = InstalledRuntimeManifest {
            runtime_key: "k".to_owned(),
            runtime_id: "id".to_owned(),
            channel: "nightly".to_owned(),
            format: "wheel".to_owned(),
            family: "gfx120X-all".to_owned(),
            family_source: "host".to_owned(),
            version: "7.14.0".to_owned(),
            install_root: PathBuf::from("/tmp/r"),
            selected_artifact_url: "https://example.invalid/a".to_owned(),
            index_url: Some("https://example.invalid/simple".to_owned()),
            tarball_file_name: None,
            python_launcher: None,
            python_executable: None,
            pip_cache_dir: None,
            rocm_sdk: None,
            read_only: false,
            imported_from: None,
            installed_at_unix_ms: 1,
        };

        assert!(matches!(install_source(&base), InstallSource::Index { .. }));

        let mut tarball = base.clone();
        tarball.tarball_file_name = Some("rocm.tar.gz".to_owned());
        assert!(matches!(
            install_source(&tarball),
            InstallSource::Tarball { .. }
        ));

        let mut adopted = base.clone();
        adopted.imported_from = Some(PathBuf::from("/opt/existing"));
        adopted.read_only = true;
        assert!(matches!(
            install_source(&adopted),
            InstallSource::Adopted { .. }
        ));

        let mut imported = base.clone();
        imported.imported_from = Some(PathBuf::from("/opt/existing"));
        assert!(matches!(
            install_source(&imported),
            InstallSource::Imported { .. }
        ));

        let mut bare = base;
        bare.index_url = None;
        assert!(matches!(install_source(&bare), InstallSource::Unknown));
    }

    /// Every reason maps to exactly one next action, so no verdict can reach the
    /// UI without something for the user to do.
    #[test]
    fn app_contract_every_reason_yields_a_next_action() {
        for code in [
            ReasonCode::PlatformWsl,
            ReasonCode::PlatformUnsupportedOs,
            ReasonCode::GpuAbsent,
            ReasonCode::GpuUnrecognisedFamily,
            ReasonCode::RuntimeAbsent,
            ReasonCode::RuntimeValidationFailed,
            ReasonCode::RuntimeActiveMissing,
            ReasonCode::RuntimeAmbiguousSelection,
            ReasonCode::DriverNotDetected,
            ReasonCode::UpdateAvailable,
            ReasonCode::UpdateMetadataUntrusted,
            ReasonCode::UpdateOffline,
            ReasonCode::ProbeIncomplete,
        ] {
            let reasons = vec![HealthReason {
                code,
                detail: "test".to_owned(),
            }];
            let action = next_action_for(code.severity(), &reasons);
            assert!(
                action.is_some_and(|a| !a.is_empty()),
                "{code:?} has no next action"
            );
        }
    }

    /// Every key in the payload is camelCase, including fields inside tagged
    /// enum variants.
    ///
    /// A container's `rename_all` does **not** reach struct-variant fields, so
    /// `SourceTrust::Signed { key_source }` shipped as `key_source` beside
    /// `checkedAtUnixMs`. A consumer then needs per-variant casing rules to
    /// decode one object, which is exactly the kind of papercut that gets
    /// worked around in the client instead of fixed here.
    #[test]
    fn app_contract_every_key_is_camel_case() {
        fn walk(value: &serde_json::Value, path: &str, bad: &mut Vec<String>) {
            match value {
                serde_json::Value::Object(map) => {
                    for (key, child) in map {
                        if key.contains('_') {
                            bad.push(format!("{path}.{key}"));
                        }
                        walk(child, &format!("{path}.{key}"), bad);
                    }
                }
                serde_json::Value::Array(items) => {
                    for (i, child) in items.iter().enumerate() {
                        walk(child, &format!("{path}[{i}]"), bad);
                    }
                }
                _ => {}
            }
        }

        let mut bad = Vec::new();
        for (name, snapshot) in golden_scenarios() {
            let value = serde_json::to_value(&snapshot).expect("serialize");
            walk(&value, name, &mut bad);
        }
        // Cover the variants the golden scenarios do not reach.
        for extra in [
            serde_json::to_value(InstallSource::Tarball {
                url: "https://example.invalid/a.tar.gz".to_owned(),
                file_name: "a.tar.gz".to_owned(),
            }),
            serde_json::to_value(ComponentState::Stale {
                version: None,
                checked_at_unix_ms: 1,
            }),
            serde_json::to_value(UpdateState::Stale {
                installed: "1".to_owned(),
                checked_at_unix_ms: 1,
            }),
        ] {
            walk(&extra.expect("serialize"), "extra", &mut bad);
        }

        assert!(bad.is_empty(), "snake_case keys in the payload: {bad:?}");
    }

    /// The scenario set the consumer's golden fixtures are cut from.
    ///
    /// Kept beside the builder so a producer change that alters any payload
    /// shows up here first, and `emit_golden_fixtures` can regenerate the
    /// consumer's copies from the real producer rather than from hand-written
    /// JSON that only looks right.
    fn golden_scenarios() -> Vec<(&'static str, AppSnapshot)> {
        let healthy = build_snapshot(healthy_inputs());

        let setup_required = {
            let mut i = healthy_inputs();
            i.runtimes.clear();
            i.update.state = UpdateState::NotApplicable;
            build_snapshot(i)
        };

        let attention = {
            let mut i = healthy_inputs();
            i.runtimes[0].validation = RuntimeValidation::Failed {
                detail: "rocm_sdk import failed in the runtime's Python".to_owned(),
            };
            i.update.state = UpdateState::Available {
                installed: "7.14.0".to_owned(),
                latest: "7.15.0".to_owned(),
            };
            build_snapshot(i)
        };

        let unsupported_wsl = {
            let mut i = healthy_inputs();
            i.platform.is_wsl = true;
            i.platform.support = SupportStatus::Unsupported {
                reason: ReasonCode::PlatformWsl,
            };
            build_snapshot(i)
        };

        let offline_stale = {
            let mut i = healthy_inputs();
            i.update.state = UpdateState::Offline {
                detail: "update catalog is unreachable".to_owned(),
            };
            i.update.checked_at_unix_ms = None;
            i.update.trust = SourceTrust::Untrusted {
                reason: "no metadata retrieved".to_owned(),
            };
            i.components.push(ComponentReport {
                kind: ComponentKind::ManagedRuntime,
                name: "therock".to_owned(),
                state: ComponentState::Stale {
                    version: Some("7.14.0".to_owned()),
                    checked_at_unix_ms: 1_767_139_200_000,
                },
            });
            build_snapshot(i)
        };

        let partial = {
            let mut i = healthy_inputs();
            i.probe_failures.push("amd-smi did not respond".to_owned());
            i.components.push(ComponentReport {
                kind: ComponentKind::PyTorch,
                name: "torch".to_owned(),
                state: ComponentState::Unknown {
                    reason: "framework probe skipped".to_owned(),
                },
            });
            build_snapshot(i)
        };

        vec![
            ("healthy", healthy),
            ("setup-required", setup_required),
            ("attention", attention),
            ("unsupported-wsl", unsupported_wsl),
            ("offline-stale", offline_stale),
            ("partial", partial),
        ]
    }

    /// Regenerate the consumer's golden fixtures from this producer.
    ///
    /// Opt-in: set `ROCM_APP_GOLDEN_DIR` to the consumer's `fixtures/contract`
    /// directory. Ordinary runs assert the scenarios build and are distinct
    /// instead of writing anything, so the suite never depends on a sibling
    /// checkout being present.
    #[test]
    fn app_contract_emit_golden_fixtures() {
        let scenarios = golden_scenarios();
        assert_eq!(scenarios.len(), 6);

        let verdicts: BTreeSet<HealthVerdict> =
            scenarios.iter().map(|(_, s)| s.health.verdict).collect();
        assert!(
            verdicts.len() >= 4,
            "golden set must span the verdict space, saw {verdicts:?}"
        );

        let Some(dir) = std::env::var_os("ROCM_APP_GOLDEN_DIR") else {
            return;
        };
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("create golden dir");
        for (name, snapshot) in scenarios {
            let json = serde_json::to_string_pretty(&snapshot).expect("serialize");
            std::fs::write(dir.join(format!("{name}.json")), format!("{json}\n"))
                .expect("write golden");
        }
    }
}
