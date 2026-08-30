// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

mod config;
mod target;
mod transaction;

use anyhow::{Context, Result, bail};
use clap::Args;
use rocm_core::{AppPaths, interactive_terminal};
use semver::Version;
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Args, Debug)]
pub(super) struct AgentsArgs {
    /// Agent harness to inspect, configure, or test.
    #[arg(value_name = "AGENT")]
    agent_name: Option<String>,
    /// Configure the harness to use a local ROCm model server.
    #[arg(long)]
    setup: bool,
    /// Show the setup plan without writing any configuration.
    #[arg(
        long,
        requires = "setup",
        conflicts_with_all = ["yes", "no_check", "test"]
    )]
    dry_run: bool,
    /// Approve the displayed setup plan without prompting.
    #[arg(short = 'y', long, requires = "setup")]
    yes: bool,
    /// Keep the configuration without running the protocol compatibility check.
    #[arg(long, requires = "setup")]
    no_check: bool,
    /// Run an isolated test through the actual agent harness.
    #[arg(long)]
    test: bool,
    /// Exact model identifier to configure or use to select a managed service.
    #[arg(long, requires = "setup")]
    model: Option<String>,
    /// Explicit loopback HTTP server URL. By default, setup auto-detects the unique
    /// ready managed service, or selects a ready service whose model exactly matches
    /// --model. If neither resolves, it falls back to http://127.0.0.1:11435/v1.
    #[arg(long, requires = "setup")]
    base_url: Option<String>,
    /// Harness version whose supported configuration schema should be used.
    #[arg(long, value_name = "VERSION")]
    agent_version: Option<String>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(super) enum AgentHarness {
    Claude,
    Hermes,
    OpenClaw,
    Codex,
    OpenCode,
    QwenCode,
    Aider,
    Continue,
}

impl AgentHarness {
    pub(super) const fn all() -> &'static [Self] {
        &[
            Self::Claude,
            Self::Hermes,
            Self::OpenClaw,
            Self::Codex,
            Self::OpenCode,
            Self::QwenCode,
            Self::Aider,
            Self::Continue,
        ]
    }

    pub(super) fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "claude" | "claude-code" => Some(Self::Claude),
            "hermes" | "hermes-agent" => Some(Self::Hermes),
            "openclaw" | "open-claw" => Some(Self::OpenClaw),
            "codex" | "codex-cli" => Some(Self::Codex),
            "opencode" | "open-code" => Some(Self::OpenCode),
            "qwen-code" | "qwen" | "qwencode" => Some(Self::QwenCode),
            "aider" | "aider-chat" => Some(Self::Aider),
            "continue" | "continue-dev" | "cn" => Some(Self::Continue),
            _ => None,
        }
    }

    pub(super) const fn canonical_name(self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Hermes => "hermes",
            Self::OpenClaw => "openclaw",
            Self::Codex => "codex",
            Self::OpenCode => "opencode",
            Self::QwenCode => "qwen-code",
            Self::Aider => "aider",
            Self::Continue => "continue",
        }
    }

    pub(super) const fn aliases(self) -> &'static [&'static str] {
        match self {
            Self::Claude => &["claude-code"],
            Self::Hermes => &["hermes-agent"],
            Self::OpenClaw => &["open-claw"],
            Self::Codex => &["codex-cli"],
            Self::OpenCode => &["open-code"],
            Self::QwenCode => &["qwen", "qwencode"],
            Self::Aider => &["aider-chat"],
            Self::Continue => &["continue-dev", "cn"],
        }
    }

    pub(super) const fn executable(self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Hermes => "hermes",
            Self::OpenClaw => "openclaw",
            Self::Codex => "codex",
            Self::OpenCode => "opencode",
            Self::QwenCode => "qwen",
            Self::Aider => "aider",
            Self::Continue => "cn",
        }
    }

    pub(super) const fn protocol(self) -> AgentProtocol {
        match self {
            Self::Claude => AgentProtocol::AnthropicMessages,
            Self::Codex => AgentProtocol::Responses,
            Self::Hermes
            | Self::OpenClaw
            | Self::OpenCode
            | Self::QwenCode
            | Self::Aider
            | Self::Continue => AgentProtocol::ChatCompletions,
        }
    }

    pub(super) const fn supports_version(self, version: &Version) -> bool {
        match self {
            Self::Claude => version.major == 1 || version.major == 2,
            Self::Hermes | Self::Codex | Self::QwenCode | Self::Aider => version.major == 0,
            Self::OpenClaw => version.major == 2026,
            Self::OpenCode | Self::Continue => version.major == 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(super) enum AgentProtocol {
    ChatCompletions,
    Responses,
    AnthropicMessages,
}

#[derive(Debug, Clone)]
pub(super) struct VersionInfo {
    pub(super) executable: Option<PathBuf>,
    pub(super) version: Option<Version>,
    pub(super) source: &'static str,
    pub(super) supported: bool,
}

#[derive(Debug, Clone)]
pub(super) struct ResolvedTarget {
    pub(super) api_base: String,
    pub(super) origin: String,
    pub(super) model: String,
    pub(super) managed_engine: Option<String>,
    pub(super) service_id: Option<String>,
}

pub(super) fn run(args: AgentsArgs) -> Result<()> {
    let has_action = args.setup
        || args.dry_run
        || args.yes
        || args.no_check
        || args.test
        || args.model.is_some()
        || args.base_url.is_some()
        || args.agent_version.is_some();
    let Some(agent_name) = args.agent_name.as_deref() else {
        if has_action {
            bail!("an agent name is required for setup, test, or version selection");
        }
        return list();
    };
    let harness = AgentHarness::parse(agent_name).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown agent '{agent_name}'; valid agents and aliases: claude (claude-code), hermes (hermes-agent), openclaw (open-claw), codex (codex-cli), opencode (open-code), qwen-code (qwen, qwencode), aider (aider-chat), continue (continue-dev, cn)"
        )
    })?;

    if args.setup && !args.dry_run && !args.yes && !interactive_terminal() {
        bail!("setup requires --yes when input is not interactive");
    }

    let version = target::detect_version(harness, args.agent_version.as_deref())
        .with_context(|| format!("failed to detect {} version", harness.canonical_name()))?;

    if args.setup {
        if !setup(harness, &version, &args)? {
            return Ok(());
        }
    } else if !args.test {
        inspect(harness, &version)?;
    }

    if args.test {
        require_testable_version(harness, &version)?;
        target::run_harness_test(harness, &version)
            .with_context(|| format!("{} harness test failed", harness.canonical_name()))?;
        println!("harness test passed");
    }
    Ok(())
}

fn list() -> Result<()> {
    println!("supported agent harnesses");
    for &harness in AgentHarness::all() {
        let version = target::detect_version(harness, None)
            .with_context(|| format!("failed to detect {} version", harness.canonical_name()))?;
        let state = config::inspect(harness).with_context(|| {
            format!(
                "failed to inspect {} configuration",
                harness.canonical_name()
            )
        })?;
        println!("  {}", harness.canonical_name());
        println!("    executable: {}", harness.executable());
        println!("    installed: {}", version.executable.is_some());
        match &version.version {
            Some(value) => println!("    version: {value}"),
            None => println!("    version: unavailable"),
        }
        println!("    configured: {}", state.configured);
    }
    Ok(())
}

fn inspect(harness: AgentHarness, version: &VersionInfo) -> Result<()> {
    let state = config::inspect(harness).with_context(|| {
        format!(
            "failed to inspect {} configuration",
            harness.canonical_name()
        )
    })?;
    println!("agent harness");
    println!("  agent: {}", harness.canonical_name());
    print!("  aliases:");
    for alias in harness.aliases() {
        print!(" {alias}");
    }
    println!();
    println!("  executable: {}", harness.executable());
    println!("  installed: {}", version.executable.is_some());
    match &version.version {
        Some(value) => println!("  version: {value}"),
        None => println!("  version: unavailable"),
    }
    println!("  version source: {}", version.source);
    println!("  supported: {}", version.supported);
    println!("  config: {}", state.path.display());
    println!("  configured: {}", state.configured);
    println!(
        "  endpoint: {}",
        state.endpoint.as_deref().unwrap_or("not configured")
    );
    println!(
        "  model: {}",
        state.model.as_deref().unwrap_or("not configured")
    );
    for warning in state.warnings {
        println!("  warning: {warning}");
    }
    Ok(())
}

fn setup(harness: AgentHarness, version: &VersionInfo, args: &AgentsArgs) -> Result<bool> {
    require_setup_version(harness, version)?;
    let paths = AppPaths::discover()?;
    let target = target::resolve_target(&paths, args.model.as_deref(), args.base_url.as_deref())
        .with_context(|| {
            format!(
                "failed to resolve a local target for {}",
                harness.canonical_name()
            )
        })?;
    let state = config::inspect(harness).with_context(|| {
        format!(
            "failed to inspect {} configuration",
            harness.canonical_name()
        )
    })?;
    let plan = config::plan(harness, version, &target, state)
        .with_context(|| format!("failed to plan {} setup", harness.canonical_name()))?;
    render_plan(harness, version, &target, &plan);

    if args.dry_run {
        println!("dry run: no changes written");
        return Ok(true);
    }
    if plan.changes.is_empty() {
        println!("configuration already correct");
        if !args.no_check {
            target::check_target(harness, &target)
                .with_context(|| format!("{} protocol check failed", harness.canonical_name()))?;
            println!("protocol check passed");
        }
        return Ok(true);
    }
    if !args.yes && !confirm()? {
        println!("setup cancelled");
        return Ok(false);
    }

    let applied = config::apply(&plan)
        .with_context(|| format!("failed to configure {}", harness.canonical_name()))?;
    if !args.no_check {
        if let Err(check_error) = target::check_target(harness, &target) {
            return match config::rollback(&applied) {
                Ok(()) => Err(check_error).with_context(|| {
                    format!(
                        "{} protocol check failed; restored original configuration",
                        harness.canonical_name()
                    )
                }),
                Err(rollback_error) => Err(anyhow::anyhow!(
                    "{} protocol check failed ({check_error}); restoring the original configuration also failed: {rollback_error}",
                    harness.canonical_name()
                )),
            };
        }
        println!("protocol check passed");
    }
    if applied.changed {
        println!("configured {}", harness.canonical_name());
    } else {
        println!("configuration already correct");
    }
    Ok(true)
}

fn render_plan(
    harness: AgentHarness,
    version: &VersionInfo,
    target: &ResolvedTarget,
    plan: &config::ConfigPlan,
) {
    println!("agent setup plan");
    println!("  agent: {}", harness.canonical_name());
    match &version.version {
        Some(value) => println!("  version: {value}"),
        None => println!("  version: unavailable"),
    }
    println!("  version source: {}", version.source);
    println!("  config: {}", plan.state.path.display());
    println!("  endpoint: {}", target.api_base);
    println!("  model: {}", target.model);
    if let Some(service_id) = &target.service_id {
        println!("  managed service: {service_id}");
    }
    if let Some(engine) = &target.managed_engine {
        println!("  managed engine: {engine}");
    }
    println!("  changes:");
    if plan.changes.is_empty() {
        println!("    none");
    } else {
        for change in &plan.changes {
            println!(
                "    {}: {} -> {}",
                change.setting,
                change.old_value.as_deref().unwrap_or("<unset>"),
                change.new_value
            );
        }
    }
    for warning in &plan.state.warnings {
        println!("  warning: {warning}");
    }
}

fn require_setup_version(harness: AgentHarness, version: &VersionInfo) -> Result<()> {
    let Some(value) = &version.version else {
        if let Some(executable) = &version.executable {
            bail!(
                "could not determine the installed {} version from {}; pass --agent-version <version> to select a supported setup schema",
                harness.canonical_name(),
                executable.display()
            );
        }
        bail!(
            "{} is not installed; pass --agent-version <version> to select a supported setup schema",
            harness.canonical_name()
        );
    };
    if !version.supported {
        bail!(
            "{} version {value} is not supported for setup",
            harness.canonical_name()
        );
    }
    Ok(())
}

fn require_testable_version(harness: AgentHarness, version: &VersionInfo) -> Result<()> {
    let Some(executable) = &version.executable else {
        bail!(
            "{} is not installed; --test requires the {} executable",
            harness.canonical_name(),
            harness.executable()
        );
    };
    let Some(value) = &version.version else {
        bail!(
            "could not determine the installed {} version from {}",
            harness.canonical_name(),
            executable.display()
        );
    };
    if !version.supported {
        bail!(
            "{} version {value} is not supported for testing",
            harness.canonical_name()
        );
    }
    Ok(())
}

fn confirm() -> Result<bool> {
    print!("Apply these changes? [y/N] ");
    io::stdout()
        .flush()
        .context("failed to write approval prompt")?;
    let mut answer = String::new();
    io::stdin()
        .read_line(&mut answer)
        .context("failed to read setup approval")?;
    let answer = answer.trim();
    Ok(answer.eq_ignore_ascii_case("y") || answer.eq_ignore_ascii_case("yes"))
}
