// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Black-box steps for `rocm agents`. All fixtures are visible only through the
//! child process environment, filesystem, HTTP requests, and fake executables.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime};

use cucumber::{given, then, when};
use e2e_cucumber::mock_server::{
    MockServer, ServiceRecordOptions, write_service_record_named_with,
};

use crate::E2eWorld;

const MODEL: &str = "org/agent-model";
const SECOND_MODEL: &str = "org/second-model";
const DEFAULT_BASE_URL: &str = "http://127.0.0.1:11435/v1";
const SECRET: &str = "e2e-real-credential-must-not-leak";

#[derive(Clone, Copy)]
struct HarnessFixture {
    name: &'static str,
    executable: &'static str,
    version: &'static str,
}

const HARNESSES: [HarnessFixture; 8] = [
    HarnessFixture {
        name: "claude",
        executable: "claude",
        version: "Claude Code 2.1.0",
    },
    HarnessFixture {
        name: "hermes",
        executable: "hermes",
        version: "Hermes 0.1.0",
    },
    HarnessFixture {
        name: "openclaw",
        executable: "openclaw",
        version: "OpenClaw 2026.1.0",
    },
    HarnessFixture {
        name: "codex",
        executable: "codex",
        version: "codex-cli 0.1.0",
    },
    HarnessFixture {
        name: "opencode",
        executable: "opencode",
        version: "opencode 1.0.0",
    },
    HarnessFixture {
        name: "qwen-code",
        executable: "qwen",
        version: "qwen 0.1.0",
    },
    HarnessFixture {
        name: "aider",
        executable: "aider",
        version: "aider 0.1.0",
    },
    HarnessFixture {
        name: "continue",
        executable: "cn",
        version: "cn 1.0.0",
    },
];

const ALIASES: [(&str, &str); 10] = [
    ("claude-code", "claude"),
    ("hermes-agent", "hermes"),
    ("open-claw", "openclaw"),
    ("codex-cli", "codex"),
    ("open-code", "opencode"),
    ("qwen", "qwen-code"),
    ("qwencode", "qwen-code"),
    ("aider-chat", "aider"),
    ("continue-dev", "continue"),
    ("cn", "continue"),
];
#[derive(Debug, Clone)]
struct CliResult {
    stdout: String,
    stderr: String,
    rc: i32,
}

impl CliResult {
    fn output(&self) -> String {
        format!("{}\n{}", self.stdout, self.stderr)
    }
}

#[derive(Debug, Clone, Copy)]
enum FakeMode {
    Success,
    Timeout,
    Nonzero,
    MissingNonce,
}

#[derive(Debug)]
pub struct AgentsState {
    home: PathBuf,
    xdg_config: PathBuf,
    xdg_data: PathBuf,
    xdg_cache: PathBuf,
    runtime: PathBuf,
    bin: PathBuf,
    project: PathBuf,
    results: Vec<CliResult>,
    original_bytes: Option<Vec<u8>>,
    checkpoint_bytes: Option<Vec<u8>>,
    checkpoint_modified: Option<SystemTime>,
    tracked_path: Option<PathBuf>,
    project_bytes: Option<Vec<u8>>,
    protocol_count: Option<usize>,
    mocks: Vec<MockServer>,
    test_timeout_secs: u64,
}

impl AgentsState {
    pub fn environment(&self) -> Vec<(&'static str, OsString)> {
        vec![
            ("HOME", self.home.clone().into_os_string()),
            ("USERPROFILE", self.home.clone().into_os_string()),
            ("XDG_CONFIG_HOME", self.xdg_config.clone().into_os_string()),
            ("XDG_DATA_HOME", self.xdg_data.clone().into_os_string()),
            ("XDG_CACHE_HOME", self.xdg_cache.clone().into_os_string()),
            ("XDG_RUNTIME_DIR", self.runtime.clone().into_os_string()),
            ("APPDATA", self.xdg_config.clone().into_os_string()),
            ("LOCALAPPDATA", self.xdg_data.clone().into_os_string()),
            ("PATH", self.bin.clone().into_os_string()),
            ("PATHEXT", OsString::from(".COM;.EXE;.BAT;.CMD")),
            (
                "CLAUDE_CONFIG_DIR",
                self.home.join(".claude").into_os_string(),
            ),
            ("HERMES_HOME", self.home.join(".hermes").into_os_string()),
            (
                "OPENCLAW_CONFIG_PATH",
                self.home.join(".openclaw/openclaw.json").into_os_string(),
            ),
            ("CODEX_HOME", self.home.join(".codex").into_os_string()),
            (
                "OPENCODE_CONFIG",
                self.xdg_config
                    .join("opencode/opencode.json")
                    .into_os_string(),
            ),
            ("QWEN_HOME", self.home.join(".qwen").into_os_string()),
            (
                "CONTINUE_GLOBAL_DIR",
                self.home.join(".continue").into_os_string(),
            ),
            (
                "ROCM_CLI_AGENT_TEST_TIMEOUT_SECS",
                self.test_timeout_secs.to_string().into(),
            ),
        ]
    }
}

#[given("an isolated agents environment")]
async fn isolated_agents_environment(world: &mut E2eWorld) {
    initialize_agents_state(world, 1);
}

#[given("all supported fake agent harnesses are installed")]
async fn all_fake_harnesses(world: &mut E2eWorld) {
    for harness in HARNESSES {
        install_fake(world, harness.name, FakeMode::Success);
    }
}

#[given("a configured fake Aider harness")]
async fn configured_fake_aider(world: &mut E2eWorld) {
    install_fake(world, "aider", FakeMode::Success);
    plant_config(
        world,
        "aider",
        &format!("model: openai/{MODEL}\nopenai-api-base: {DEFAULT_BASE_URL}\ndark-mode: true\n"),
    );
    snapshot_config(world, "aider");
}

#[given("one ready managed agent service")]
async fn one_managed_service(world: &mut E2eWorld) {
    world.model_name = Some(MODEL.to_string());
    world.mock = Some(MockServer::start(MODEL).await);
    world.register_mock_service();
}

#[given("two ready managed agent services")]
async fn two_managed_services(world: &mut E2eWorld) {
    let first = MockServer::start(MODEL).await;
    let second = MockServer::start(SECOND_MODEL).await;
    let services = isolated_root(world).join("data/services");
    let options = ServiceRecordOptions {
        supervisor_pid: std::process::id(),
        engine_pid: Some(std::process::id()),
        ..ServiceRecordOptions::default()
    };
    write_service_record_named_with(&services, "agent-first", MODEL, first.port(), options);
    write_service_record_named_with(
        &services,
        "agent-second",
        SECOND_MODEL,
        second.port(),
        options,
    );
    state_mut(world).mocks.extend([first, second]);
}

#[given("an unmanaged agent endpoint advertising one model")]
async fn unmanaged_single_model(world: &mut E2eWorld) {
    world.mock = Some(MockServer::start(MODEL).await);
}

#[given("an unmanaged agent endpoint advertising several models")]
async fn unmanaged_multiple_models(world: &mut E2eWorld) {
    world.mock = Some(MockServer::start_with_models(&[MODEL, SECOND_MODEL]).await);
}

#[given("a representative Claude configuration")]
async fn representative_claude(world: &mut E2eWorld) {
    plant_config(
        world,
        "claude",
        "{\n  \"permissions\": {\"allow\": [\"Read\"]},\n  \"unrelated\": \"keep-claude\"\n}\n",
    );
    snapshot_config(world, "claude");
}

#[given("a Claude configuration containing a credential")]
async fn credential_claude(world: &mut E2eWorld) {
    plant_config(
        world,
        "claude",
        &format!(
            "{{\n  \"unrelated\": \"keep-claude\",\n  \"env\": {{\"ANTHROPIC_API_KEY\": \"{SECRET}\"}}\n}}\n"
        ),
    );
    snapshot_config(world, "claude");
}

#[given("representative global configurations for every harness")]
async fn representative_configs(world: &mut E2eWorld) {
    plant_config(world, "claude", "{\"unrelated\":\"keep-claude\"}\n");
    plant_config(
        world,
        "hermes",
        "# retained hermes comment\nunrelated: keep-hermes\n",
    );
    plant_config(world, "openclaw", "{\"unrelated\":\"keep-openclaw\"}\n");
    plant_config(
        world,
        "codex",
        "# retained codex comment\nunrelated = \"keep-codex\"\n",
    );
    plant_config(
        world,
        "opencode",
        "{\n  // retained comment\n  \"unrelated\": \"keep-opencode\"\n}\n",
    );
    plant_config(world, "qwen-code", "{\"unrelated\":\"keep-qwen-code\"}\n");
    plant_config(
        world,
        "aider",
        "# retained aider comment\nunrelated: keep-aider\n",
    );
    plant_config(
        world,
        "continue",
        "# retained continue comment\nname: User Config\nversion: 1.0.0\nschema: v1\nunrelated: keep-continue\nmodels: []\n",
    );
}

#[given("a symlinked Claude configuration")]
async fn symlinked_claude(world: &mut E2eWorld) {
    let config = config_path(world, "claude");
    let target = state(world).home.join("claude-settings-target.json");
    std::fs::create_dir_all(config.parent().expect("Claude config has parent"))
        .expect("failed to create Claude config directory");
    std::fs::write(&target, "{\"unrelated\":\"symlink-target\"}\n")
        .expect("failed to write symlink target");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&target, &config).expect("failed to create config symlink");
    state_mut(world).tracked_path = Some(target.clone());
    state_mut(world).original_bytes =
        Some(std::fs::read(target).expect("failed to snapshot target"));
}

#[given("a restricted Claude configuration")]
async fn restricted_claude(world: &mut E2eWorld) {
    representative_claude(world).await;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let path = config_path(world, "claude");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640))
            .expect("failed to restrict Claude config");
    }
}

#[given("an endpoint that only lists agent models")]
async fn models_only_endpoint(world: &mut E2eWorld) {
    world.mock = Some(MockServer::start_models_only(MODEL).await);
}

#[given("a supported fake Claude harness is installed")]
async fn supported_fake_claude(world: &mut E2eWorld) {
    install_fake(world, "claude", FakeMode::Success);
}

#[given("an unsupported fake Claude harness is installed")]
async fn unsupported_fake_claude(world: &mut E2eWorld) {
    install_fake_with_version(world, "claude", "Claude Code 9.0.0", FakeMode::Success);
}

#[given("a protocol-complete agent endpoint")]
async fn protocol_complete_endpoint(world: &mut E2eWorld) {
    world.mock = Some(MockServer::start(MODEL).await);
}

#[given("an Aider project override")]
async fn aider_project_override(world: &mut E2eWorld) {
    plant_config(world, "aider", "unrelated: keep-aider\n");
    let project_config = state(world).project.join(".aider.conf.yml");
    std::fs::write(&project_config, "model: project/model\n")
        .expect("failed to write project override");
    state_mut(world).tracked_path = Some(project_config.clone());
    state_mut(world).project_bytes =
        Some(std::fs::read(project_config).expect("failed to snapshot project override"));
}

#[given("an isolated environment with real Claude Code and Codex")]
async fn real_agents_environment(world: &mut E2eWorld) {
    initialize_agents_state(world, 120);
    for executable in ["claude", "codex"] {
        let real = find_ambient_executable(executable)
            .unwrap_or_else(|| panic!("E2E_INCLUDE_REAL_AGENTS=1 requires {executable} on PATH"));
        link_real_executable(&real, &state(world).bin.join(executable));
    }
    if let Some(node) = find_ambient_executable("node") {
        link_real_executable(&node, &state(world).bin.join("node"));
    }
}

#[when("the user reads agents help and lists supported agent harnesses")]
async fn help_and_list_harnesses(world: &mut E2eWorld) {
    run_agents(world, &["agents", "--help"]);
    run_agents(world, &["agents"]);
}

#[when("the user inspects the Aider harness")]
async fn inspect_aider(world: &mut E2eWorld) {
    run_agents(world, &["agents", "aider"]);
}

#[when("the user inspects every documented agent alias")]
async fn inspect_aliases(world: &mut E2eWorld) {
    for (alias, _) in ALIASES {
        run_agents(world, &["agents", alias]);
    }
}

#[when("the user names an unknown harness and requests setup without a harness")]
async fn invalid_agent_invocations(world: &mut E2eWorld) {
    run_agents(world, &["agents", "not-a-harness"]);
    run_agents(
        world,
        &["agents", "--setup", "--yes", "--no-check", "--model", MODEL],
    );
}

#[when("the user previews Aider setup without an explicit target")]
async fn preview_unique_managed(world: &mut E2eWorld) {
    run_agents(
        world,
        &[
            "agents",
            "aider",
            "--setup",
            "--dry-run",
            "--agent-version",
            "0.1.0",
        ],
    );
}

#[when("the user previews setup without and then with a model filter")]
async fn preview_ambiguous_then_filtered(world: &mut E2eWorld) {
    run_agents(
        world,
        &[
            "agents",
            "aider",
            "--setup",
            "--dry-run",
            "--agent-version",
            "0.1.0",
        ],
    );
    run_agents(
        world,
        &[
            "agents",
            "aider",
            "--setup",
            "--dry-run",
            "--model",
            SECOND_MODEL,
            "--agent-version",
            "0.1.0",
        ],
    );
}

#[when("the user previews setup with only that base URL")]
async fn preview_explicit_single(world: &mut E2eWorld) {
    let base = world.mock.as_ref().expect("no mock").base_url();
    run_agents(
        world,
        &[
            "agents",
            "aider",
            "--setup",
            "--dry-run",
            "--base-url",
            &base,
            "--agent-version",
            "0.1.0",
        ],
    );
}

#[when("the user previews setup without and then with an advertised model")]
async fn preview_explicit_multiple(world: &mut E2eWorld) {
    let base = world.mock.as_ref().expect("no mock").base_url();
    run_agents(
        world,
        &[
            "agents",
            "aider",
            "--setup",
            "--dry-run",
            "--base-url",
            &base,
            "--agent-version",
            "0.1.0",
        ],
    );
    run_agents(
        world,
        &[
            "agents",
            "aider",
            "--setup",
            "--dry-run",
            "--base-url",
            &base,
            "--model",
            SECOND_MODEL,
            "--agent-version",
            "0.1.0",
        ],
    );
}

#[when("the user applies offline Aider setup with only an explicit model")]
async fn apply_fallback(world: &mut E2eWorld) {
    run_offline_setup(world, "aider", None, Some("0.1.0"));
}

#[when("the user previews setup with no service or model")]
async fn preview_no_server(world: &mut E2eWorld) {
    run_agents(
        world,
        &[
            "agents",
            "aider",
            "--setup",
            "--dry-run",
            "--agent-version",
            "0.1.0",
        ],
    );
}

#[when("the user previews setup with invalid endpoint forms")]
async fn invalid_endpoints(world: &mut E2eWorld) {
    for url in [
        "https://127.0.0.1:11435/v1",
        "http://example.com:11435/v1",
        "http://user:pass@127.0.0.1:11435/v1",
        "http://127.0.0.1:11435/v1?key=value",
        "http://127.0.0.1:11435/v1#fragment",
        "http://127.0.0.1:11435/not-v1",
    ] {
        run_agents(
            world,
            &[
                "agents",
                "aider",
                "--setup",
                "--dry-run",
                "--base-url",
                url,
                "--model",
                MODEL,
                "--agent-version",
                "0.1.0",
            ],
        );
    }
}

#[when("the user previews and then attempts unapproved Claude setup")]
async fn dry_run_and_unapproved(world: &mut E2eWorld) {
    run_agents(
        world,
        &[
            "agents",
            "claude",
            "--setup",
            "--dry-run",
            "--base-url",
            DEFAULT_BASE_URL,
            "--model",
            MODEL,
            "--agent-version",
            "2.1.0",
        ],
    );
    run_agents(
        world,
        &[
            "agents",
            "claude",
            "--setup",
            "--no-check",
            "--base-url",
            DEFAULT_BASE_URL,
            "--model",
            MODEL,
            "--agent-version",
            "2.1.0",
        ],
    );
}

#[when("the user previews and applies the same Claude setup twice")]
async fn redaction_and_repeat(world: &mut E2eWorld) {
    run_agents(
        world,
        &[
            "agents",
            "claude",
            "--setup",
            "--dry-run",
            "--base-url",
            DEFAULT_BASE_URL,
            "--model",
            MODEL,
            "--agent-version",
            "2.1.0",
        ],
    );
    run_offline_setup(world, "claude", Some(DEFAULT_BASE_URL), Some("2.1.0"));
    let path = config_path(world, "claude");
    state_mut(world).checkpoint_bytes = Some(std::fs::read(&path).expect("failed to checkpoint"));
    state_mut(world).checkpoint_modified = Some(
        std::fs::metadata(&path)
            .and_then(|metadata| metadata.modified())
            .expect("failed to read modified time"),
    );
    run_offline_setup(world, "claude", Some(DEFAULT_BASE_URL), Some("2.1.0"));
}

#[when("the user applies offline setup to every supported harness")]
async fn setup_every_harness(world: &mut E2eWorld) {
    for harness in HARNESSES {
        run_offline_setup(world, harness.name, Some(DEFAULT_BASE_URL), None);
    }
}

#[when("the user attempts offline Claude setup")]
async fn attempt_claude_setup(world: &mut E2eWorld) {
    run_offline_setup(world, "claude", Some(DEFAULT_BASE_URL), Some("2.1.0"));
}

#[when("the Claude configuration changes at the approval prompt")]
async fn change_during_approval(world: &mut E2eWorld) {
    let args = [
        "agents",
        "claude",
        "--setup",
        "--no-check",
        "--base-url",
        DEFAULT_BASE_URL,
        "--model",
        MODEL,
        "--agent-version",
        "2.1.0",
    ];
    let mut session = crate::e2e::tui_driver::TuiSession::spawn(world, &args)
        .expect("failed to spawn interactive setup");
    session
        .wait_for_screen("agent setup plan", Duration::from_secs(10))
        .await
        .expect("setup plan was not shown");
    let path = config_path(world, "claude");
    std::fs::write(&path, "{\"concurrent\":\"keep-this-edit\"}\n")
        .expect("failed to make concurrent edit");
    session.send("y\r").expect("failed to approve setup");
    session
        .wait_for_screen("changed", Duration::from_secs(10))
        .await
        .expect("stale update was not reported");
    world.cli_output = Some(session.screen_text());
    drop(session);
}

#[when("the user applies offline Claude setup")]
async fn apply_claude_setup(world: &mut E2eWorld) {
    run_offline_setup(world, "claude", Some(DEFAULT_BASE_URL), Some("2.1.0"));
}

#[when("checked Claude setup is applied to the incompatible endpoint")]
async fn checked_claude_incompatible(world: &mut E2eWorld) {
    let base = world.mock.as_ref().expect("no mock").base_url();
    run_checked_setup(world, "claude", &base, Some("2.1.0"));
}

#[when("no-check Claude setup is applied to that endpoint")]
async fn no_check_claude(world: &mut E2eWorld) {
    let base = world.mock.as_ref().expect("no mock").base_url();
    run_agents(
        world,
        &[
            "agents",
            "claude",
            "--setup",
            "--yes",
            "--no-check",
            "--base-url",
            &base,
            "--agent-version",
            "2.1.0",
        ],
    );
}

#[when("the user previews Claude setup using detected version selection")]
async fn detected_version_plan(world: &mut E2eWorld) {
    run_agents(
        world,
        &[
            "agents",
            "claude",
            "--setup",
            "--dry-run",
            "--base-url",
            DEFAULT_BASE_URL,
            "--model",
            MODEL,
        ],
    );
}

#[when(
    "the user configures uninstalled harnesses by explicit version and retries in managed modes"
)]
async fn setup_uninstalled_and_refuse_managed(world: &mut E2eWorld) {
    run_agents(
        world,
        &[
            "agents",
            "claude",
            "--setup",
            "--dry-run",
            "--base-url",
            DEFAULT_BASE_URL,
            "--model",
            MODEL,
            "--agent-version",
            "1.9.0",
        ],
    );
    for (agent, version) in [("hermes", "0.1.0"), ("openclaw", "2026.1.0")] {
        run_offline_setup(world, agent, Some(DEFAULT_BASE_URL), Some(version));
    }
    for (agent, version, managed_mode) in [
        ("hermes", "0.1.0", "HERMES_MANAGED"),
        ("openclaw", "2026.1.0", "OPENCLAW_NIX_MODE"),
    ] {
        run_agents_with_env(
            world,
            &[
                "agents",
                agent,
                "--setup",
                "--yes",
                "--no-check",
                "--base-url",
                DEFAULT_BASE_URL,
                "--model",
                MODEL,
                "--agent-version",
                version,
            ],
            &[(managed_mode, "1")],
        );
    }
}

#[when("the user inspects and then previews setup for that harness")]
async fn inspect_then_unsupported_setup(world: &mut E2eWorld) {
    run_agents(world, &["agents", "claude"]);
    run_agents(
        world,
        &[
            "agents",
            "claude",
            "--setup",
            "--dry-run",
            "--base-url",
            DEFAULT_BASE_URL,
            "--model",
            MODEL,
        ],
    );
}

#[when("checked setup is applied to every Chat Completions harness")]
async fn setup_chat_harnesses(world: &mut E2eWorld) {
    let base = world.mock.as_ref().expect("no mock").base_url();
    for agent in [
        "hermes",
        "openclaw",
        "opencode",
        "qwen-code",
        "aider",
        "continue",
    ] {
        run_checked_setup(world, agent, &base, None);
    }
}

#[when("checked Codex setup is applied")]
async fn setup_codex_checked(world: &mut E2eWorld) {
    let base = world.mock.as_ref().expect("no mock").base_url();
    run_checked_setup(world, "codex", &base, Some("0.1.0"));
}

#[when("checked Claude setup is applied")]
async fn setup_claude_checked(world: &mut E2eWorld) {
    let base = world.mock.as_ref().expect("no mock").base_url();
    run_checked_setup(world, "claude", &base, Some("2.1.0"));
}

#[when("the user configures and tests every fake harness")]
async fn test_every_fake_harness(world: &mut E2eWorld) {
    std::fs::write(
        state(world).project.join("caller.txt"),
        "caller repository remains untouched\n",
    )
    .expect("failed to seed caller repository");
    for harness in HARNESSES {
        let capture = capture_path(world, harness.name);
        let _ = std::fs::remove_file(&capture);
        run_setup_test(world, harness.name, true, DEFAULT_BASE_URL, None);
    }
}

#[when("fake Claude tests time out exit nonzero and omit the nonce")]
async fn fake_test_failures(world: &mut E2eWorld) {
    for mode in [FakeMode::Timeout, FakeMode::Nonzero, FakeMode::MissingNonce] {
        install_fake(world, "claude", mode);
        run_setup_test(world, "claude", true, DEFAULT_BASE_URL, Some("2.1.0"));
    }
}

#[when("the user runs setup-test and then no-check-test")]
async fn setup_test_combinations(world: &mut E2eWorld) {
    let base = world.mock.as_ref().expect("no mock").base_url();
    run_setup_test(world, "claude", false, &base, None);
    let first_count = world
        .mock
        .as_ref()
        .expect("no mock")
        .protocol_requests()
        .len();
    state_mut(world).protocol_count = Some(first_count);
    world
        .mock
        .as_ref()
        .expect("no mock")
        .clear_protocol_requests();
    let _ = std::fs::remove_file(capture_path(world, "claude"));
    run_setup_test(world, "claude", true, &base, None);
}

#[when("the user applies offline Aider setup")]
async fn apply_aider_setup(world: &mut E2eWorld) {
    run_offline_setup(world, "aider", Some(DEFAULT_BASE_URL), Some("0.1.0"));
}

#[when("the user requests invalid agents flag combinations")]
async fn invalid_flag_combinations(world: &mut E2eWorld) {
    for args in [
        vec!["agents", "claude", "--dry-run"],
        vec!["agents", "claude", "--setup", "--dry-run", "--yes"],
        vec!["agents", "claude", "--yes"],
        vec!["agents", "claude", "--setup", "--dry-run", "--test"],
        vec!["agents", "claude", "--setup", "--dry-run", "--no-check"],
        vec!["agents", "claude", "--model", MODEL],
        vec!["agents", "claude", "--base-url", DEFAULT_BASE_URL],
    ] {
        run_agents(world, &args);
    }
}

#[when("real Claude Code and Codex are configured and tested through agents")]
async fn test_real_agents(world: &mut E2eWorld) {
    for agent in ["claude", "codex"] {
        run_agents(world, &["agents", agent, "--setup", "--yes", "--test"]);
    }
}

#[then("help describes managed-service detection and the ROCm fallback endpoint")]
async fn help_describes_target_selection(world: &mut E2eWorld) {
    let result = state(world).results.first().expect("no help result");
    assert_contains(&result.output(), "auto-detects");
    assert_contains(&result.output(), "managed service");
    assert_contains(&result.output(), "http://127.0.0.1:11435/v1");
}

#[then("every canonical harness and both installation states are listed")]
async fn every_harness_listed(world: &mut E2eWorld) {
    let result = last_success(world);
    assert_contains(&result.output(), "supported agent harnesses");
    for harness in HARNESSES {
        assert_contains(&result.output(), harness.name);
    }
    assert_contains(&result.output(), "installed: true");
    assert_contains(&result.output(), "installed: false");
    assert_contains(&result.output(), "version: 2.1.0");
    assert_contains(&result.output(), "version: unavailable");
}

#[then("the Aider harness status is shown without changing its configuration")]
async fn aider_status_visible(world: &mut E2eWorld) {
    let result = last_success(world);
    for field in [
        "agent harness",
        "agent:",
        "aider",
        "executable:",
        "installed:",
        "version:",
        "config:",
        "configured:",
        "endpoint:",
        "model:",
    ] {
        assert_contains(&result.output(), field);
    }
    assert_config_unchanged(world, "aider");
}

#[then("every alias reports its canonical harness")]
async fn aliases_are_canonical(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), ALIASES.len(), "one inspect result per alias");
    for (result, (_, canonical)) in results.iter().zip(ALIASES) {
        assert_eq!(result.rc, 0, "alias inspect failed: {}", result.output());
        assert_contains(&result.output(), &format!("agent: {canonical}"));
    }
}

#[then("both agent invocations fail with valid-name guidance")]
async fn invalid_agents_fail(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 2);
    assert!(
        results.iter().all(|result| result.rc != 0),
        "unexpected success: {results:#?}"
    );
    assert_contains(&results[0].output(), "unknown agent");
    for harness in HARNESSES {
        assert_contains(&results[0].output(), harness.name);
    }
    assert_contains(&results[1].output(), "an agent name is required");
}

#[then("the plan uses the unique managed endpoint and model")]
async fn unique_target_selected(world: &mut E2eWorld) {
    let result = last_success(world);
    assert_contains(&result.output(), "agent setup plan");
    assert_contains(&result.output(), MODEL);
    assert_contains(
        &result.output(),
        &world.mock.as_ref().expect("no mock").base_url(),
    );
    assert_contains(&result.output(), "dry run: no changes written");
}

#[then("ambiguity is refused and the exact matching service is selected")]
async fn ambiguity_then_selection(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 2);
    assert_ne!(results[0].rc, 0, "ambiguous target unexpectedly passed");
    assert_contains(
        &results[0].output(),
        "more than one ready loopback ROCm service matches",
    );
    assert_contains(&results[0].output(), &format!("agent-first ({MODEL} at "));
    assert_contains(
        &results[0].output(),
        &format!("agent-second ({SECOND_MODEL} at "),
    );
    assert_eq!(
        results[1].rc,
        0,
        "filtered target failed: {}",
        results[1].output()
    );
    assert_contains(&results[1].output(), &format!("model: {SECOND_MODEL}"));
    assert_contains(&results[1].output(), "managed service: agent-second");
    let _ = std::fs::remove_dir_all(isolated_root(world).join("data/services"));
}

#[then("the advertised model and normalized endpoint appear in the plan")]
async fn explicit_single_selected(world: &mut E2eWorld) {
    let result = last_success(world);
    assert_contains(&result.output(), MODEL);
    assert_contains(
        &result.output(),
        &world.mock.as_ref().expect("no mock").base_url(),
    );
}

#[then("multiple models are refused until the exact model is supplied")]
async fn explicit_multiple_requires_model(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 2);
    assert_ne!(results[0].rc, 0);
    assert_contains(
        &results[0].output(),
        "/v1/models advertised more than one model",
    );
    assert_eq!(
        results[1].rc,
        0,
        "explicit model failed: {}",
        results[1].output()
    );
    assert_contains(&results[1].output(), SECOND_MODEL);
}

#[then("the configuration uses the ROCm default loopback endpoint")]
async fn fallback_is_rocm_default(world: &mut E2eWorld) {
    last_success(world);
    let config = read_config(world, "aider");
    assert_contains(&config, DEFAULT_BASE_URL);
    assert_contains(&config, MODEL);
}

#[then("setup fails and tells the user to run rocm serve")]
async fn no_server_guidance(world: &mut E2eWorld) {
    let result = last_result(world);
    assert_ne!(result.rc, 0);
    assert_contains(&result.output(), "rocm serve");
}

#[then("every invalid endpoint is rejected before configuration is written")]
async fn all_invalid_endpoints_fail(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 6);
    assert!(
        results.iter().all(|result| result.rc != 0),
        "unsafe URL passed: {results:#?}"
    );
    assert!(
        results.iter().all(|result| {
            let output = result.output().to_ascii_lowercase();
            [
                "loopback",
                "http",
                "credential",
                "query",
                "fragment",
                "path",
                "invalid",
            ]
            .iter()
            .any(|word| output.contains(word))
        }),
        "URL errors were not actionable: {results:#?}"
    );
    assert!(
        !config_path(world, "aider").exists(),
        "invalid target wrote config"
    );
}

#[then("both commands leave the configuration unchanged and explain why")]
async fn dry_run_and_approval_safe(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].rc, 0, "dry run failed: {}", results[0].output());
    assert_contains(&results[0].output(), "dry run: no changes written");
    assert_ne!(results[1].rc, 0, "unapproved setup unexpectedly passed");
    assert_contains(&results[1].output(), "requires --yes");
    assert_config_unchanged(world, "claude");
}

#[then("the credential is redacted and the second setup is a filesystem no-op")]
async fn redacted_and_idempotent(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 3);
    assert!(
        !results[0].output().contains(SECRET),
        "credential leaked in plan"
    );
    assert_contains(&results[0].output(), "redact");
    assert_eq!(
        results[1].rc,
        0,
        "first setup failed: {}",
        results[1].output()
    );
    assert_eq!(
        results[2].rc,
        0,
        "repeat setup failed: {}",
        results[2].output()
    );
    assert_contains(&results[2].output(), "configuration already correct");
    let path = config_path(world, "claude");
    assert_eq!(
        std::fs::read(&path).expect("failed to read repeated config"),
        state(world)
            .checkpoint_bytes
            .clone()
            .expect("no checkpoint"),
        "idempotent setup changed bytes"
    );
    assert_eq!(
        std::fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .expect("failed to read repeated mtime"),
        state(world)
            .checkpoint_modified
            .expect("no mtime checkpoint"),
        "idempotent setup rewrote the file"
    );
}

#[then("every global config visibly selects the exact local model and keeps unrelated settings")]
async fn all_adapters_persist_safely(world: &mut E2eWorld) {
    for harness in HARNESSES {
        let config = read_config(world, harness.name);
        assert_contains(&config, MODEL);
        assert_contains(&config, "127.0.0.1:11435");
        assert_contains(&config, &format!("keep-{}", harness.name));
    }
    for (agent, comment) in [
        ("hermes", "# retained hermes comment"),
        ("codex", "# retained codex comment"),
        ("opencode", "// retained comment"),
        ("aider", "# retained aider comment"),
        ("continue", "# retained continue comment"),
    ] {
        let config = read_config(world, agent);
        assert!(
            config.lines().any(|line| line.trim() == comment),
            "{agent} config did not retain exact comment {comment:?}:\n{config}"
        );
    }
    assert!(
        state(world).results.iter().all(|result| result.rc == 0),
        "adapter setup failed: {:#?}",
        state(world).results
    );
}

#[then("the symlink target is unchanged and setup explains the refusal")]
async fn symlink_refused(world: &mut E2eWorld) {
    let result = last_result(world);
    assert_ne!(result.rc, 0);
    assert_contains(&result.output(), "symlink");
    let target = state(world).tracked_path.as_ref().expect("no target");
    assert_eq!(
        std::fs::read(target).expect("failed to read symlink target"),
        state(world)
            .original_bytes
            .clone()
            .expect("no original target")
    );
}

#[then("the stale plan is refused without losing the concurrent edit")]
async fn stale_plan_refused(world: &mut E2eWorld) {
    assert_contains(
        world.cli_output.as_deref().expect("no interactive output"),
        "changed",
    );
    assert_eq!(
        read_config(world, "claude"),
        "{\"concurrent\":\"keep-this-edit\"}\n"
    );
}

#[then("its permissions are preserved and the atomic replacement is complete")]
async fn permissions_and_atomicity(world: &mut E2eWorld) {
    last_success(world);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let path = config_path(world, "claude");
        let mode = std::fs::metadata(&path)
            .expect("failed to stat Claude config")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o640, "setup changed config permissions");
        let parent = path.parent().expect("config has parent");
        let leftovers: Vec<_> = std::fs::read_dir(parent)
            .expect("failed to inspect config directory")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|name| name.contains("tmp") || name.contains("rollback"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "atomic replacement debris: {leftovers:?}"
        );
    }
    assert_contains(&read_config(world, "claude"), MODEL);
}

#[then("setup fails and restores the original configuration")]
async fn failed_check_rolls_back(world: &mut E2eWorld) {
    let result = last_result(world);
    assert_ne!(result.rc, 0);
    assert_contains(&result.output(), "restored original configuration");
    assert_config_unchanged(world, "claude");
}

#[then("setup is retained without sending a protocol request")]
async fn no_check_retained(world: &mut E2eWorld) {
    last_success(world);
    assert_contains(&read_config(world, "claude"), MODEL);
    assert!(
        world
            .mock
            .as_ref()
            .expect("no mock")
            .protocol_requests()
            .is_empty(),
        "--no-check sent a protocol request"
    );
}

#[then("the setup plan reports the detected version source")]
async fn detected_version_reported(world: &mut E2eWorld) {
    let result = last_success(world);
    assert_contains(&result.output(), "version: 2.1.0");
    assert_contains(&result.output(), "version source: detected");
}

#[then("every override is visible direct setup succeeds and known managed modes are refused")]
async fn direct_setup_and_managed_refusal(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 5);
    assert_eq!(
        results[0].rc,
        0,
        "Claude override preview failed: {}",
        results[0].output()
    );
    assert_contains(&results[0].output(), "version: 1.9.0");
    assert_contains(&results[0].output(), "version source: override");
    for (result, agent) in results[1..3].iter().zip(["hermes", "openclaw"]) {
        assert_eq!(
            result.rc,
            0,
            "direct {agent} setup failed: {}",
            result.output()
        );
        assert_contains(&result.output(), "version source: override");
        let config = read_config(world, agent);
        assert_contains(&config, DEFAULT_BASE_URL);
        assert_contains(&config, MODEL);
    }
    for (result, managed_mode) in results[3..]
        .iter()
        .zip(["HERMES_MANAGED", "OPENCLAW_NIX_MODE"])
    {
        assert_ne!(result.rc, 0, "{managed_mode} setup unexpectedly passed");
        assert_contains(&result.output(), managed_mode);
        assert_contains(&result.output(), "managed/declarative");
    }
}

#[then("inspection succeeds but mutation is refused as unsupported")]
async fn unsupported_read_only(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 2);
    assert_eq!(
        results[0].rc,
        0,
        "inspection failed: {}",
        results[0].output()
    );
    assert_contains(&results[0].output(), "9.0.0");
    assert_ne!(results[1].rc, 0, "unsupported setup passed");
    assert_contains(&results[1].output(), "not supported for setup");
    assert!(!config_path(world, "claude").exists());
}

#[then("every check reaches v1 chat completions with the exact model")]
async fn chat_protocol_requests(world: &mut E2eWorld) {
    let requests = world.mock.as_ref().expect("no mock").protocol_requests();
    assert_eq!(
        requests.len(),
        6,
        "unexpected protocol requests: {requests:#?}"
    );
    for request in requests {
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(
            request
                .body
                .get("model")
                .and_then(serde_json::Value::as_str),
            Some(MODEL)
        );
    }
}

#[then("the check reaches v1 responses with the exact model")]
async fn responses_protocol_request(world: &mut E2eWorld) {
    assert_protocol_request(world, "/v1/responses");
}

#[then("the check reaches v1 messages with the exact model")]
async fn messages_protocol_request(world: &mut E2eWorld) {
    assert_protocol_request(world, "/v1/messages");
}

#[then(
    "every fake process proves safe arguments caller isolation nonce integrity and disposable workspace state"
)]
async fn fake_harness_evidence(world: &mut E2eWorld) {
    assert_eq!(state(world).results.len(), HARNESSES.len());
    for (result, harness) in state(world).results.iter().zip(HARNESSES) {
        assert_eq!(
            result.rc,
            0,
            "{} test failed: {}",
            harness.name,
            result.output()
        );
        assert_contains(&result.output(), "harness test passed");
        let capture = std::fs::read_to_string(capture_path(world, harness.name))
            .unwrap_or_else(|error| panic!("missing {} process capture: {error}", harness.name));
        assert_contains(&capture, "args=");
        assert_contains(&capture, "probe.txt");
        assert_contains(&capture, "workspace-state=created");
        assert!(
            capture
                .lines()
                .filter_map(|line| line.strip_prefix("cwd="))
                .any(|cwd| Path::new(cwd) != state(world).project.as_path()),
            "{} did not run in a temporary workspace: {capture}",
            harness.name
        );
        assert_safe_fake_args(harness.name, &capture);
    }
    assert_eq!(
        std::fs::read(state(world).project.join("caller.txt"))
            .expect("caller repository marker was removed"),
        b"caller repository remains untouched\n"
    );
    assert_eq!(
        std::fs::read_dir(&state(world).project)
            .expect("failed to inspect caller repository")
            .count(),
        1,
        "harness test wrote into the caller repository"
    );
}

#[then("each harness failure has its concise category")]
async fn fake_failure_categories(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 3);
    for result in results {
        assert_ne!(result.rc, 0, "failure fake passed: {}", result.output());
    }
    assert_contains(&results[0].output(), "harness test timed out");
    assert_contains(&results[1].output(), "harness exited with");
    assert_contains(&results[2].output(), "did not return the probe nonce");
}

#[then("the first checks the protocol and both invoke the isolated harness")]
async fn combinations_are_literal(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 2);
    assert!(
        results.iter().all(|result| result.rc == 0),
        "combination failed: {results:#?}"
    );
    assert!(state(world).protocol_count.expect("no first request count") > 0);
    assert!(
        world
            .mock
            .as_ref()
            .expect("no mock")
            .protocol_requests()
            .is_empty(),
        "--no-check still sent the generic protocol check"
    );
    let capture = std::fs::read_to_string(capture_path(world, "claude"))
        .expect("second harness invocation was not captured");
    assert_contains(&capture, "probe.txt");
    assert!(
        results
            .iter()
            .all(|result| result.output().contains("harness test passed"))
    );
}

#[then("global setup succeeds with an override warning and the project file is unchanged")]
async fn project_warning_without_mutation(world: &mut E2eWorld) {
    let result = last_success(world);
    assert_contains(&result.output(), "project configuration");
    assert_contains(
        &result.output(),
        "has higher precedence; the user-level configuration is the only file changed",
    );
    assert_contains(&read_config(world, "aider"), MODEL);
    let project = state(world).tracked_path.as_ref().expect("no project path");
    assert_eq!(
        std::fs::read(project).expect("failed to read project override"),
        state(world)
            .project_bytes
            .clone()
            .expect("no project snapshot")
    );
}

#[then("every invalid flag combination fails without creating a harness config")]
async fn invalid_flags_fail_before_writes(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 7);
    assert!(
        results.iter().all(|result| result.rc != 0),
        "invalid flag combination passed: {results:#?}"
    );
    assert!(!config_path(world, "claude").exists());
}

#[then("both real harnesses pass their protocol and isolated nonce checks")]
async fn real_agents_pass(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 2);
    for result in results {
        assert_eq!(result.rc, 0, "real agent failed: {}", result.output());
        assert_contains(&result.output(), "protocol check passed");
        assert_contains(&result.output(), "harness test passed");
    }
}

fn initialize_agents_state(world: &mut E2eWorld, test_timeout_secs: u64) {
    let root = isolated_root(world).join("agents");
    let home = root.join("home");
    let xdg_config = root.join("xdg/config");
    let xdg_data = root.join("xdg/data");
    let xdg_cache = root.join("xdg/cache");
    let runtime = root.join("xdg/runtime");
    let bin = root.join("bin");
    let project = root.join("project");
    for path in [
        &home,
        &xdg_config,
        &xdg_data,
        &xdg_cache,
        &runtime,
        &bin,
        &project,
    ] {
        std::fs::create_dir_all(path).expect("failed to create isolated agents directory");
    }
    world.agents = Some(AgentsState {
        home,
        xdg_config,
        xdg_data,
        xdg_cache,
        runtime,
        bin,
        project,
        results: Vec::new(),
        original_bytes: None,
        checkpoint_bytes: None,
        checkpoint_modified: None,
        tracked_path: None,
        project_bytes: None,
        protocol_count: None,
        mocks: Vec::new(),
        test_timeout_secs,
    });
}

fn state(world: &E2eWorld) -> &AgentsState {
    world
        .agents
        .as_ref()
        .expect("agents environment was not initialized")
}

fn state_mut(world: &mut E2eWorld) -> &mut AgentsState {
    world
        .agents
        .as_mut()
        .expect("agents environment was not initialized")
}

fn isolated_root(world: &E2eWorld) -> PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("no isolated E2E root")
        .path()
        .to_path_buf()
}

fn config_path(world: &E2eWorld, agent: &str) -> PathBuf {
    let state = state(world);
    match agent {
        "claude" => state.home.join(".claude/settings.json"),
        "hermes" => state.home.join(".hermes/config.yaml"),
        "openclaw" => state.home.join(".openclaw/openclaw.json"),
        "codex" => state.home.join(".codex/config.toml"),
        "opencode" => state.xdg_config.join("opencode/opencode.json"),
        "qwen-code" => state.home.join(".qwen/settings.json"),
        "aider" => state.home.join(".aider.conf.yml"),
        "continue" => state.home.join(".continue/config.yaml"),
        other => panic!("unknown test harness {other}"),
    }
}

fn plant_config(world: &E2eWorld, agent: &str, contents: &str) {
    let path = config_path(world, agent);
    std::fs::create_dir_all(path.parent().expect("config has parent"))
        .expect("failed to create config directory");
    std::fs::write(path, contents).expect("failed to write agent config");
}

fn read_config(world: &E2eWorld, agent: &str) -> String {
    let path = config_path(world, agent);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn snapshot_config(world: &mut E2eWorld, agent: &str) {
    let path = config_path(world, agent);
    state_mut(world).tracked_path = Some(path.clone());
    state_mut(world).original_bytes = Some(std::fs::read(path).expect("failed to snapshot config"));
}

fn assert_config_unchanged(world: &E2eWorld, agent: &str) {
    assert_eq!(
        std::fs::read(config_path(world, agent)).expect("failed to read config"),
        state(world)
            .original_bytes
            .clone()
            .expect("no config snapshot"),
        "agent configuration changed"
    );
}

fn run_agents(world: &mut E2eWorld, args: &[&str]) {
    run_agents_with_env(world, args, &[]);
}

fn run_agents_with_env(world: &mut E2eWorld, args: &[&str], environment: &[(&str, &str)]) {
    let binary = crate::rocm_binary();
    let project = state(world).project.clone();
    let mut command = Command::new(&binary);
    command.args(args).current_dir(project);
    world.isolate_cmd(&mut command);
    for key in [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENCODE_CONFIG_CONTENT",
        "HERMES_MANAGED",
        "OPENCLAW_NIX_MODE",
    ] {
        command.env_remove(key);
    }
    command.envs(environment.iter().copied());
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to run {binary}: {error}"));
    let result = CliResult {
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        rc: output.status.code().unwrap_or(-1),
    };
    crate::record_command(
        world.current_scenario.as_deref(),
        args,
        result.rc,
        &result.stdout,
    );
    world.cli_output = Some(result.stdout.clone());
    world.cli_stderr = Some(result.stderr.clone());
    world.cli_rc = Some(result.rc);
    state_mut(world).results.push(result);
}

fn run_offline_setup(
    world: &mut E2eWorld,
    agent: &str,
    base_url: Option<&str>,
    version: Option<&str>,
) {
    let mut args = vec![
        "agents",
        agent,
        "--setup",
        "--yes",
        "--no-check",
        "--model",
        MODEL,
    ];
    if let Some(base_url) = base_url {
        args.extend(["--base-url", base_url]);
    }
    if let Some(version) = version {
        args.extend(["--agent-version", version]);
    }
    run_agents(world, &args);
}

fn run_checked_setup(world: &mut E2eWorld, agent: &str, base_url: &str, version: Option<&str>) {
    let mut args = vec![
        "agents",
        agent,
        "--setup",
        "--yes",
        "--base-url",
        base_url,
        "--model",
        MODEL,
    ];
    if let Some(version) = version {
        args.extend(["--agent-version", version]);
    }
    run_agents(world, &args);
}

fn run_setup_test(
    world: &mut E2eWorld,
    agent: &str,
    no_check: bool,
    base_url: &str,
    version: Option<&str>,
) {
    let mut args = vec![
        "agents",
        agent,
        "--setup",
        "--yes",
        "--test",
        "--base-url",
        base_url,
        "--model",
        MODEL,
    ];
    if no_check {
        args.push("--no-check");
    }
    if let Some(version) = version {
        args.extend(["--agent-version", version]);
    }
    run_agents(world, &args);
}

fn last_result(world: &E2eWorld) -> &CliResult {
    state(world).results.last().expect("no CLI result")
}

fn last_success(world: &E2eWorld) -> &CliResult {
    let result = last_result(world);
    assert_eq!(result.rc, 0, "CLI failed: {}", result.output());
    result
}

fn assert_contains(haystack: &str, needle: &str) {
    assert!(
        haystack
            .to_ascii_lowercase()
            .contains(&needle.to_ascii_lowercase()),
        "expected output to contain {needle:?}\n--- output ---\n{haystack}\n--- end ---"
    );
}

fn assert_protocol_request(world: &E2eWorld, expected_path: &str) {
    let requests = world.mock.as_ref().expect("no mock").protocol_requests();
    assert_eq!(
        requests.len(),
        1,
        "unexpected protocol requests: {requests:#?}"
    );
    assert_eq!(requests[0].path, expected_path);
    assert_eq!(
        requests[0]
            .body
            .get("model")
            .and_then(serde_json::Value::as_str),
        Some(MODEL)
    );
}

fn harness(name: &str) -> HarnessFixture {
    HARNESSES
        .into_iter()
        .find(|harness| harness.name == name)
        .unwrap_or_else(|| panic!("unknown fake harness {name}"))
}

fn capture_path(world: &E2eWorld, agent: &str) -> PathBuf {
    state(world).home.join(format!("{agent}-process.log"))
}

fn install_fake(world: &E2eWorld, agent: &str, mode: FakeMode) {
    install_fake_with_version(world, agent, harness(agent).version, mode);
}

fn install_fake_with_version(world: &E2eWorld, agent: &str, version: &str, mode: FakeMode) {
    let harness = harness(agent);
    let executable = state(world).bin.join(harness.executable);
    let capture = capture_path(world, agent);
    let config = config_path(world, agent);
    let _ = std::fs::remove_file(&capture);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        let timeout = matches!(mode, FakeMode::Timeout);
        let nonzero = matches!(mode, FakeMode::Nonzero);
        let missing_nonce = matches!(mode, FakeMode::MissingNonce);
        let script = format!(
            "#!/bin/sh\n\
             if [ \"$1\" = \"--version\" ]; then printf '%s\\n' {version}; exit 0; fi\n\
             {{ printf 'cwd=%s\\nargs=' \"$PWD\"; for arg in \"$@\"; do printf '<%s>' \"$arg\"; done; printf '\\npermission=%s\\n' \"${{OPENCODE_PERMISSION-}}\"; }} >> {capture}\n\
             if [ {agent} = hermes ] && [ \"$1\" = config ] && [ \"$2\" = set ]; then\n\
               if [ \"$3\" = model.default ]; then printf '%s' \"$4\" > {hermes_model}; fi\n\
               if [ \"$3\" = model.base_url ]; then model=$(/bin/cat {hermes_model}); printf '\\nmodel:\\n  provider: custom\\n  default: %s\\n  base_url: %s\\n' \"$model\" \"$4\" >> {config}; fi\n\
               exit 0\n\
             fi\n\
             if [ {agent} = openclaw ] && [ \"$1\" = config ] && [ \"$2\" = patch ]; then\n\
               patch=$(/bin/cat); printf 'stdin=%s\\n' \"$patch\" >> {capture}; inner=${{patch#\\{{}}; inner=${{inner%\\}}}}; printf '{{\"unrelated\":\"keep-openclaw\",%s}}\\n' \"$inner\" > {config}; exit 0\n\
             fi\n\
             printf 'disposable session\\n' > harness-session.cache || exit 9\n\
             printf 'workspace-state=created\\n' >> {capture}\n\
             if [ {timeout} = true ]; then /bin/sleep 60; exit 0; fi\n\
             if [ {nonzero} = true ]; then printf 'fake harness failure\\n' >&2; exit 7; fi\n\
             if [ {missing_nonce} = true ]; then printf 'finished without probe\\n'; exit 0; fi\n\
             if [ -f probe.txt ]; then /bin/cat probe.txt; exit 0; fi\n\
             printf 'probe file missing\\n' >&2; exit 8\n",
            version = shell_quote(version),
            capture = shell_quote_path(&capture),
            config = shell_quote_path(&config),
            hermes_model = shell_quote_path(&state(world).home.join(".hermes/.e2e-model")),
            agent = shell_quote(agent),
            timeout = timeout,
            nonzero = nonzero,
            missing_nonce = missing_nonce,
        );
        std::fs::write(&executable, script).expect("failed to write fake harness");
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755))
            .expect("failed to make fake harness executable");
    }

    #[cfg(windows)]
    {
        let executable = executable.with_extension("cmd");
        let hermes_model = state(world).home.join(".hermes/.e2e-model");
        let openclaw_patch = state(world).bin.join("openclaw-patch.ps1");
        if agent == "openclaw" {
            std::fs::write(
                &openclaw_patch,
                "$patch = [Console]::In.ReadToEnd().Trim()\r\n\
                 Add-Content -LiteralPath $args[0] -Value (\"stdin=$patch\")\r\n\
                 $inner = $patch.Substring(1, $patch.Length - 2)\r\n\
                 [IO.File]::WriteAllText($args[1], '{\"unrelated\":\"keep-openclaw\",' + $inner + '}' + [Environment]::NewLine)\r\n",
            )
            .expect("failed to write OpenClaw patch helper");
        }
        let behavior = match mode {
            FakeMode::Success => {
                "if exist probe.txt (type probe.txt& exit /b 0)\r\necho probe file missing 1>&2\r\nexit /b 8"
            }
            FakeMode::Timeout => {
                "\"%SystemRoot%\\System32\\PING.EXE\" -n 61 127.0.0.1 >nul\r\nexit /b 0"
            }
            FakeMode::Nonzero => "exit /b 7",
            FakeMode::MissingNonce => "echo finished without probe\r\nexit /b 0",
        };
        std::fs::write(
            executable,
            format!(
                "@echo off\r\nsetlocal EnableDelayedExpansion\r\n\
                 if \"%1\"==\"--version\" (echo {version}& exit /b 0)\r\n\
                 echo cwd=%CD%>>\"{capture}\"\r\n\
                 echo args=%*>>\"{capture}\"\r\n\
                 echo permission=%OPENCODE_PERMISSION%>>\"{capture}\"\r\n\
                 if /I \"%~1 %~2 %~3\"==\"config set model.default\" (>\"{hermes_model}\" <nul set /p \"=%~4\"& exit /b 0)\r\n\
                 if /I \"%~1 %~2 %~3\"==\"config set model.base_url\" (set /p \"HERMES_MODEL=\"<\"{hermes_model}\"& (>>\"{config}\" echo.& >>\"{config}\" echo model:& >>\"{config}\" echo   provider: custom& >>\"{config}\" echo   default: !HERMES_MODEL!& >>\"{config}\" echo   base_url: %~4)& exit /b 0)\r\n\
                 if /I \"%~1 %~2\"==\"config patch\" (\"%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File \"{openclaw_patch}\" \"{capture}\" \"{config}\"& exit /b !ERRORLEVEL!)\r\n\
                 >harness-session.cache echo disposable session\r\n\
                 if errorlevel 1 exit /b 9\r\n\
                 echo workspace-state=created>>\"{capture}\"\r\n\
                 {behavior}\r\n",
                capture = capture.display(),
                config = config.display(),
                hermes_model = hermes_model.display(),
                openclaw_patch = openclaw_patch.display(),
            ),
        )
        .expect("failed to write fake harness command");
    }
}

#[cfg(unix)]
fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

#[cfg(unix)]
fn shell_quote_path(path: &Path) -> String {
    shell_quote(&path.to_string_lossy())
}

fn assert_safe_fake_args(agent: &str, capture: &str) {
    let required: &[&str] = match agent {
        "claude" => &[
            "--permission-mode",
            "dontAsk",
            "--tools",
            "Read",
            "--no-session-persistence",
        ],
        "hermes" => &["chat", "--oneshot", "--toolsets", "file", "--max-turns"],
        "openclaw" => &["agent", "exec", "--cwd", "--code-mode", "direct"],
        "codex" => &["exec", "--sandbox", "read-only", "--ephemeral"],
        "opencode" => &["run", "--title", "rocm-agent-probe"],
        "qwen-code" => &["--prompt", "--approval-mode", "plan"],
        "aider" => &["--read", "probe.txt", "--no-git", "--no-auto-commits"],
        "continue" => &["--exclude", "Bash", "--exclude", "Write"],
        _ => unreachable!(),
    };
    for flag in required {
        assert_contains(capture, flag);
    }
    if agent == "opencode" {
        assert_contains(capture, "read");
        assert_contains(capture, "deny");
    }
}

fn find_ambient_executable(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

fn link_real_executable(source: &Path, destination: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::write(
            destination,
            format!(
                "#!/bin/sh\nexec {} \"$@\"\n",
                shell_quote(&source.to_string_lossy())
            ),
        )
        .expect("failed to write real harness launcher");
        std::fs::set_permissions(destination, std::fs::Permissions::from_mode(0o755))
            .expect("failed to make real harness launcher executable");
    }
    #[cfg(windows)]
    std::fs::copy(source, destination).expect("failed to copy real harness");
}
