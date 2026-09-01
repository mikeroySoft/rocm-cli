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
const OMP_CONFIG_ORIGINAL: &str = "# retained omp config comment\nmodelRoles:\n  default: upstream/existing-model\n  commit: upstream/commit-model\n  title: upstream/title-model\nunrelatedConfig: keep-omp-config\n";
const OMP_CONFIG_DEFAULT: &str = "# retained omp config comment\nmodelRoles:\n  default: rocm-local/org/agent-model\n  commit: upstream/commit-model\n  title: upstream/title-model\nunrelatedConfig: keep-omp-config\n";
const INVALID_PI_OMP_ALIASES: [&str; 3] = ["pi-coding-agent", "oh-my-pi", "omp-agent"];

#[derive(Clone, Copy)]
struct HarnessFixture {
    name: &'static str,
    executable: &'static str,
    version: &'static str,
}

const HARNESSES: [HarnessFixture; 10] = [
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
    HarnessFixture {
        name: "pi",
        executable: "pi",
        version: "0.84.4",
    },
    HarnessFixture {
        name: "omp",
        executable: "omp",
        version: "omp/18.0.11",
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
    original_files: Vec<(PathBuf, Vec<u8>)>,
    checkpoint_files: Vec<(PathBuf, Vec<u8>, SystemTime)>,
    project_files: Vec<(PathBuf, Vec<u8>)>,
    interactive_outputs: Vec<String>,
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
                "PI_CODING_AGENT_DIR",
                self.home.join("pi-agent-root").into_os_string(),
            ),
            ("PI_CONFIG_DIR", OsString::from("omp-root")),
            ("PI_PROFILE", OsString::from("must-not-win")),
            ("OMP_PROFILE", OsString::from("e2e")),
            ("PI_CONFIG_FILES", OsString::new()),
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

#[given("supported fake Pi and OMP harnesses are installed")]
async fn supported_fake_pi_omp(world: &mut E2eWorld) {
    for agent in ["pi", "omp"] {
        install_fake(world, agent, FakeMode::Success);
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
    plant_pi_omp_configs(world);
}

#[given("representative Pi and OMP configurations")]
async fn representative_pi_omp(world: &mut E2eWorld) {
    plant_pi_omp_configs(world);
    snapshot_configs(world, "pi");
    snapshot_configs(world, "omp");
}

#[given("a legacy OMP models.json without a YAML registry")]
async fn legacy_omp_models_json(world: &mut E2eWorld) {
    let legacy = config_path(world, "omp").with_extension("json");
    std::fs::create_dir_all(legacy.parent().expect("legacy config has parent"))
        .expect("failed to create legacy OMP config directory");
    std::fs::write(&legacy, b"{\"legacy\":\"must-remain-for-OMP-migration\"}\n")
        .expect("failed to write legacy OMP model registry");
    snapshot_path(world, legacy);
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
    snapshot_path(world, target);
}

#[given("a symlinked Pi second configuration")]
async fn symlinked_pi_second_config(world: &mut E2eWorld) {
    plant_config(
        world,
        "pi",
        "{\n  // retained pi models comment\n  \"unrelatedModels\": \"keep-pi-models\"\n}\n",
    );
    let config = secondary_config_path(world, "pi").expect("Pi has a second config");
    let target = state(world).home.join("pi-settings-target.json");
    std::fs::create_dir_all(config.parent().expect("Pi config has parent"))
        .expect("failed to create Pi config directory");
    std::fs::write(
        &target,
        "{\n  // retained pi settings comment\n  \"unrelatedSettings\": \"keep-pi-settings-target\"\n}\n",
    )
    .expect("failed to write Pi symlink target");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&target, &config).expect("failed to create Pi config symlink");
    snapshot_path(world, config_path(world, "pi"));
    snapshot_path(world, target);
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

#[given("unsupported fake Pi and OMP harnesses are installed")]
async fn unsupported_fake_pi_omp(world: &mut E2eWorld) {
    install_fake_with_version(world, "pi", "0.84.5", FakeMode::Success);
    install_fake_with_version(world, "omp", "omp/19.0.0", FakeMode::Success);
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
    snapshot_project_path(world, project_config);
}

#[given("Pi and OMP project and overlay overrides")]
async fn pi_omp_overrides(world: &mut E2eWorld) {
    let pi_project = state(world).project.join(".pi/settings.json");
    let omp_project = state(world).project.join(".omp/config.yml");
    let omp_overlay = state(world).home.join("omp-overlay.yml");
    for path in [&pi_project, &omp_project, &omp_overlay] {
        std::fs::create_dir_all(path.parent().expect("override has parent"))
            .expect("failed to create override directory");
    }
    std::fs::write(&pi_project, "{\"defaultModel\":\"project/pi-model\"}\n")
        .expect("failed to write Pi project override");
    std::fs::write(&omp_project, "modelRoles:\n  default: project/omp-model\n")
        .expect("failed to write OMP project override");
    std::fs::write(&omp_overlay, "modelRoles:\n  default: overlay/omp-model\n")
        .expect("failed to write OMP overlay");
    for path in [pi_project, omp_project, omp_overlay] {
        snapshot_project_path(world, path);
    }
}

#[given("a Pi second configuration larger than the bounded writer")]
async fn oversized_pi_second_config(world: &mut E2eWorld) {
    plant_config(
        world,
        "pi",
        "{\n  // retained pi models comment\n  \"unrelatedModels\": \"keep-pi-models\"\n}\n",
    );
    let padding = "x".repeat(12 * 1024);
    plant_secondary_config(
        world,
        "pi",
        &format!(
            "{{\n  // retained pi settings comment\n  \"unrelatedSettings\": \"keep-pi-settings\",\n  \"padding\": \"{padding}\"\n}}\n"
        ),
    );
    snapshot_configs(world, "pi");
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

#[when("the user also inspects canonical Pi and OMP")]
#[when("the user inspects both alias-free harnesses")]
async fn inspect_pi_omp(world: &mut E2eWorld) {
    for agent in ["pi", "omp"] {
        run_agents(world, &["agents", agent]);
    }
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

#[when("the user names plausible Pi and OMP aliases")]
async fn invalid_pi_omp_aliases(world: &mut E2eWorld) {
    for name in INVALID_PI_OMP_ALIASES {
        run_agents(world, &["agents", name]);
    }
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

#[when("the user previews setup with no target and no managed server")]
async fn preview_no_managed_server(world: &mut E2eWorld) {
    let base_url = unreachable_loopback_url();
    world.endpoint = Some(base_url.clone());
    run_agents_with_env(
        world,
        &[
            "agents",
            "aider",
            "--setup",
            "--dry-run",
            "--agent-version",
            "0.1.0",
        ],
        &[("ROCM_CLI_AGENT_TARGET_FALLBACK_URL", &base_url)],
    );
}

#[when("the user previews setup against a local endpoint with no server")]
async fn preview_no_server(world: &mut E2eWorld) {
    let base_url = unreachable_loopback_url();
    world.endpoint = Some(base_url.clone());
    run_agents(
        world,
        &[
            "agents",
            "aider",
            "--setup",
            "--dry-run",
            "--base-url",
            &base_url,
            "--agent-version",
            "0.1.0",
        ],
    );
}

#[when("the user inspects OMP with a relative config directory and normalized profiles")]
async fn inspect_omp_profile_matrix(world: &mut E2eWorld) {
    let default_agent = state(world)
        .home
        .join("omp-default-agent")
        .to_string_lossy()
        .into_owned();
    for (omp_profile, pi_profile) in [
        ("omp-wins", "pi-loses"),
        ("", "pi-must-not-win"),
        ("default", "pi-must-not-win"),
        (" \t ", "pi-must-not-win"),
    ] {
        run_agents_with_env(
            world,
            &["agents", "omp"],
            &[
                ("PI_CONFIG_DIR", "relative-omp"),
                ("PI_CODING_AGENT_DIR", &default_agent),
                ("OMP_PROFILE", omp_profile),
                ("PI_PROFILE", pi_profile),
            ],
        );
    }
}

#[when("the user attempts offline OMP setup with the legacy registry")]
async fn attempt_legacy_omp_setup(world: &mut E2eWorld) {
    run_offline_setup(world, "omp", Some(DEFAULT_BASE_URL), Some("18.0.11"));
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
    checkpoint_configs(world, "claude");
    run_offline_setup(world, "claude", Some(DEFAULT_BASE_URL), Some("2.1.0"));
}

#[when("the user previews and applies the same Pi and OMP setup twice")]
async fn preview_and_repeat_pi_omp(world: &mut E2eWorld) {
    for (agent, version) in [("pi", "0.84.4"), ("omp", "18.0.11")] {
        run_setup_dry_run(world, agent, version);
    }
    assert_original_files_unchanged(world);
    for (agent, version) in [("pi", "0.84.4"), ("omp", "18.0.11")] {
        run_offline_setup(world, agent, Some(DEFAULT_BASE_URL), Some(version));
        checkpoint_configs(world, agent);
        run_offline_setup(world, agent, Some(DEFAULT_BASE_URL), Some(version));
    }
}

#[when("the user applies offline setup to every supported harness")]
async fn setup_every_harness(world: &mut E2eWorld) {
    for harness in HARNESSES {
        run_offline_setup(world, harness.name, Some(DEFAULT_BASE_URL), None);
    }
}

#[when("the user interactively registers OMP and declines the default")]
async fn register_omp_and_decline_default(world: &mut E2eWorld) {
    run_interactive_omp_setup(world, false, true).await;
}

#[when("the user interactively registers OMP and accepts the default")]
async fn register_omp_and_accept_default(world: &mut E2eWorld) {
    run_interactive_omp_setup(world, true, false).await;
}

#[when("the user attempts offline Claude setup")]
async fn attempt_claude_setup(world: &mut E2eWorld) {
    run_offline_setup(world, "claude", Some(DEFAULT_BASE_URL), Some("2.1.0"));
}

#[when("the user attempts offline Pi setup")]
async fn attempt_pi_setup(world: &mut E2eWorld) {
    run_offline_setup(world, "pi", Some(DEFAULT_BASE_URL), Some("0.84.4"));
}

#[when("the Claude configuration changes at the approval prompt")]
async fn change_during_approval(world: &mut E2eWorld) {
    let path = config_path(world, "claude");
    change_file_during_approval(
        world,
        "claude",
        "2.1.0",
        &path,
        "{\"concurrent\":\"keep-this-edit\"}\n",
    )
    .await;
}

#[when("the OMP model registry changes at the approval prompt")]
async fn change_omp_registry_during_approval(world: &mut E2eWorld) {
    let path = config_path(world, "omp");
    change_file_during_approval(
        world,
        "omp",
        "18.0.11",
        &path,
        "# concurrent OMP registry edit\nunrelatedModels: keep-this-edit\n",
    )
    .await;
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

#[when("checked Pi and OMP setup is applied to the incompatible endpoint")]
async fn checked_pi_omp_incompatible(world: &mut E2eWorld) {
    let base = world.mock.as_ref().expect("no mock").base_url();
    for (agent, version) in [("pi", "0.84.4"), ("omp", "18.0.11")] {
        run_checked_setup(world, agent, &base, Some(version));
    }
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

#[when("the user previews Pi and OMP setup using detected version selection")]
async fn detected_pi_omp_version_plans(world: &mut E2eWorld) {
    for agent in ["pi", "omp"] {
        run_agents(
            world,
            &[
                "agents",
                agent,
                "--setup",
                "--dry-run",
                "--base-url",
                DEFAULT_BASE_URL,
                "--model",
                MODEL,
            ],
        );
    }
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
    for (agent, version) in [
        ("hermes", "0.1.0"),
        ("openclaw", "2026.1.0"),
        ("pi", "0.84.4"),
        ("omp", "18.0.11"),
    ] {
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

#[when("the user inspects and previews unsupported Pi and OMP harnesses")]
async fn inspect_unsupported_pi_omp(world: &mut E2eWorld) {
    for agent in ["pi", "omp"] {
        run_agents(world, &["agents", agent]);
        run_agents(
            world,
            &[
                "agents",
                agent,
                "--setup",
                "--dry-run",
                "--base-url",
                DEFAULT_BASE_URL,
                "--model",
                MODEL,
            ],
        );
    }
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
        "pi",
        "omp",
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

#[when("the user applies offline Pi and OMP setup with project overlays present")]
async fn apply_pi_omp_with_overrides(world: &mut E2eWorld) {
    run_offline_setup(world, "pi", Some(DEFAULT_BASE_URL), Some("0.84.4"));
    let overlay = state(world).home.join("omp-overlay.yml");
    let overlay = overlay.to_string_lossy();
    run_agents_with_env(
        world,
        &[
            "agents",
            "omp",
            "--setup",
            "--yes",
            "--no-check",
            "--base-url",
            DEFAULT_BASE_URL,
            "--model",
            MODEL,
            "--agent-version",
            "18.0.11",
        ],
        &[("PI_CONFIG_FILES", overlay.as_ref())],
    );
}

#[when("Pi setup hits a bounded second-file write failure")]
async fn pi_partial_write_failure(world: &mut E2eWorld) {
    run_agents_with_file_limit(
        world,
        &[
            "agents",
            "pi",
            "--setup",
            "--yes",
            "--no-check",
            "--base-url",
            DEFAULT_BASE_URL,
            "--model",
            MODEL,
            "--agent-version",
            "0.84.4",
        ],
    );
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
    let result = &state(world).results[0];
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

#[then("canonical Pi and OMP status reports their distinct executables and versions")]
async fn pi_omp_status_visible(world: &mut E2eWorld) {
    for (index, agent, executable, version) in
        [(1, "pi", "pi", "0.84.4"), (2, "omp", "omp", "18.0.11")]
    {
        let result = &state(world).results[index];
        assert_eq!(result.rc, 0, "{agent} inspect failed: {}", result.output());
        let output = result.output();
        assert_contains(&output, &format!("agent: {agent}"));
        assert_contains(&output, &format!("executable: {executable}"));
        assert_contains(&output, &format!("version: {version}"));
        for path in config_paths(world, agent) {
            assert_contains(&output, &path.to_string_lossy());
        }
    }
}

#[then("every alias reports its canonical harness")]
async fn aliases_are_canonical(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(
        results.len(),
        ALIASES.len() + 2,
        "one inspect result per alias and alias-free harness"
    );
    for (result, (_, canonical)) in results.iter().take(ALIASES.len()).zip(ALIASES) {
        assert_eq!(result.rc, 0, "alias inspect failed: {}", result.output());
        assert_contains(&result.output(), &format!("agent: {canonical}"));
    }
}

#[then("Pi and OMP report no aliases")]
async fn pi_omp_have_no_aliases(world: &mut E2eWorld) {
    for result in &state(world).results[ALIASES.len()..] {
        assert_eq!(
            result.rc,
            0,
            "canonical inspect failed: {}",
            result.output()
        );
        let output = result.output();
        let aliases = output
            .lines()
            .find(|line| line.trim_start().starts_with("aliases:"))
            .expect("inspect omitted aliases field");
        assert_eq!(aliases.trim(), "aliases:", "unexpected alias: {aliases}");
    }
}

#[then("both agent invocations fail with valid-name guidance")]
async fn invalid_agents_fail(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 5);
    assert!(
        results[..2].iter().all(|result| result.rc != 0),
        "unexpected success: {results:#?}"
    );
    assert_contains(&results[0].output(), "unknown agent");
    for harness in HARNESSES {
        assert_contains(&results[0].output(), harness.name);
    }
    assert_contains(&results[1].output(), "an agent name is required");
}

#[then("the alias-free Pi and OMP names are rejected with canonical guidance")]
async fn invalid_pi_omp_aliases_fail(world: &mut E2eWorld) {
    for (result, invented) in state(world).results[2..].iter().zip(INVALID_PI_OMP_ALIASES) {
        assert_ne!(result.rc, 0, "invented alias unexpectedly passed");
        let output = result.output();
        assert_contains(&output, "unknown agent");
        let guidance = output
            .split_once("valid agents and aliases:")
            .map(|(_, guidance)| guidance)
            .expect("unknown-agent error omitted canonical guidance");
        assert_contains(guidance, "(continue-dev, cn), pi, omp");
        assert!(
            !guidance.contains(invented),
            "invented name {invented:?} was presented as an alias: {guidance}"
        );
    }
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

#[then("setup fails with deterministic rocm serve guidance")]
async fn no_managed_server_gives_serve_guidance(world: &mut E2eWorld) {
    let result = last_result(world);
    let endpoint = world.endpoint.as_deref().expect("no fallback endpoint");
    assert_ne!(result.rc, 0, "missing target unexpectedly passed");
    assert_contains(
        &result.output(),
        &format!("no ready ROCm model server was found at {endpoint}"),
    );
    assert_contains(&result.output(), "run `rocm serve <model>`");
    assert!(
        !config_path(world, "aider").exists(),
        "missing target wrote configuration"
    );
}

#[then("setup fails and identifies the unreachable local endpoint")]
async fn unreachable_server_rejected(world: &mut E2eWorld) {
    let result = last_result(world);
    let endpoint = world.endpoint.as_deref().expect("no unreachable endpoint");
    assert_ne!(result.rc, 0, "unreachable endpoint unexpectedly passed");
    assert_contains(
        &result.output(),
        &format!("failed to resolve a model from the local endpoint {endpoint}"),
    );
    assert!(
        !config_path(world, "aider").exists(),
        "unreachable endpoint wrote configuration"
    );
}

#[then("the relative OMP root and normalized profile precedence select exact files")]
async fn omp_profile_paths_are_exact(world: &mut E2eWorld) {
    let state = state(world);
    assert_eq!(state.results.len(), 4);
    let named = state.home.join("relative-omp/profiles/omp-wins/agent");
    let default = state.home.join("omp-default-agent");
    let cwd_relative = state.project.join("relative-omp");
    for (result, root) in
        state
            .results
            .iter()
            .zip([named, default.clone(), default.clone(), default])
    {
        assert_eq!(result.rc, 0, "OMP inspect failed: {}", result.output());
        let output = result.output();
        assert_contains(&output, &root.join("models.yml").to_string_lossy());
        assert_contains(&output, &root.join("config.yml").to_string_lossy());
        assert!(
            !output.contains(cwd_relative.to_string_lossy().as_ref()),
            "relative PI_CONFIG_DIR was resolved from the working directory: {output}"
        );
    }
}

#[then("OMP setup refuses migration and preserves the legacy registry")]
async fn legacy_omp_registry_is_preserved(world: &mut E2eWorld) {
    let result = last_result(world);
    assert_ne!(result.rc, 0, "legacy OMP setup unexpectedly passed");
    let output = result.output();
    let legacy = config_path(world, "omp").with_extension("json");
    assert_contains(
        &output,
        &format!(
            "legacy OMP model registry {} needs OMP's YAML migration; run OMP once to migrate models.json to models.yml, then rerun this setup",
            legacy.display()
        ),
    );
    assert_original_files_unchanged(world);

    let model_path = config_path(world, "omp");
    let agent = model_path.parent().expect("OMP config has parent");
    for name in ["models.yml", "models.yaml", "config.yml", "config.yaml"] {
        assert!(
            !agent.join(name).exists(),
            "refused legacy setup created {name}"
        );
    }
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
    assert_checkpoint_files_unchanged(world);
}

#[then("dry runs write nothing and repeated setup rewrites no registered configuration")]
async fn pi_omp_dry_run_and_idempotence(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 6);
    for result in &results[..2] {
        assert_eq!(result.rc, 0, "dry run failed: {}", result.output());
        assert_contains(&result.output(), "dry run: no changes written");
    }
    for path in config_paths(world, "pi") {
        assert_contains(&results[0].output(), &path.to_string_lossy());
    }
    let omp_models = config_path(world, "omp");
    assert_contains(&results[1].output(), &omp_models.to_string_lossy());
    assert_contains(&results[1].output(), "providers.rocm-local");
    assert!(
        !results[1].output().contains("modelRoles.default"),
        "OMP dry run planned a default change: {}",
        results[1].output()
    );
    assert_contains(
        &results[1].output(),
        &format!(
            "dry run: after registration, setup will ask whether to use {MODEL} as the default for new OMP sessions"
        ),
    );
    for (first, repeat, agent) in [(2, 3, "pi"), (4, 5, "omp")] {
        assert_eq!(
            results[first].rc,
            0,
            "{agent} setup failed: {}",
            results[first].output()
        );
        assert_eq!(
            results[repeat].rc,
            0,
            "{agent} repeat failed: {}",
            results[repeat].output()
        );
        assert_contains(&results[repeat].output(), "configuration already correct");
    }
    assert_eq!(read_secondary_config(world, "omp"), OMP_CONFIG_ORIGINAL);
    assert_checkpoint_files_unchanged(world);
}

#[then("every global config registers the exact local model and keeps unrelated settings")]
async fn all_adapters_persist_safely(world: &mut E2eWorld) {
    for harness in HARNESSES
        .into_iter()
        .filter(|harness| !matches!(harness.name, "pi" | "omp"))
    {
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
        assert_exact_line(&read_config(world, agent), comment, agent);
    }

    let pi_models = read_config(world, "pi");
    let pi_settings = read_secondary_config(world, "pi");
    for value in [
        DEFAULT_BASE_URL,
        "openai-completions",
        "\"apiKey\"",
        "rocm-local",
        MODEL,
        "keep-pi-models",
    ] {
        assert_contains(&pi_models, value);
    }
    for value in [
        "\"defaultProvider\"",
        "rocm-local",
        "\"defaultModel\"",
        MODEL,
        "keep-pi-settings",
    ] {
        assert_contains(&pi_settings, value);
    }
    assert_exact_line(&pi_models, "// retained pi models comment", "pi models");
    assert_exact_line(
        &pi_settings,
        "// retained pi settings comment",
        "pi settings",
    );

    let omp_models = read_config(world, "omp");
    let omp_config = read_secondary_config(world, "omp");
    for value in [
        DEFAULT_BASE_URL,
        "auth: none",
        "api: openai-completions",
        MODEL,
        "name:",
        "keep-omp-models",
    ] {
        assert_contains(&omp_models, value);
    }
    assert_eq!(omp_config, OMP_CONFIG_ORIGINAL);
    assert_contains(
        &last_result(world).output(),
        "default for new OMP sessions remains unchanged",
    );
    assert_exact_line(&omp_models, "# retained omp models comment", "omp models");
    assert!(
        state(world).results.iter().all(|result| result.rc == 0),
        "adapter setup failed: {:#?}",
        state(world).results
    );
}

#[then("OMP setup and test use the registered model without changing the existing default")]
async fn omp_declined_default_is_unchanged(world: &mut E2eWorld) {
    let outputs = &state(world).interactive_outputs;
    assert_eq!(outputs.len(), 1);
    let output = &outputs[0];
    assert_contains(output, "protocol check passed");
    assert_contains(output, &format!("Use {MODEL} as the default"));
    assert_contains(output, "[y/N]:");
    assert_contains(output, "default for new OMP sessions remains unchanged");
    assert_contains(output, "harness test passed");
    assert_eq!(read_secondary_config(world, "omp"), OMP_CONFIG_ORIGINAL);
    assert_omp_model_registered(world, &world.mock.as_ref().expect("no mock").base_url());
    let capture = std::fs::read_to_string(capture_path(world, "omp"))
        .expect("OMP setup test invocation was not captured");
    assert_safe_fake_args("omp", &capture);
    assert!(
        !world
            .mock
            .as_ref()
            .expect("no mock")
            .protocol_requests()
            .is_empty(),
        "OMP default prompt appeared before the protocol check"
    );
}

#[then("only the OMP default role changes after registration")]
async fn omp_accepted_default_changes_only_default(world: &mut E2eWorld) {
    let outputs = &state(world).interactive_outputs;
    assert_eq!(outputs.len(), 1);
    let output = &outputs[0];
    assert_contains(output, "protocol check passed");
    assert_contains(output, &format!("Use {MODEL} as the default"));
    assert_contains(output, "[y/N]:");
    assert_contains(output, &format!("default for new OMP sessions: {MODEL}"));
    assert_eq!(read_secondary_config(world, "omp"), OMP_CONFIG_DEFAULT);
    assert_omp_model_registered(world, &world.mock.as_ref().expect("no mock").base_url());
}

#[then("the symlink targets are unchanged and both setups explain the refusal")]
async fn symlink_refused(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 2);
    for result in results {
        assert_ne!(result.rc, 0);
        assert_contains(&result.output(), "symlink");
    }
    assert_original_files_unchanged(world);
}

#[then("both stale plans are refused without losing either concurrent edit")]
async fn stale_plan_refused(world: &mut E2eWorld) {
    assert_eq!(state(world).interactive_outputs.len(), 2);
    for output in &state(world).interactive_outputs {
        assert_contains(output, "changed");
    }
    assert_eq!(
        read_config(world, "claude"),
        "{\"concurrent\":\"keep-this-edit\"}\n"
    );
    assert_eq!(
        read_config(world, "omp"),
        "# concurrent OMP registry edit\nunrelatedModels: keep-this-edit\n"
    );
    assert_eq!(
        read_secondary_config(world, "omp"),
        OMP_CONFIG_ORIGINAL,
        "stale registry plan changed the OMP default configuration"
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

#[then("every checked setup fails and restores all original configuration files")]
async fn failed_check_rolls_back(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 3);
    for result in results {
        assert_ne!(result.rc, 0);
        assert_contains(&result.output(), "restored original configuration");
    }
    assert_original_files_unchanged(world);
}

#[then("the first Pi file is rolled back and the oversized second file is unchanged")]
async fn partial_apply_rolls_back(world: &mut E2eWorld) {
    let result = last_result(world);
    assert_ne!(result.rc, 0, "bounded write unexpectedly succeeded");
    assert_contains(&result.output(), "failed to configure pi");
    assert_original_files_unchanged(world);
    assert_no_replacement_debris(world, "pi");
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

#[then("all three setup plans report their exact detected version source")]
async fn detected_version_reported(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 3);
    for (result, version) in results.iter().zip(["2.1.0", "0.84.4", "18.0.11"]) {
        assert_eq!(result.rc, 0, "detected setup failed: {}", result.output());
        assert_contains(&result.output(), &format!("version: {version}"));
        assert_contains(&result.output(), "version source: detected");
    }
}

#[then("every override is visible direct setup succeeds and known managed modes are refused")]
async fn direct_setup_and_managed_refusal(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 7);
    assert_eq!(
        results[0].rc,
        0,
        "Claude override preview failed: {}",
        results[0].output()
    );
    assert_contains(&results[0].output(), "version: 1.9.0");
    assert_contains(&results[0].output(), "version source: override");
    for (result, agent) in results[1..5]
        .iter()
        .zip(["hermes", "openclaw", "pi", "omp"])
    {
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
        if agent == "pi" {
            assert_contains(&read_secondary_config(world, agent), MODEL);
        } else if agent == "omp" {
            assert!(
                !secondary_config_path(world, agent)
                    .expect("OMP has a second config")
                    .exists(),
                "noninteractive OMP setup created a default configuration"
            );
            assert_contains(
                &result.output(),
                "default for new OMP sessions remains unchanged",
            );
        }
    }
    for (result, managed_mode) in results[5..]
        .iter()
        .zip(["HERMES_MANAGED", "OPENCLAW_NIX_MODE"])
    {
        assert_ne!(result.rc, 0, "{managed_mode} setup unexpectedly passed");
        assert_contains(&result.output(), managed_mode);
        assert_contains(&result.output(), "managed/declarative");
    }
}

#[then("all unsupported versions remain inspectable and refuse mutation")]
async fn unsupported_read_only(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 6);
    for (pair, version, agent) in [
        (&results[0..2], "9.0.0", "claude"),
        (&results[2..4], "0.84.5", "pi"),
        (&results[4..6], "19.0.0", "omp"),
    ] {
        assert_eq!(
            pair[0].rc,
            0,
            "{agent} inspection failed: {}",
            pair[0].output()
        );
        assert_contains(&pair[0].output(), version);
        assert_ne!(pair[1].rc, 0, "unsupported {agent} setup passed");
        assert_contains(&pair[1].output(), "not supported for setup");
        assert!(!config_path(world, agent).exists());
    }
}

#[then("every check reaches v1 chat completions with the exact model")]
async fn chat_protocol_requests(world: &mut E2eWorld) {
    let requests = world.mock.as_ref().expect("no mock").protocol_requests();
    assert_eq!(
        requests.len(),
        8,
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
    assert_omp_model_registered(world, DEFAULT_BASE_URL);
    let omp_config = secondary_config_path(world, "omp").expect("OMP has a second config");
    assert!(
        !omp_config.exists(),
        "noninteractive OMP setup created a default configuration"
    );
    assert_contains(
        &last_result(world).output(),
        "default for new OMP sessions remains unchanged",
    );
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

#[then(
    "all global setups warn about higher-precedence project and overlay files without changing them"
)]
async fn project_warning_without_mutation(world: &mut E2eWorld) {
    let results = &state(world).results;
    assert_eq!(results.len(), 3);
    assert!(
        results.iter().all(|result| result.rc == 0),
        "setup with override warnings failed: {results:#?}"
    );
    for result in results {
        assert_contains(&result.output(), "higher precedence");
        assert_contains(
            &result.output(),
            "user-level configuration is the only file changed",
        );
    }
    assert_contains(&results[0].output(), ".aider.conf.yml");
    assert_contains(&results[1].output(), ".pi/settings.json");
    assert_contains(&results[2].output(), ".omp/config.yml");
    assert_contains(&results[2].output(), "PI_CONFIG_FILES");
    for agent in ["aider", "pi", "omp"] {
        assert_contains(&read_config(world, agent), MODEL);
    }
    assert!(
        !secondary_config_path(world, "omp")
            .expect("OMP has a second config")
            .exists(),
        "global OMP default configuration was created"
    );
    assert_project_files_unchanged(world);
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
        original_files: Vec::new(),
        checkpoint_files: Vec::new(),
        project_files: Vec::new(),
        interactive_outputs: Vec::new(),
        protocol_count: None,
        mocks: Vec::new(),
        test_timeout_secs,
    });
}

const fn state(world: &E2eWorld) -> &AgentsState {
    world
        .agents
        .as_ref()
        .expect("agents environment was not initialized")
}

const fn state_mut(world: &mut E2eWorld) -> &mut AgentsState {
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
        "pi" => state.home.join("pi-agent-root/models.json"),
        "omp" => state.home.join("omp-root/profiles/e2e/agent/models.yml"),
        other => panic!("unknown test harness {other}"),
    }
}

fn secondary_config_path(world: &E2eWorld, agent: &str) -> Option<PathBuf> {
    let state = state(world);
    match agent {
        "pi" => Some(state.home.join("pi-agent-root/settings.json")),
        "omp" => Some(state.home.join("omp-root/profiles/e2e/agent/config.yml")),
        _ => None,
    }
}

fn config_paths(world: &E2eWorld, agent: &str) -> Vec<PathBuf> {
    let mut paths = vec![config_path(world, agent)];
    paths.extend(secondary_config_path(world, agent));
    paths
}

fn plant_config(world: &E2eWorld, agent: &str, contents: &str) {
    let path = config_path(world, agent);
    std::fs::create_dir_all(path.parent().expect("config has parent"))
        .expect("failed to create config directory");
    std::fs::write(path, contents).expect("failed to write agent config");
}

fn plant_secondary_config(world: &E2eWorld, agent: &str, contents: &str) {
    let path = secondary_config_path(world, agent).expect("harness has no second config");
    std::fs::create_dir_all(path.parent().expect("config has parent"))
        .expect("failed to create config directory");
    std::fs::write(path, contents).expect("failed to write second agent config");
}

fn plant_pi_omp_configs(world: &E2eWorld) {
    plant_config(
        world,
        "pi",
        "{\n  // retained pi models comment\n  \"unrelatedModels\": \"keep-pi-models\"\n}\n",
    );
    plant_secondary_config(
        world,
        "pi",
        "{\n  // retained pi settings comment\n  \"unrelatedSettings\": \"keep-pi-settings\"\n}\n",
    );
    plant_config(
        world,
        "omp",
        "# retained omp models comment\nunrelatedModels: keep-omp-models\n",
    );
    plant_secondary_config(world, "omp", OMP_CONFIG_ORIGINAL);
}

fn read_config(world: &E2eWorld, agent: &str) -> String {
    let path = config_path(world, agent);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn read_secondary_config(world: &E2eWorld, agent: &str) -> String {
    let path = secondary_config_path(world, agent).expect("harness has no second config");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn snapshot_config(world: &mut E2eWorld, agent: &str) {
    snapshot_path(world, config_path(world, agent));
}

fn snapshot_configs(world: &mut E2eWorld, agent: &str) {
    for path in config_paths(world, agent) {
        snapshot_path(world, path);
    }
}

fn snapshot_path(world: &mut E2eWorld, path: PathBuf) {
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("failed to snapshot {}: {error}", path.display()));
    state_mut(world).original_files.push((path, bytes));
}

fn snapshot_project_path(world: &mut E2eWorld, path: PathBuf) {
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("failed to snapshot {}: {error}", path.display()));
    state_mut(world).project_files.push((path, bytes));
}

fn checkpoint_configs(world: &mut E2eWorld, agent: &str) {
    for path in config_paths(world, agent) {
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("failed to checkpoint {}: {error}", path.display()));
        let modified = std::fs::metadata(&path)
            .and_then(|metadata| metadata.modified())
            .unwrap_or_else(|error| panic!("failed to stat {}: {error}", path.display()));
        state_mut(world)
            .checkpoint_files
            .push((path, bytes, modified));
    }
}

fn assert_config_unchanged(world: &E2eWorld, agent: &str) {
    let path = config_path(world, agent);
    let original = state(world)
        .original_files
        .iter()
        .find(|(candidate, _)| candidate == &path)
        .map(|(_, bytes)| bytes)
        .expect("no config snapshot");
    assert_eq!(
        std::fs::read(&path).expect("failed to read config"),
        *original,
        "agent configuration changed"
    );
}

fn assert_original_files_unchanged(world: &E2eWorld) {
    for (path, original) in &state(world).original_files {
        assert_eq!(
            std::fs::read(path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display())),
            *original,
            "{} changed",
            path.display()
        );
    }
}

fn assert_checkpoint_files_unchanged(world: &E2eWorld) {
    for (path, checkpoint, modified) in &state(world).checkpoint_files {
        assert_eq!(
            std::fs::read(path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display())),
            *checkpoint,
            "idempotent setup changed {} bytes",
            path.display()
        );
        assert_eq!(
            std::fs::metadata(path)
                .and_then(|metadata| metadata.modified())
                .unwrap_or_else(|error| panic!("failed to stat {}: {error}", path.display())),
            *modified,
            "idempotent setup rewrote {}",
            path.display()
        );
    }
}

fn assert_project_files_unchanged(world: &E2eWorld) {
    for (path, original) in &state(world).project_files {
        assert_eq!(
            std::fs::read(path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display())),
            *original,
            "override {} changed",
            path.display()
        );
    }
}

fn run_agents(world: &mut E2eWorld, args: &[&str]) {
    run_agents_with_env(world, args, &[]);
}

fn run_agents_with_env(world: &mut E2eWorld, args: &[&str], environment: &[(&str, &str)]) {
    run_agents_command(world, args, environment, Path::new(&crate::rocm_binary()));
}

fn run_agents_command(
    world: &mut E2eWorld,
    args: &[&str],
    environment: &[(&str, &str)],
    binary: &Path,
) {
    let project = state(world).project.clone();
    let mut command = Command::new(binary);
    command.args(args).current_dir(project);
    for key in [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENCODE_CONFIG_CONTENT",
        "HERMES_MANAGED",
        "OPENCLAW_NIX_MODE",
        "PI_CODING_AGENT_DIR",
        "PI_CONFIG_DIR",
        "PI_PROFILE",
        "OMP_PROFILE",
        "PI_CONFIG_FILES",
        "ROCM_CLI_AGENT_TARGET_FALLBACK_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ] {
        command.env_remove(key);
    }
    world.isolate_cmd(&mut command);
    command.envs(environment.iter().copied());
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", binary.display()));
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

fn run_setup_dry_run(world: &mut E2eWorld, agent: &str, version: &str) {
    run_agents(
        world,
        &[
            "agents",
            agent,
            "--setup",
            "--dry-run",
            "--base-url",
            DEFAULT_BASE_URL,
            "--model",
            MODEL,
            "--agent-version",
            version,
        ],
    );
}

async fn change_file_during_approval(
    world: &mut E2eWorld,
    agent: &str,
    version: &str,
    path: &Path,
    replacement: &str,
) {
    let args = [
        "agents",
        agent,
        "--setup",
        "--no-check",
        "--base-url",
        DEFAULT_BASE_URL,
        "--model",
        MODEL,
        "--agent-version",
        version,
    ];
    let mut session = crate::e2e::tui_driver::TuiSession::spawn(world, &args)
        .expect("failed to spawn interactive setup");
    session
        .wait_for_screen("agent setup plan", Duration::from_secs(10))
        .await
        .expect("setup plan was not shown");
    std::fs::write(path, replacement).expect("failed to make concurrent config edit");
    session.send("y\r").expect("failed to approve setup");
    session
        .wait_for_screen("changed", Duration::from_secs(10))
        .await
        .expect("stale update was not reported");
    let output = session.screen_text();
    world.cli_output = Some(output.clone());
    state_mut(world).interactive_outputs.push(output);
    drop(session);
}

async fn run_interactive_omp_setup(world: &mut E2eWorld, accept_default: bool, test: bool) {
    let base_url = world.mock.as_ref().expect("no mock").base_url();
    let mut args = vec![
        "agents",
        "omp",
        "--setup",
        "--base-url",
        base_url.as_str(),
        "--model",
        MODEL,
    ];
    if test {
        args.push("--test");
    }
    let mut session = crate::e2e::tui_driver::TuiSession::spawn(world, &args)
        .expect("failed to spawn interactive OMP setup");
    session
        .wait_for_screen("agent setup plan", Duration::from_secs(10))
        .await
        .expect("OMP registration plan was not shown");
    let mut transcript = session.screen_text();
    session
        .send("y\r")
        .expect("failed to approve OMP registration");
    session
        .wait_for_screen("protocol check passed", Duration::from_secs(10))
        .await
        .expect("OMP protocol check did not pass before default selection");
    transcript.push_str(&session.screen_text());
    session
        .wait_for_screen(
            &format!("Use {MODEL} as the default"),
            Duration::from_secs(10),
        )
        .await
        .expect("OMP default selection prompt was not shown");
    transcript.push_str(&session.screen_text());
    session
        .send(if accept_default { "y\r" } else { "n\r" })
        .expect("failed to answer OMP default selection");
    let completion = if accept_default {
        format!("default for new OMP sessions: {MODEL}")
    } else {
        "harness test passed".to_owned()
    };
    session
        .wait_for_screen(&completion, Duration::from_secs(10))
        .await
        .expect("interactive OMP setup did not complete");
    transcript.push_str(&session.screen_text());
    session
        .wait_for_exit(Duration::from_secs(10))
        .await
        .expect("interactive OMP setup did not exit successfully");
    world.cli_output = Some(transcript.clone());
    state_mut(world).interactive_outputs.push(transcript);
}

fn run_agents_with_file_limit(world: &mut E2eWorld, args: &[&str]) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;

        let configured = PathBuf::from(crate::rocm_binary());
        let binary = if configured.is_absolute() {
            configured
        } else if configured.components().count() > 1 {
            std::env::current_dir()
                .expect("failed to resolve working directory")
                .join(configured)
        } else {
            find_ambient_executable(configured.to_str().expect("non-UTF-8 ROCm binary"))
                .expect("failed to resolve ROCm CLI binary")
        };
        let launcher = isolated_root(world).join("agents/limited-rocm.sh");
        std::fs::write(
            &launcher,
            format!(
                "#!/bin/sh\ntrap '' XFSZ\nulimit -f 8\nexec {} \"$@\"\n",
                shell_quote(&binary.to_string_lossy())
            ),
        )
        .expect("failed to write bounded ROCm launcher");
        std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o755))
            .expect("failed to make bounded ROCm launcher executable");
        run_agents_command(world, args, &[], &launcher);
    }
    #[cfg(not(unix))]
    {
        let _ = (world, args);
        panic!("bounded write fixture requires Unix");
    }
}

fn assert_exact_line(contents: &str, expected: &str, label: &str) {
    assert!(
        contents.lines().any(|line| line.trim() == expected),
        "{label} did not retain exact line {expected:?}:\n{contents}"
    );
}

fn assert_no_replacement_debris(world: &E2eWorld, agent: &str) {
    let parent = config_path(world, agent)
        .parent()
        .expect("config has parent")
        .to_path_buf();
    let leftovers: Vec<_> = std::fs::read_dir(parent)
        .expect("failed to inspect config directory")
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .filter(|name| {
            name.contains(".rocm")
                || Path::new(name)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("tmp"))
        })
        .collect();
    assert!(
        leftovers.is_empty(),
        "atomic replacement debris: {leftovers:?}"
    );
}

fn unreachable_loopback_url() -> String {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))
        .expect("failed to reserve an unreachable loopback endpoint");
    let address = listener
        .local_addr()
        .expect("unreachable loopback endpoint has no address");
    drop(listener);
    format!("http://{address}/v1")
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

fn assert_omp_model_registered(world: &E2eWorld, base_url: &str) {
    let models = read_config(world, "omp");
    for expected in [base_url, "rocm-local", MODEL] {
        assert_contains(&models, expected);
    }
}

fn assert_fake_arg_pair(capture: &str, flag: &str, value: &str) {
    let expected = if cfg!(windows) {
        format!("{flag} {value}")
    } else {
        format!("<{flag}><{value}>")
    };
    assert!(
        capture.contains(&expected),
        "expected fake argv to contain exact pair {expected:?}:\n{capture}"
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
             {{ printf 'cwd=%s\\nargs=' \"$PWD\"; for arg in \"$@\"; do printf '<%s>' \"$arg\"; done; printf '\\npermission=%s\\npi-offline=<%s>\\npi-skip-version-check=<%s>\\nhttp-proxy=<%s>\\nhttps-proxy=<%s>\\n' \"${{OPENCODE_PERMISSION-}}\" \"${{PI_OFFLINE-}}\" \"${{PI_SKIP_VERSION_CHECK-}}\" \"${{HTTP_PROXY-}}\" \"${{HTTPS_PROXY-}}\"; }} >> {capture}\n\
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
                 echo pi-offline=^<%PI_OFFLINE%^>>>\"{capture}\"\r\n\
                 echo pi-skip-version-check=^<%PI_SKIP_VERSION_CHECK%^>>>\"{capture}\"\r\n\
                 echo http-proxy=^<%HTTP_PROXY%^>>>\"{capture}\"\r\n\
                 echo https-proxy=^<%HTTPS_PROXY%^>>>\"{capture}\"\r\n\
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
        "pi" => &[
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
            "--model",
            "--api-key",
            "--",
        ],
        "omp" => &[
            "-p",
            "--cwd",
            "--no-session",
            "--no-title",
            "--no-tools",
            "--no-lsp",
            "--no-pty",
            "--no-extensions",
            "--no-skills",
            "--no-rules",
            "--max-time",
            "@probe.txt",
        ],
        _ => unreachable!(),
    };
    for flag in required {
        assert_contains(capture, flag);
    }
    match agent {
        "opencode" => {
            assert_contains(capture, "read");
            assert_contains(capture, "deny");
        }
        "pi" => {
            assert_fake_arg_pair(capture, "--provider", "rocm-local");
            assert_fake_arg_pair(capture, "--model", MODEL);
            assert_fake_arg_pair(capture, "--api-key", "rocm-local");
            assert_contains(capture, "pi-offline=<1>");
            assert_contains(capture, "pi-skip-version-check=<1>");
            assert_contains(capture, "http-proxy=<>");
            assert_contains(capture, "https-proxy=<>");
        }
        "omp" => {
            assert_contains(
                capture,
                if cfg!(windows) {
                    "@probe.txt"
                } else {
                    "<@probe.txt>"
                },
            );
            assert_fake_arg_pair(capture, "--model", &format!("rocm-local/{MODEL}"));
            assert_contains(
                capture,
                "Return exactly the contents of the attached probe.txt.",
            );
        }
        _ => {}
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
