// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use super::transaction::{self, Rollback, Snapshot};
use super::{AgentHarness, ResolvedTarget, VersionInfo};
use anyhow::{Context, Result, bail};
use jsonc_parser::cst::{CstInputValue, CstObject, CstRootNode};
use jsonc_parser::{JsonValue, ParseOptions};
use rocm_core::{runtime_config_dir, runtime_home_dir};
use serde_json::{Value, json};
use std::env;
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::str::FromStr;
use toml_edit::{DocumentMut, Item, value};
use yaml_edit::path::YamlPath;
use yaml_edit::{Mapping, MappingBuilder, SequenceBuilder, YamlFile};

const LOCAL_CREDENTIAL: &str = "rocm-local";
const QWEN_CREDENTIAL_ENV: &str = "ROCM_LOCAL_API_KEY";

#[derive(Debug, Clone)]
pub(super) struct ConfigState {
    pub(super) path: PathBuf,
    pub(super) configured: bool,
    pub(super) endpoint: Option<String>,
    pub(super) model: Option<String>,
    pub(super) warnings: Vec<String>,
    snapshot: Snapshot,
}

#[derive(Debug, Clone)]
pub(super) struct SemanticChange {
    pub(super) setting: String,
    pub(super) old_value: Option<String>,
    pub(super) new_value: String,
}

#[derive(Debug, Clone)]
pub(super) struct ConfigPlan {
    pub(super) state: ConfigState,
    pub(super) changes: Vec<SemanticChange>,
    payload: PlanPayload,
}

#[derive(Debug, Clone)]
pub(super) struct AppliedConfig {
    pub(super) changed: bool,
    rollback: Option<Rollback>,
}

#[derive(Debug, Clone)]
enum PlanPayload {
    None,
    Direct(Vec<u8>),
    Native(NativePlan),
}

#[derive(Debug, Clone)]
struct NativePlan {
    executable: PathBuf,
    invocations: Vec<NativeInvocation>,
}

#[derive(Debug, Clone)]
struct NativeInvocation {
    arguments: Vec<OsString>,
    stdin: Option<Vec<u8>>,
}

struct HermesDesired<'a> {
    model: &'a str,
    base_url: &'a str,
}

impl<'a> HermesDesired<'a> {
    fn new(target: &'a ResolvedTarget) -> Self {
        Self {
            model: &target.model,
            base_url: &target.api_base,
        }
    }

    const fn settings(&self) -> [(&'static str, &str); 3] {
        [
            ("model.provider", "custom"),
            ("model.default", self.model),
            ("model.base_url", self.base_url),
        ]
    }
}

struct OpenClawDesired<'a> {
    provider: &'static str,
    mode: &'static str,
    base_url: &'a str,
    api_key: &'static str,
    api: &'static str,
    model: &'a str,
    primary: String,
}

impl<'a> OpenClawDesired<'a> {
    fn new(target: &'a ResolvedTarget) -> Self {
        let provider = "rocm-local";
        Self {
            provider,
            mode: "merge",
            base_url: &target.api_base,
            api_key: LOCAL_CREDENTIAL,
            api: "openai-completions",
            model: &target.model,
            primary: format!("{provider}/{}", target.model),
        }
    }

    fn patch(&self, provider_models: Vec<Value>) -> Value {
        json!({
            "models": {
                "mode": self.mode,
                "providers": {
                    (self.provider): {
                        "baseUrl": self.base_url,
                        "apiKey": self.api_key,
                        "api": self.api,
                        "models": provider_models
                    }
                }
            },
            "agents": { "defaults": { "model": { "primary": self.primary } } }
        })
    }
}

pub(super) fn inspect(harness: AgentHarness) -> Result<ConfigState> {
    let path = config_path(harness)?;
    let snapshot = transaction::snapshot(&path)?;
    let (endpoint, model) = inspect_values(harness, snapshot.raw.as_deref(), &path)?;
    Ok(ConfigState {
        configured: endpoint.is_some() && model.is_some(),
        endpoint,
        model,
        warnings: precedence_warnings(harness),
        path,
        snapshot,
    })
}

pub(super) fn plan(
    harness: AgentHarness,
    version: &VersionInfo,
    target: &ResolvedTarget,
    state: ConfigState,
) -> Result<ConfigPlan> {
    if !version.supported {
        let rendered = version
            .version
            .as_ref()
            .map_or_else(|| "unknown".to_owned(), ToString::to_string);
        bail!(
            "{} version {rendered} has no supported configuration schema",
            harness.canonical_name()
        );
    }
    if state.snapshot.symlink {
        bail!(
            "refusing to configure {} through symlink {}; replace it with a regular file or configure it manually",
            harness.canonical_name(),
            state.path.display()
        );
    }

    let (changes, payload) = match harness {
        AgentHarness::Claude => direct_json_plan(&state, |root, changes| {
            let env = json_object(root, "env")?;
            json_set(
                &env,
                "ANTHROPIC_BASE_URL",
                &target.origin,
                "env.ANTHROPIC_BASE_URL",
                changes,
            );
            json_set(
                &env,
                "ANTHROPIC_AUTH_TOKEN",
                LOCAL_CREDENTIAL,
                "env.ANTHROPIC_AUTH_TOKEN",
                changes,
            );
            json_set(
                &env,
                "ANTHROPIC_API_KEY",
                LOCAL_CREDENTIAL,
                "env.ANTHROPIC_API_KEY",
                changes,
            );
            json_set(
                &env,
                "ANTHROPIC_DEFAULT_OPUS_MODEL",
                &target.model,
                "env.ANTHROPIC_DEFAULT_OPUS_MODEL",
                changes,
            );
            json_set(
                &env,
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
                &target.model,
                "env.ANTHROPIC_DEFAULT_SONNET_MODEL",
                changes,
            );
            json_set(
                &env,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                &target.model,
                "env.ANTHROPIC_DEFAULT_HAIKU_MODEL",
                changes,
            );
            Ok(())
        })?,
        AgentHarness::OpenCode => direct_json_plan(&state, |root, changes| {
            let providers = json_object(root, "provider")?;
            let provider = json_object(&providers, "rocm-local")?;
            json_set(
                &provider,
                "npm",
                "@ai-sdk/openai-compatible",
                "provider.rocm-local.npm",
                changes,
            );
            json_set(
                &provider,
                "name",
                "ROCm Local",
                "provider.rocm-local.name",
                changes,
            );
            let options = json_object(&provider, "options")?;
            json_set(
                &options,
                "baseURL",
                &target.api_base,
                "provider.rocm-local.options.baseURL",
                changes,
            );
            json_set(
                &options,
                "apiKey",
                LOCAL_CREDENTIAL,
                "provider.rocm-local.options.apiKey",
                changes,
            );
            let models = json_object(&provider, "models")?;
            let model = json_object(&models, &target.model)?;
            json_set(
                &model,
                "name",
                &target.model,
                &format!("provider.rocm-local.models.{}.name", target.model),
                changes,
            );
            json_set(
                root,
                "model",
                &format!("rocm-local/{}", target.model),
                "model",
                changes,
            );
            Ok(())
        })?,
        AgentHarness::QwenCode => direct_json_plan(&state, |root, changes| {
            qwen_json(root, version, target, changes)
        })?,
        AgentHarness::Codex => direct_toml_plan(&state, target)?,
        AgentHarness::Aider => direct_aider_plan(&state, target)?,
        AgentHarness::Continue => direct_continue_plan(&state, target)?,
        AgentHarness::Hermes => {
            if version.executable.is_some() {
                native_hermes_plan(&state, version, target)?
            } else {
                refuse_managed_direct(harness, "HERMES_MANAGED")?;
                direct_hermes_plan(&state, target)?
            }
        }
        AgentHarness::OpenClaw => {
            if version.executable.is_some() {
                native_openclaw_plan(&state, version, target)?
            } else {
                refuse_managed_direct(harness, "OPENCLAW_NIX_MODE")?;
                direct_openclaw_plan(&state, target)?
            }
        }
    };

    Ok(ConfigPlan {
        state,
        changes,
        payload,
    })
}

pub(super) fn apply(plan: &ConfigPlan) -> Result<AppliedConfig> {
    if plan.changes.is_empty() {
        return Ok(AppliedConfig {
            changed: false,
            rollback: None,
        });
    }
    transaction::ensure_fresh(&plan.state.path, &plan.state.snapshot)?;

    match &plan.payload {
        PlanPayload::None => Ok(AppliedConfig {
            changed: false,
            rollback: None,
        }),
        PlanPayload::Direct(after) => {
            transaction::atomic_write(&plan.state.path, after, plan.state.snapshot.mode)?;
            Ok(AppliedConfig {
                changed: true,
                rollback: Some(Rollback::new(
                    plan.state.path.clone(),
                    plan.state.snapshot.clone(),
                    Some(after.clone()),
                )),
            })
        }
        PlanPayload::Native(native) => apply_native(plan, native),
    }
}

pub(super) fn rollback(applied: &AppliedConfig) -> Result<()> {
    match &applied.rollback {
        Some(rollback) => rollback.restore(),
        None => Ok(()),
    }
}

fn config_path(harness: AgentHarness) -> Result<PathBuf> {
    let home = user_home()?;
    let path = match harness {
        AgentHarness::Claude => env_path("CLAUDE_CONFIG_DIR")
            .unwrap_or_else(|| home.join(".claude"))
            .join("settings.json"),
        AgentHarness::Hermes => env_path("HERMES_HOME")
            .unwrap_or_else(|| {
                if cfg!(windows) {
                    env_path("LOCALAPPDATA")
                        .unwrap_or_else(|| home.join("AppData").join("Local"))
                        .join("hermes")
                } else {
                    home.join(".hermes")
                }
            })
            .join("config.yaml"),
        AgentHarness::OpenClaw => {
            if let Some(path) = env_path("OPENCLAW_CONFIG_PATH") {
                path
            } else {
                env_path("OPENCLAW_STATE_DIR")
                    .unwrap_or_else(|| openclaw_home(&home).join(".openclaw"))
                    .join("openclaw.json")
            }
        }
        AgentHarness::Codex => env_path("CODEX_HOME")
            .unwrap_or_else(|| home.join(".codex"))
            .join("config.toml"),
        AgentHarness::OpenCode => {
            if let Some(path) = env_path("OPENCODE_CONFIG") {
                absolute_from_cwd(path)?
            } else {
                let root = config_root(&home).join("opencode");
                let json = root.join("opencode.json");
                let jsonc = root.join("opencode.jsonc");
                if !json.exists() && jsonc.exists() {
                    jsonc
                } else {
                    json
                }
            }
        }
        AgentHarness::QwenCode => env_path("QWEN_HOME")
            .unwrap_or_else(|| home.join(".qwen"))
            .join("settings.json"),
        AgentHarness::Aider => home.join(".aider.conf.yml"),
        AgentHarness::Continue => env_path("CONTINUE_GLOBAL_DIR")
            .map(absolute_from_cwd)
            .transpose()?
            .unwrap_or_else(|| home.join(".continue"))
            .join("config.yaml"),
    };
    Ok(path)
}

fn user_home() -> Result<PathBuf> {
    runtime_home_dir().ok_or_else(|| anyhow::anyhow!("could not determine the user home directory"))
}

fn openclaw_home(home: &Path) -> PathBuf {
    env_path("OPENCLAW_HOME").unwrap_or_else(|| home.to_path_buf())
}

fn config_root(home: &Path) -> PathBuf {
    runtime_config_dir().unwrap_or_else(|| {
        if cfg!(windows) {
            home.join("AppData").join("Roaming")
        } else {
            home.join(".config")
        }
    })
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn absolute_from_cwd(path: PathBuf) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(std::env::current_dir()
            .context("failed to resolve a relative configuration path")?
            .join(path))
    }
}

fn precedence_warnings(harness: AgentHarness) -> Vec<String> {
    let mut warnings = Vec::new();
    let candidates: &[&str] = match harness {
        AgentHarness::Claude => &[".claude/settings.json", ".claude/settings.local.json"],
        AgentHarness::Codex => &[".codex/config.toml"],
        AgentHarness::OpenCode => &["opencode.json", "opencode.jsonc"],
        AgentHarness::QwenCode => &[".qwen/settings.json"],
        AgentHarness::Aider => &[".aider.conf.yml"],
        AgentHarness::Continue => &[".continue/config.yaml"],
        AgentHarness::Hermes | AgentHarness::OpenClaw => &[],
    };
    for candidate in candidates {
        if Path::new(candidate).is_file() {
            warnings.push(format!(
                "project configuration {candidate} has higher precedence; the user-level configuration is the only file changed"
            ));
        }
    }
    if harness == AgentHarness::OpenCode && std::env::var_os("OPENCODE_CONFIG_CONTENT").is_some() {
        warnings.push(
            "OPENCODE_CONFIG_CONTENT has higher precedence than the user-level file".to_owned(),
        );
    }
    if harness == AgentHarness::Continue {
        warnings.push("Continue may retain an explicit last-selected IDE model; setup does not change opaque editor state".to_owned());
    }
    warnings
}

fn inspect_values(
    harness: AgentHarness,
    raw: Option<&[u8]>,
    path: &Path,
) -> Result<(Option<String>, Option<String>)> {
    let Some(raw) = raw else {
        return Ok((None, None));
    };
    let text = std::str::from_utf8(raw)
        .with_context(|| format!("{} is not valid UTF-8", path.display()))?;
    match harness {
        AgentHarness::Claude => {
            let value = parse_json_value(text, path)?;
            Ok((
                json_string_at(&value, &["env", "ANTHROPIC_BASE_URL"]),
                json_string_at(&value, &["env", "ANTHROPIC_DEFAULT_SONNET_MODEL"]),
            ))
        }
        AgentHarness::OpenCode => {
            let value = parse_json_value(text, path)?;
            let endpoint =
                json_string_at(&value, &["provider", "rocm-local", "options", "baseURL"]);
            let model = json_string_at(&value, &["model"])
                .and_then(|value| value.strip_prefix("rocm-local/").map(str::to_owned));
            Ok((endpoint, model))
        }
        AgentHarness::QwenCode => {
            let value = parse_json_value(text, path)?;
            let model = json_string_at(&value, &["model", "name"]);
            let endpoint = model
                .as_deref()
                .and_then(|model| qwen_endpoint(&value, model));
            Ok((endpoint, model))
        }
        AgentHarness::OpenClaw => {
            let value = parse_json_value(text, path)?;
            let endpoint =
                json_string_at(&value, &["models", "providers", "rocm-local", "baseUrl"]);
            let model = json_string_at(&value, &["agents", "defaults", "model", "primary"])
                .and_then(|value| value.strip_prefix("rocm-local/").map(str::to_owned));
            Ok((endpoint, model))
        }
        AgentHarness::Codex => {
            let document = text
                .parse::<DocumentMut>()
                .with_context(|| format!("failed to parse {} as TOML", path.display()))?;
            let endpoint = document
                .get("model_providers")
                .and_then(Item::as_table_like)
                .and_then(|providers| providers.get("rocm-local"))
                .and_then(Item::as_table_like)
                .and_then(|provider| provider.get("base_url"))
                .and_then(Item::as_str)
                .map(str::to_owned);
            let model = document
                .get("model")
                .and_then(Item::as_str)
                .map(str::to_owned);
            Ok((endpoint, model))
        }
        AgentHarness::Hermes => {
            let document = parse_yaml(text, path)?;
            Ok((
                yaml_path_string(&document, "model.base_url"),
                yaml_path_string(&document, "model.default")
                    .or_else(|| yaml_path_string(&document, "model.model")),
            ))
        }
        AgentHarness::Aider => {
            let document = parse_yaml(text, path)?;
            let model = yaml_path_string(&document, "model")
                .map(|value| value.strip_prefix("openai/").unwrap_or(&value).to_owned());
            Ok((yaml_path_string(&document, "openai-api-base"), model))
        }
        AgentHarness::Continue => {
            let document = parse_yaml(text, path)?;
            if let Some(models) = document
                .document()
                .and_then(|document| document.get_path("models"))
                .and_then(|node| node.as_sequence().cloned())
            {
                for node in models.values() {
                    let Some(mapping) = node.as_mapping() else {
                        continue;
                    };
                    if yaml_mapping_string(mapping, "name").as_deref() == Some("ROCm Local") {
                        return Ok((
                            yaml_mapping_string(mapping, "apiBase"),
                            yaml_mapping_string(mapping, "model"),
                        ));
                    }
                }
            }
            Ok((None, None))
        }
    }
}

fn parse_json_value(text: &str, path: &Path) -> Result<Value> {
    if text.trim().is_empty() {
        return Ok(json!({}));
    }
    jsonc_parser::parse_to_value(text, &ParseOptions::default())
        .with_context(|| format!("failed to parse {} as JSONC", path.display()))?
        .map(json_value_to_serde)
        .ok_or_else(|| anyhow::anyhow!("{} must contain a JSON object", path.display()))
}

fn json_value_to_serde(value: JsonValue<'_>) -> Value {
    match value {
        JsonValue::String(value) => Value::String(value.into_owned()),
        JsonValue::Number(raw) => {
            let unsigned = raw.trim_start_matches(['-', '+']);
            if unsigned.len() > 2 && (unsigned.starts_with("0x") || unsigned.starts_with("0X")) {
                i64::from_str_radix(&unsigned[2..], 16).map_or_else(
                    |_| Value::String(raw.to_owned()),
                    |value| {
                        Value::Number((if raw.starts_with('-') { -value } else { value }).into())
                    },
                )
            } else {
                raw.trim_start_matches('+')
                    .parse()
                    .map_or_else(|_| Value::String(raw.to_owned()), Value::Number)
            }
        }
        JsonValue::Boolean(value) => Value::Bool(value),
        JsonValue::Object(value) => Value::Object(
            value
                .into_iter()
                .map(|(key, value)| (key.into_owned(), json_value_to_serde(value)))
                .collect(),
        ),
        JsonValue::Array(value) => {
            Value::Array(value.into_iter().map(json_value_to_serde).collect())
        }
        JsonValue::Null => Value::Null,
    }
}

fn direct_json_plan<F>(state: &ConfigState, edit: F) -> Result<(Vec<SemanticChange>, PlanPayload)>
where
    F: FnOnce(&CstObject, &mut Vec<SemanticChange>) -> Result<()>,
{
    let text = state
        .snapshot
        .raw
        .as_deref()
        .map(std::str::from_utf8)
        .transpose()
        .with_context(|| format!("{} is not valid UTF-8", state.path.display()))?
        .unwrap_or("{}\n");
    let root = CstRootNode::parse(text, &ParseOptions::default())
        .with_context(|| format!("failed to parse {} as JSONC", state.path.display()))?;
    let object = match root.object_value() {
        Some(object) => object,
        None if root.value().is_none() => root.object_value_or_set(),
        None => bail!("{} must contain a JSON object", state.path.display()),
    };
    let mut changes = Vec::new();
    edit(&object, &mut changes)?;
    if changes.is_empty() {
        Ok((changes, PlanPayload::None))
    } else {
        Ok((changes, PlanPayload::Direct(root.to_string().into_bytes())))
    }
}

fn json_object(parent: &CstObject, key: &str) -> Result<CstObject> {
    match parent.get(key) {
        Some(property) => property
            .object_value()
            .ok_or_else(|| anyhow::anyhow!("configuration setting {key} must be an object")),
        None => Ok(parent
            .append(key, CstInputValue::Object(Vec::new()))
            .object_value_or_set()),
    }
}

fn json_array(parent: &CstObject, key: &str) -> Result<jsonc_parser::cst::CstArray> {
    match parent.get(key) {
        Some(property) => property
            .array_value()
            .ok_or_else(|| anyhow::anyhow!("configuration setting {key} must be an array")),
        None => Ok(parent
            .append(key, CstInputValue::Array(Vec::new()))
            .array_value_or_set()),
    }
}

fn json_set(
    parent: &CstObject,
    key: &str,
    desired: &str,
    setting: &str,
    changes: &mut Vec<SemanticChange>,
) {
    let old = parent
        .get(key)
        .and_then(|property| property.value())
        .and_then(|value| {
            let source = value.to_string();
            jsonc_parser::parse_to_value(&source, &ParseOptions::default())
                .ok()
                .flatten()
                .map(json_value_to_serde)
        });
    let desired_value = Value::String(desired.to_owned());
    if old.as_ref() == Some(&desired_value) {
        return;
    }
    push_change(changes, setting, old.as_ref(), &desired_value);
    if let Some(property) = parent.get(key) {
        property.set_value(desired.into());
    } else {
        parent.append(key, desired.into());
    }
}

fn qwen_json(
    root: &CstObject,
    version: &VersionInfo,
    target: &ResolvedTarget,
    changes: &mut Vec<SemanticChange>,
) -> Result<()> {
    let providers = json_object(root, "modelProviders")?;
    let current_schema = version
        .version
        .as_ref()
        .is_some_and(|version| version.major == 0 && version.minor >= 10);
    let models = if current_schema {
        let protocols = json_object(root, "providerProtocol")?;
        json_set(
            &protocols,
            "rocm-local",
            "openai",
            "providerProtocol.rocm-local",
            changes,
        );
        json_array(&providers, "rocm-local")?
    } else {
        let provider = json_object(&providers, "rocm-local")?;
        json_set(
            &provider,
            "protocol",
            "openai",
            "modelProviders.rocm-local.protocol",
            changes,
        );
        json_array(&provider, "models")?
    };

    let existing = models.elements().into_iter().find_map(|node| {
        let object = node.as_object()?;
        let id = object
            .get("id")?
            .value()?
            .as_string_lit()?
            .decoded_value()
            .ok()?;
        (id == target.model).then_some(object)
    });
    let model = if let Some(model) = existing {
        model
    } else {
        push_change(
            changes,
            "modelProviders.rocm-local[].id",
            None,
            &Value::String(target.model.clone()),
        );
        models
            .append(CstInputValue::Object(vec![(
                "id".to_owned(),
                target.model.clone().into(),
            )]))
            .as_object()
            .expect("appended object")
    };
    json_set(
        &model,
        "name",
        &target.model,
        "modelProviders.rocm-local[].name",
        changes,
    );
    json_set(
        &model,
        "envKey",
        QWEN_CREDENTIAL_ENV,
        "modelProviders.rocm-local[].envKey",
        changes,
    );
    json_set(
        &model,
        "baseUrl",
        &target.api_base,
        "modelProviders.rocm-local[].baseUrl",
        changes,
    );

    let env = json_object(root, "env")?;
    json_set(
        &env,
        QWEN_CREDENTIAL_ENV,
        LOCAL_CREDENTIAL,
        &format!("env.{QWEN_CREDENTIAL_ENV}"),
        changes,
    );
    let selected_model = json_object(root, "model")?;
    json_set(
        &selected_model,
        "name",
        &target.model,
        "model.name",
        changes,
    );
    let security = json_object(root, "security")?;
    let auth = json_object(&security, "auth")?;
    json_set(
        &auth,
        "selectedType",
        "rocm-local",
        "security.auth.selectedType",
        changes,
    );
    Ok(())
}

fn direct_toml_plan(
    state: &ConfigState,
    target: &ResolvedTarget,
) -> Result<(Vec<SemanticChange>, PlanPayload)> {
    let text = state
        .snapshot
        .raw
        .as_deref()
        .map(std::str::from_utf8)
        .transpose()
        .with_context(|| format!("{} is not valid UTF-8", state.path.display()))?
        .unwrap_or("");
    let mut document = text
        .parse::<DocumentMut>()
        .with_context(|| format!("failed to parse {} as TOML", state.path.display()))?;
    let mut changes = Vec::new();
    toml_set_string(&mut document["model"], &target.model, "model", &mut changes);
    toml_set_string(
        &mut document["model_provider"],
        "rocm-local",
        "model_provider",
        &mut changes,
    );
    toml_set_string(
        &mut document["model_providers"]["rocm-local"]["name"],
        "ROCm Local",
        "model_providers.rocm-local.name",
        &mut changes,
    );
    toml_set_string(
        &mut document["model_providers"]["rocm-local"]["base_url"],
        &target.api_base,
        "model_providers.rocm-local.base_url",
        &mut changes,
    );
    toml_set_string(
        &mut document["model_providers"]["rocm-local"]["wire_api"],
        "responses",
        "model_providers.rocm-local.wire_api",
        &mut changes,
    );
    let auth_item = &mut document["model_providers"]["rocm-local"]["requires_openai_auth"];
    if auth_item.as_bool() != Some(false) {
        push_change(
            &mut changes,
            "model_providers.rocm-local.requires_openai_auth",
            auth_item.as_value().map(toml_value_to_json).as_ref(),
            &Value::Bool(false),
        );
        *auth_item = value(false);
    }
    if changes.is_empty() {
        Ok((changes, PlanPayload::None))
    } else {
        Ok((
            changes,
            PlanPayload::Direct(document.to_string().into_bytes()),
        ))
    }
}

fn toml_set_string(
    item: &mut Item,
    desired: &str,
    setting: &str,
    changes: &mut Vec<SemanticChange>,
) {
    if item.as_str() == Some(desired) {
        return;
    }
    let old = item.as_value().map(toml_value_to_json);
    push_change(
        changes,
        setting,
        old.as_ref(),
        &Value::String(desired.to_owned()),
    );
    *item = value(desired);
}

fn toml_value_to_json(value: &toml_edit::Value) -> Value {
    if let Some(value) = value.as_str() {
        Value::String(value.to_owned())
    } else if let Some(value) = value.as_bool() {
        Value::Bool(value)
    } else {
        Value::String(value.to_string())
    }
}

fn direct_aider_plan(
    state: &ConfigState,
    target: &ResolvedTarget,
) -> Result<(Vec<SemanticChange>, PlanPayload)> {
    let file = yaml_file(state)?;
    let root = file
        .document()
        .and_then(|document| document.as_mapping())
        .ok_or_else(|| anyhow::anyhow!("{} must contain a YAML mapping", state.path.display()))?;
    let mut changes = Vec::new();
    yaml_set_string(
        &root,
        "model",
        &format!("openai/{}", target.model),
        "model",
        &mut changes,
    );
    yaml_set_string(
        &root,
        "openai-api-base",
        &target.api_base,
        "openai-api-base",
        &mut changes,
    );
    yaml_set_string(
        &root,
        "openai-api-key",
        LOCAL_CREDENTIAL,
        "openai-api-key",
        &mut changes,
    );
    yaml_result(&file, changes)
}

fn direct_continue_plan(
    state: &ConfigState,
    target: &ResolvedTarget,
) -> Result<(Vec<SemanticChange>, PlanPayload)> {
    let file = yaml_file(state)?;
    let root = file
        .document()
        .and_then(|document| document.as_mapping())
        .ok_or_else(|| anyhow::anyhow!("{} must contain a YAML mapping", state.path.display()))?;
    let mut changes = Vec::new();
    let mut found = false;
    if let Some(node) = root.get("models") {
        let models = node
            .as_sequence()
            .ok_or_else(|| anyhow::anyhow!("configuration setting models must be a YAML sequence"))?
            .clone();
        for entry in models.values() {
            let Some(mapping) = entry.as_mapping() else {
                continue;
            };
            if yaml_mapping_string(mapping, "name").as_deref() == Some("ROCm Local") {
                yaml_set_string(
                    mapping,
                    "provider",
                    "openai",
                    "models[ROCm Local].provider",
                    &mut changes,
                );
                yaml_set_string(
                    mapping,
                    "model",
                    &target.model,
                    "models[ROCm Local].model",
                    &mut changes,
                );
                yaml_set_string(
                    mapping,
                    "apiBase",
                    &target.api_base,
                    "models[ROCm Local].apiBase",
                    &mut changes,
                );
                yaml_set_string(
                    mapping,
                    "apiKey",
                    LOCAL_CREDENTIAL,
                    "models[ROCm Local].apiKey",
                    &mut changes,
                );
                found = true;
                break;
            }
        }
        if !found {
            changes.push(SemanticChange {
                setting: "models[ROCm Local]".to_owned(),
                old_value: None,
                new_value: "ROCm Local".to_owned(),
            });
            models.push(continue_model(target));
        }
    } else {
        changes.push(SemanticChange {
            setting: "models[ROCm Local]".to_owned(),
            old_value: None,
            new_value: "ROCm Local".to_owned(),
        });
        let models = SequenceBuilder::new()
            .item(continue_model(target))
            .build_document()
            .as_sequence()
            .expect("built sequence");
        root.set("models", models);
    }
    yaml_result(&file, changes)
}

fn continue_model(target: &ResolvedTarget) -> Mapping {
    MappingBuilder::new()
        .pair("name", "ROCm Local")
        .pair("provider", "openai")
        .pair("model", target.model.clone())
        .pair("apiBase", target.api_base.clone())
        .pair("apiKey", LOCAL_CREDENTIAL)
        .build_document()
        .as_mapping()
        .expect("built mapping")
}

fn yaml_file(state: &ConfigState) -> Result<YamlFile> {
    let text = state
        .snapshot
        .raw
        .as_deref()
        .map(std::str::from_utf8)
        .transpose()
        .with_context(|| format!("{} is not valid UTF-8", state.path.display()))?
        .unwrap_or("");
    if text.trim().is_empty() {
        let file = YamlFile::new();
        file.ensure_document();
        Ok(file)
    } else {
        parse_yaml(text, &state.path)
    }
}

fn parse_yaml(text: &str, path: &Path) -> Result<YamlFile> {
    let file = YamlFile::from_str(text)
        .with_context(|| format!("failed to parse {} as YAML", path.display()))?;
    if file.documents().nth(1).is_some() {
        bail!(
            "failed to parse {} as YAML: input contains multiple YAML documents",
            path.display()
        );
    }
    Ok(file)
}

fn yaml_result(
    file: &YamlFile,
    changes: Vec<SemanticChange>,
) -> Result<(Vec<SemanticChange>, PlanPayload)> {
    if changes.is_empty() {
        Ok((changes, PlanPayload::None))
    } else {
        Ok((changes, PlanPayload::Direct(file.to_string().into_bytes())))
    }
}

fn yaml_set_string(
    mapping: &Mapping,
    key: &str,
    desired: &str,
    setting: &str,
    changes: &mut Vec<SemanticChange>,
) {
    let old = yaml_mapping_string(mapping, key);
    if old.as_deref() == Some(desired) {
        return;
    }
    push_change(
        changes,
        setting,
        old.as_ref()
            .map(|value| Value::String(value.clone()))
            .as_ref(),
        &Value::String(desired.to_owned()),
    );
    mapping.set(key, desired);
}

fn yaml_mapping_string(mapping: &Mapping, key: &str) -> Option<String> {
    mapping
        .get(key)
        .and_then(|node| node.as_scalar().map(yaml_edit::Scalar::as_string))
}

fn yaml_path_string(file: &YamlFile, path: &str) -> Option<String> {
    file.document()?
        .get_path(path)
        .and_then(|node| node.as_scalar().map(yaml_edit::Scalar::as_string))
}

fn direct_hermes_plan(
    state: &ConfigState,
    target: &ResolvedTarget,
) -> Result<(Vec<SemanticChange>, PlanPayload)> {
    let file = yaml_file(state)?;
    let root = file
        .document()
        .and_then(|document| document.as_mapping())
        .ok_or_else(|| anyhow::anyhow!("{} must contain a YAML mapping", state.path.display()))?;
    let model = if let Some(node) = root.get("model") {
        node.as_mapping()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("configuration setting model must be a YAML mapping"))?
    } else {
        let empty = MappingBuilder::new()
            .build_document()
            .as_mapping()
            .expect("built mapping");
        root.set("model", empty);
        root.get("model")
            .and_then(|node| node.as_mapping().cloned())
            .expect("inserted model mapping")
    };
    let desired = HermesDesired::new(target);
    let mut changes = Vec::new();
    for (setting, value) in desired.settings() {
        yaml_set_string(
            &model,
            setting
                .rsplit('.')
                .next()
                .expect("Hermes setting has a leaf"),
            value,
            setting,
            &mut changes,
        );
    }
    yaml_result(&file, changes)
}

fn direct_openclaw_plan(
    state: &ConfigState,
    target: &ResolvedTarget,
) -> Result<(Vec<SemanticChange>, PlanPayload)> {
    let desired = OpenClawDesired::new(target);
    direct_json_plan(state, |root, changes| {
        let models = json_object(root, "models")?;
        json_set(&models, "mode", desired.mode, "models.mode", changes);
        let providers = json_object(&models, "providers")?;
        let provider = json_object(&providers, desired.provider)?;
        json_set(
            &provider,
            "baseUrl",
            desired.base_url,
            "models.providers.rocm-local.baseUrl",
            changes,
        );
        json_set(
            &provider,
            "apiKey",
            desired.api_key,
            "models.providers.rocm-local.apiKey",
            changes,
        );
        json_set(
            &provider,
            "api",
            desired.api,
            "models.providers.rocm-local.api",
            changes,
        );
        let provider_models = json_array(&provider, "models")?;
        let existing = provider_models.elements().into_iter().find_map(|node| {
            let object = node.as_object()?;
            let id = object
                .get("id")?
                .value()?
                .as_string_lit()?
                .decoded_value()
                .ok()?;
            (id == desired.model).then_some(object)
        });
        if let Some(model) = existing {
            json_set(
                &model,
                "name",
                desired.model,
                "models.providers.rocm-local.models[].name",
                changes,
            );
        } else {
            push_change(
                changes,
                "models.providers.rocm-local.models[].id",
                None,
                &Value::String(desired.model.to_owned()),
            );
            provider_models.append(CstInputValue::Object(vec![
                ("id".to_owned(), desired.model.to_owned().into()),
                ("name".to_owned(), desired.model.to_owned().into()),
            ]));
        }
        let agents = json_object(root, "agents")?;
        let defaults = json_object(&agents, "defaults")?;
        let selected = json_object(&defaults, "model")?;
        json_set(
            &selected,
            "primary",
            &desired.primary,
            "agents.defaults.model.primary",
            changes,
        );
        Ok(())
    })
}

fn native_hermes_plan(
    state: &ConfigState,
    version: &VersionInfo,
    target: &ResolvedTarget,
) -> Result<(Vec<SemanticChange>, PlanPayload)> {
    let executable = native_executable(version, AgentHarness::Hermes)?;
    let desired = HermesDesired::new(target);
    let mut changes = Vec::new();
    for (setting, value) in desired.settings() {
        native_change(
            &mut changes,
            setting,
            yaml_raw_string(state, setting)?,
            value,
        );
    }
    let invocations = if changes.is_empty() {
        Vec::new()
    } else {
        desired
            .settings()
            .into_iter()
            .map(|(setting, value)| native_set(setting, value))
            .collect()
    };
    let payload = if changes.is_empty() {
        PlanPayload::None
    } else {
        PlanPayload::Native(NativePlan {
            executable,
            invocations,
        })
    };
    Ok((changes, payload))
}

fn native_openclaw_plan(
    state: &ConfigState,
    version: &VersionInfo,
    target: &ResolvedTarget,
) -> Result<(Vec<SemanticChange>, PlanPayload)> {
    let executable = native_executable(version, AgentHarness::OpenClaw)?;
    let desired = OpenClawDesired::new(target);
    let value = state
        .snapshot
        .raw
        .as_deref()
        .map(std::str::from_utf8)
        .transpose()
        .with_context(|| format!("{} is not valid UTF-8", state.path.display()))?
        .map(|text| parse_json_value(text, &state.path))
        .transpose()?
        .unwrap_or_else(|| json!({}));
    let mut changes = Vec::new();
    native_json_change(
        &mut changes,
        "models.mode",
        json_at(&value, &["models", "mode"]),
        desired.mode,
    );
    native_json_change(
        &mut changes,
        "models.providers.rocm-local.baseUrl",
        json_at(
            &value,
            &["models", "providers", desired.provider, "baseUrl"],
        ),
        desired.base_url,
    );
    native_json_change(
        &mut changes,
        "models.providers.rocm-local.apiKey",
        json_at(&value, &["models", "providers", desired.provider, "apiKey"]),
        desired.api_key,
    );
    native_json_change(
        &mut changes,
        "models.providers.rocm-local.api",
        json_at(&value, &["models", "providers", desired.provider, "api"]),
        desired.api,
    );
    native_json_change(
        &mut changes,
        "agents.defaults.model.primary",
        json_at(&value, &["agents", "defaults", "model", "primary"]),
        &desired.primary,
    );
    let mut provider_models = json_at(&value, &["models", "providers", desired.provider, "models"])
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if let Some(model) = provider_models
        .iter_mut()
        .find(|model| model.get("id").and_then(Value::as_str) == Some(desired.model))
    {
        let desired_name = Value::String(desired.model.to_owned());
        if model.get("name") != Some(&desired_name) {
            push_change(
                &mut changes,
                "models.providers.rocm-local.models[].name",
                model.get("name"),
                &desired_name,
            );
            let object = model.as_object_mut().ok_or_else(|| {
                anyhow::anyhow!("OpenClaw rocm-local model entry must be an object")
            })?;
            object.insert("name".to_owned(), desired_name);
        }
    } else {
        changes.push(SemanticChange {
            setting: "models.providers.rocm-local.models[]".to_owned(),
            old_value: None,
            new_value: desired.model.to_owned(),
        });
        provider_models.push(json!({ "id": desired.model, "name": desired.model }));
    }
    if changes.is_empty() {
        return Ok((changes, PlanPayload::None));
    }
    let patch = desired.patch(provider_models);
    let invocation = NativeInvocation {
        arguments: vec![
            OsString::from("config"),
            OsString::from("patch"),
            OsString::from("--stdin"),
        ],
        stdin: Some(
            serde_json::to_vec(&patch).context("failed to encode OpenClaw configuration patch")?,
        ),
    };
    Ok((
        changes,
        PlanPayload::Native(NativePlan {
            executable,
            invocations: vec![invocation],
        }),
    ))
}

fn native_executable(version: &VersionInfo, harness: AgentHarness) -> Result<PathBuf> {
    version.executable.clone().ok_or_else(|| anyhow::anyhow!(
        "{} setup requires the installed {} executable because its validated native configuration command is the only supported write surface",
        harness.canonical_name(), harness.executable()
    ))
}

fn refuse_managed_direct(harness: AgentHarness, variable: &str) -> Result<()> {
    if env::var_os(variable).is_some() {
        bail!(
            "refusing direct {} configuration because {variable} marks the environment as managed/declarative; install {} and rerun setup so its native configuration policy can be enforced",
            harness.canonical_name(),
            harness.executable()
        );
    }
    Ok(())
}

fn native_set(setting: &str, value: &str) -> NativeInvocation {
    NativeInvocation {
        arguments: ["config", "set", setting, value]
            .into_iter()
            .map(OsString::from)
            .collect(),
        stdin: None,
    }
}

fn yaml_raw_string(state: &ConfigState, path: &str) -> Result<Option<String>> {
    let Some(raw) = state.snapshot.raw.as_deref() else {
        return Ok(None);
    };
    let text = std::str::from_utf8(raw)
        .with_context(|| format!("{} is not valid UTF-8", state.path.display()))?;
    Ok(yaml_path_string(&parse_yaml(text, &state.path)?, path))
}

fn native_change(
    changes: &mut Vec<SemanticChange>,
    setting: &str,
    old: Option<String>,
    desired: &str,
) {
    if old.as_deref() != Some(desired) {
        push_change(
            changes,
            setting,
            old.as_ref()
                .map(|value| Value::String(value.clone()))
                .as_ref(),
            &Value::String(desired.to_owned()),
        );
    }
}

fn native_json_change(
    changes: &mut Vec<SemanticChange>,
    setting: &str,
    old: Option<&Value>,
    desired: &str,
) {
    let desired = Value::String(desired.to_owned());
    if old != Some(&desired) {
        push_change(changes, setting, old, &desired);
    }
}

fn apply_native(plan: &ConfigPlan, native: &NativePlan) -> Result<AppliedConfig> {
    for invocation in &native.invocations {
        if let Err(error) = run_native(&native.executable, invocation) {
            let _ = transaction::restore_if_changed(&plan.state.path, &plan.state.snapshot);
            return Err(error);
        }
    }
    transaction::reject_symlink(&plan.state.path)?;
    let after = transaction::read_optional(&plan.state.path)?;
    Ok(AppliedConfig {
        changed: true,
        rollback: Some(Rollback::new(
            plan.state.path.clone(),
            plan.state.snapshot.clone(),
            after,
        )),
    })
}

fn run_native(executable: &Path, invocation: &NativeInvocation) -> Result<()> {
    let mut command = Command::new(executable);
    command
        .args(&invocation.arguments)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if invocation.stdin.is_some() {
        command.stdin(Stdio::piped());
    } else {
        command.stdin(Stdio::null());
    }
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to run {}", executable.display()))?;
    if let Some(input) = &invocation.stdin {
        child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("failed to open native configuration command input"))?
            .write_all(input)
            .context("failed to send native configuration patch")?;
    }
    let output = child
        .wait_with_output()
        .context("failed to wait for native configuration command")?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !stderr.is_empty() {
        bail!(stderr)
    }
    if !stdout.is_empty() {
        bail!(stdout)
    }
    bail!("native configuration command exited with {}", output.status)
}

fn push_change(changes: &mut Vec<SemanticChange>, setting: &str, old: Option<&Value>, new: &Value) {
    let redacted = credential_shaped(setting);
    changes.push(SemanticChange {
        setting: setting.to_owned(),
        old_value: old.map(|value| {
            if redacted {
                "[redacted]".to_owned()
            } else {
                display_value(value)
            }
        }),
        new_value: if redacted {
            "[redacted]".to_owned()
        } else {
            display_value(new)
        },
    });
}

fn credential_shaped(setting: &str) -> bool {
    let leaf = setting
        .rsplit('.')
        .next()
        .unwrap_or(setting)
        .to_ascii_lowercase();
    matches!(
        leaf.as_str(),
        "apikey"
            | "anthropic_api_key"
            | "anthropic_auth_token"
            | "rocm_local_api_key"
            | "openai-api-key"
    )
}

fn display_value(value: &Value) -> String {
    value
        .as_str()
        .map_or_else(|| value.to_string(), str::to_owned)
}

fn json_string_at(value: &Value, path: &[&str]) -> Option<String> {
    json_at(value, path)
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn json_at<'a>(mut value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    for part in path {
        value = value.get(*part)?;
    }
    Some(value)
}

fn qwen_endpoint(value: &Value, model: &str) -> Option<String> {
    let providers = value.get("modelProviders")?;
    for (_provider, entry) in providers.as_object()? {
        let models = entry
            .as_array()
            .or_else(|| entry.get("models")?.as_array())?;
        for candidate in models {
            if candidate.get("id").and_then(Value::as_str) == Some(model) {
                return candidate
                    .get("baseUrl")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
            }
        }
    }
    None
}
