// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `hypercricket.recipe.v1` — a measured serving configuration produced by an external
//! optimizer and replayed by `rocm serve --recipe`.
//!
//! Every value in a recipe was measured together, on one machine, with one engine build.
//! That is why this module reports instead of enforcing: overriding a recipe value from
//! the command line, or running a recipe on a machine whose facts have drifted since the
//! measurement, invalidates the recipe's *numbers* without invalidating its
//! *configuration*. Refusing to serve would strand the user; serving silently would let a
//! measured claim outlive its evidence. So both cases print one line and serve.

use anyhow::{Context, Result, bail};
use rocm_core::AppPaths;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

/// The only recipe schema this build understands.
pub const SCHEMA: &str = "hypercricket.recipe.v1";

/// Sections beyond these (`tuned_for`, `measured`, `quality`) are the optimizer's
/// evidence, not serving input, and are deliberately ignored rather than rejected so a
/// richer producer stays loadable by an older CLI.
#[derive(Debug, Clone, Deserialize)]
pub struct Recipe {
    pub schema: String,
    pub name: String,
    #[serde(default)]
    pub model: RecipeModel,
    #[serde(default)]
    pub engine: RecipeEngine,
    #[serde(default)]
    pub provenance: RecipeProvenance,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RecipeModel {
    #[serde(rename = "ref")]
    pub model_ref: Option<String>,
    pub weights: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RecipeEngine {
    pub name: Option<String>,
    pub build_id: Option<String>,
    pub binary: Option<String>,
    pub device: Option<String>,
    #[serde(default)]
    pub args: BTreeMap<String, ArgValue>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RecipeProvenance {
    pub host_gfx: Option<String>,
    pub rocm_runtime: Option<String>,
}

impl Recipe {
    /// One line naming what this recipe was tuned for and which file answers to it, so a
    /// bare model name on the command line cannot quietly resolve to a different
    /// quantization than the user expects.
    pub fn applied_line(&self) -> String {
        let weights = self.model.weights.as_deref().map_or_else(
            || "<engine default>".to_owned(),
            |weights| expand_tilde(weights).display().to_string(),
        );
        format!(
            "recipe {}: tuned for {}, serving {weights}",
            self.name,
            self.model.model_ref.as_deref().unwrap_or("<any model>")
        )
    }
}

/// A recipe engine-arg value. The TOML type is kept rather than collapsed to a string so
/// a table or array is rejected loudly instead of being flattened into something the
/// engine would misread, and so `true` can mean "switch present" — the TOML-natural
/// spelling for llama.cpp-family flags that take no argument.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum ArgValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl ArgValue {
    /// The engine-argv value for this arg, or `None` when the arg is a switch that the
    /// recipe explicitly turned off (`false`) and must therefore not be passed at all.
    fn as_arg(&self) -> Option<String> {
        match self {
            Self::Bool(false) => None,
            Self::Bool(true) => Some(String::new()),
            other => Some(other.to_string()),
        }
    }
}

impl fmt::Display for ArgValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(value) => write!(f, "{value}"),
            Self::Int(value) => write!(f, "{value}"),
            Self::Float(value) => write!(f, "{value}"),
            Self::Str(value) => write!(f, "{value}"),
        }
    }
}

/// Load a recipe named on the command line. A bare name resolves under the CLI-owned
/// recipes directory; anything that looks like a path is used as one.
pub fn load(paths: &AppPaths, reference: &str) -> Result<Recipe> {
    let path = resolve_path(paths, reference);
    let text = fs::read_to_string(&path)
        .with_context(|| format!("failed to read recipe {}", path.display()))?;
    parse(&text, &path)
}

pub fn resolve_path(paths: &AppPaths, reference: &str) -> PathBuf {
    let looks_like_path = reference.contains('/')
        || reference.contains('\\')
        || Path::new(reference)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"));
    if looks_like_path {
        return expand_tilde(reference);
    }
    paths.recipes_dir().join(format!("{reference}.toml"))
}

pub fn parse(text: &str, source: &Path) -> Result<Recipe> {
    let recipe: Recipe = toml::from_str(text)
        .with_context(|| format!("failed to parse recipe {}", source.display()))?;
    if recipe.schema != SCHEMA {
        bail!(
            "recipe {} declares schema `{}`; this build understands `{SCHEMA}`",
            source.display(),
            recipe.schema
        );
    }
    Ok(recipe)
}

/// Recipes are authored with `~`-relative paths so they survive being copied between
/// machines; the engine receives an absolute path either way.
pub fn expand_tilde(value: &str) -> PathBuf {
    match value
        .strip_prefix("~/")
        .or_else(|| value.strip_prefix("~\\"))
    {
        Some(rest) => match rocm_core::runtime_home_dir() {
            Some(home) => home.join(rest),
            None => PathBuf::from(value),
        },
        None => PathBuf::from(value),
    }
}

/// The machine facts a recipe's measurements depend on. `None` is *unknown*, not
/// *mismatched*: a fact this machine cannot observe is never evidence that the recipe
/// drifted, because a false staleness warning teaches users to ignore the real one.
#[derive(Debug, Clone, Default)]
pub struct HostFacts {
    pub engine_build_id: Option<String>,
    pub gfx: Option<String>,
    pub rocm_runtime: Option<String>,
}

/// Recipe fields whose recorded value disagrees with this machine, named exactly as they
/// are spelled in the recipe file so the warning points at the line that went out of date.
pub fn drifted_fields(recipe: &Recipe, host: &HostFacts) -> Vec<&'static str> {
    [
        (
            "engine.build_id",
            recipe.engine.build_id.as_deref(),
            host.engine_build_id.as_deref(),
        ),
        (
            "provenance.host_gfx",
            recipe.provenance.host_gfx.as_deref(),
            host.gfx.as_deref(),
        ),
        (
            "provenance.rocm_runtime",
            recipe.provenance.rocm_runtime.as_deref(),
            host.rocm_runtime.as_deref(),
        ),
    ]
    .into_iter()
    .filter(|(_, recorded, observed)| {
        matches!((recorded, observed), (Some(left), Some(right)) if left != right)
    })
    .map(|(field, _, _)| field)
    .collect()
}

pub fn staleness_warning(recipe: &Recipe, host: &HostFacts) -> Option<String> {
    let drifted = drifted_fields(recipe, host);
    if drifted.is_empty() {
        return None;
    }
    Some(format!(
        "warning: recipe `{}` is stale: {} no longer match this machine; serving it anyway, \
         but its measured numbers no longer describe this run",
        recipe.name,
        drifted.join(", ")
    ))
}

/// A build id dropped next to the engine binary by whatever built it. rocm-cli cannot
/// derive an engine build id on its own (a locally built `llama-server` has no identity
/// the CLI installed), so the build that produced the binary states it here.
///
/// ponytail: sidecar file, one read, no format. Upgrade to asking the binary itself once
/// engines report a stable id on `--version`.
pub fn engine_build_id(binary: &Path) -> Option<String> {
    let sidecar = binary.with_extension("build_id");
    let text = fs::read_to_string(sidecar).ok()?;
    let id = text.trim().to_owned();
    (!id.is_empty()).then_some(id)
}

/// The engine flag that selects a compute device on the llama.cpp family,
/// spelled with its dashes so the renderer passes it through untouched.
///
/// Distinct from `rocm serve --device`, which is a placement *policy* taking
/// gpu_required|gpu_preferred|cpu_only. llama.cpp spells this one
/// `-dev, --device <dev1,dev2,..>`; a bare `device` key would render `-device`,
/// which the engine rejects.
const ENGINE_DEVICE_ARG: &str = "--device";

/// The recipe-controlled slice of `rocm serve`'s configuration. Every field starts as the
/// explicit command-line value (`None`/empty = not given) and a recipe fills only the gaps.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ServeOverrides {
    pub engine: Option<String>,
    pub device: Option<String>,
    pub binary: Option<String>,
    pub weights: Option<String>,
    pub args: BTreeMap<String, String>,
}

impl ServeOverrides {
    /// Fill unset fields from `recipe`, returning one line per command-line value that
    /// displaced a recipe value. Only overrides are reported: they are the moment the
    /// running configuration stops being the configuration that was measured.
    pub fn merge_recipe(&mut self, recipe: &Recipe) -> Vec<String> {
        let mut overrides = Vec::new();
        for (field, slot, recipe_value) in [
            ("--engine", &mut self.engine, recipe.engine.name.as_deref()),
            (
                "--engine-binary",
                &mut self.binary,
                recipe.engine.binary.as_deref(),
            ),
        ] {
            let Some(recipe_value) = recipe_value else {
                continue;
            };
            match slot.as_deref() {
                None => *slot = Some(recipe_value.to_owned()),
                Some(supplied) if supplied != recipe_value => {
                    overrides.push(override_line(field, supplied, recipe_value));
                }
                Some(_) => {}
            }
        }
        // Weights have no command-line counterpart: the positional model names what
        // clients ask for, the recipe says which file answers to that name.
        if self.weights.is_none() {
            self.weights = recipe.model.weights.clone();
        }
        // `[engine] device` is an ENGINE device selector (`ROCm0`, `Vulkan0`), not
        // `rocm serve --device`, which is a placement policy taking
        // gpu_required|gpu_preferred|cpu_only. Feeding one to the other fails with
        // "unsupported device policy: ROCm0". The selector is an engine flag, so it
        // rides the same passthrough as every other tuned argument.
        if let Some(selector) = recipe.engine.device.as_deref()
            && !selector.is_empty()
        {
            match self.args.get(ENGINE_DEVICE_ARG) {
                None => {
                    self.args
                        .insert(ENGINE_DEVICE_ARG.to_owned(), selector.to_owned());
                }
                Some(supplied) if supplied != selector => {
                    overrides.push(override_line(
                        &format!("--engine-arg {ENGINE_DEVICE_ARG}"),
                        supplied,
                        selector,
                    ));
                }
                Some(_) => {}
            }
        }
        for (key, value) in &recipe.engine.args {
            let Some(recipe_value) = value.as_arg() else {
                continue;
            };
            match self.args.get(key) {
                None => {
                    self.args.insert(key.clone(), recipe_value);
                }
                Some(supplied) if *supplied != recipe_value => {
                    overrides.push(override_line(
                        &format!("--engine-arg {key}"),
                        supplied,
                        &recipe_value,
                    ));
                }
                Some(_) => {}
            }
        }
        overrides
    }
}

fn override_line(field: &str, supplied: &str, recipe_value: &str) -> String {
    let shown = if supplied.is_empty() {
        "<set>"
    } else {
        supplied
    };
    format!(
        "recipe override: {field} = {shown} replaces recipe value {recipe_value}; the recipe's measured numbers no longer describe this run"
    )
}

/// Render `KEY=VAL` engine args into engine argv.
///
/// A key that already carries its dashes is passed through untouched; a bare key
/// gets exactly one dash (`fa` -> `-fa`, `ub` -> `-ub`, `ngl` -> `-ngl`). No length
/// heuristic works here: llama.cpp's tuning flags are single-dash short options of
/// one to three characters, so spelling `fa` as `--fa` produces
/// `error: invalid argument: --fa`. Write the dashes yourself
/// (`--ctx-size=8192`) for a GNU-style long option. An empty value renders the
/// flag alone, for valueless switches.
///
/// `rocm bench run` renders through this same function. That is the whole point:
/// a recipe promises that what was measured is what gets served, and two
/// renderers that disagree about how to spell a flag break that promise in the
/// one place nobody would think to check.
pub fn engine_argv(args: &BTreeMap<String, String>) -> Vec<String> {
    let mut argv = Vec::with_capacity(args.len() * 2);
    for (key, value) in args {
        argv.push(if key.starts_with('-') {
            key.clone()
        } else {
            format!("-{key}")
        });
        if !value.is_empty() {
            argv.push(value.clone());
        }
    }
    argv
}

/// Parse one `--engine-arg KEY=VAL`. A bare `KEY` is a valueless switch.
pub fn parse_engine_arg(raw: &str) -> Result<(String, String), String> {
    let (key, value) = raw.split_once('=').unwrap_or((raw, ""));
    let key = key.trim();
    if key.is_empty() {
        return Err(format!(
            "engine arg `{raw}` has an empty key; expected KEY=VAL"
        ));
    }
    Ok((key.to_owned(), value.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
schema = "hypercricket.recipe.v1"
name   = "qwen3-8b-interactive"

[model]
ref     = "Qwen/Qwen3-8B"
weights = "/models/qwen3-8b-ROCMFP4_FAST.gguf"
format  = "Q4_0_ROCMFP4_FAST"
sha256  = "3f9c"

[engine]
name     = "lemonade"
build_id = "rocmfpx-b1305"
binary   = "/engines/rocmfpx/bin/llama-server"
device   = "Vulkan0"

[engine.args]
fa                 = "on"
c                  = 8192
ngl                = 999
"spec-draft-p-min" = 0.55
"no-mmap"          = true
"no-warmup"        = false

[tuned_for]
profile = "interactive"

[measured]
output_tok_s = 84.6

[provenance]
host_gfx     = "gfx1201"
rocm_runtime = "nightly-7-14-0"
"#;

    fn sample() -> Recipe {
        parse(SAMPLE, Path::new("sample.toml")).expect("sample recipe parses")
    }

    #[test]
    fn parse_reads_serving_fields_and_ignores_evidence_sections() {
        let recipe = sample();
        assert_eq!(recipe.name, "qwen3-8b-interactive");
        assert_eq!(recipe.model.model_ref.as_deref(), Some("Qwen/Qwen3-8B"));
        assert_eq!(
            recipe.applied_line(),
            "recipe qwen3-8b-interactive: tuned for Qwen/Qwen3-8B, serving /models/qwen3-8b-ROCMFP4_FAST.gguf"
        );
        assert_eq!(recipe.engine.name.as_deref(), Some("lemonade"));
        assert_eq!(recipe.engine.build_id.as_deref(), Some("rocmfpx-b1305"));
        assert_eq!(recipe.provenance.host_gfx.as_deref(), Some("gfx1201"));
        assert_eq!(recipe.engine.args["c"], ArgValue::Int(8192));
        assert_eq!(recipe.engine.args["fa"], ArgValue::Str("on".to_owned()));
        assert_eq!(
            recipe.engine.args["spec-draft-p-min"],
            ArgValue::Float(0.55)
        );
    }

    #[test]
    fn parse_rejects_a_foreign_schema() {
        let text = SAMPLE.replace("hypercricket.recipe.v1", "hypercricket.recipe.v2");
        let error = parse(&text, Path::new("sample.toml")).expect_err("schema is checked");
        assert!(
            error.to_string().contains("hypercricket.recipe.v2"),
            "error should name the unknown schema: {error}"
        );
    }

    #[test]
    fn parse_rejects_a_structured_engine_arg() {
        // A table would otherwise be flattened into argv the engine misreads.
        let text = SAMPLE.replace("ngl                = 999", "ngl = { value = 999 }");
        assert!(parse(&text, Path::new("sample.toml")).is_err());
    }

    #[test]
    fn merge_recipe_fills_unset_fields() {
        let mut overrides = ServeOverrides::default();
        assert!(overrides.merge_recipe(&sample()).is_empty());
        assert_eq!(overrides.engine.as_deref(), Some("lemonade"));
        assert_eq!(
            overrides.args["--device"], "Vulkan0",
            "engine selector must ride the passthrough"
        );
        assert_eq!(
            overrides.device, None,
            "serve --device is a policy and must stay unset"
        );
        assert_eq!(
            overrides.binary.as_deref(),
            Some("/engines/rocmfpx/bin/llama-server")
        );
        assert_eq!(
            overrides.weights.as_deref(),
            Some("/models/qwen3-8b-ROCMFP4_FAST.gguf")
        );
        assert_eq!(overrides.args["c"], "8192");
        assert_eq!(overrides.args["fa"], "on");
        assert_eq!(overrides.args["spec-draft-p-min"], "0.55");
        // `true` is a valueless switch; `false` leaves the switch off entirely.
        assert_eq!(overrides.args["no-mmap"], "");
        assert!(!overrides.args.contains_key("no-warmup"));
    }

    #[test]
    fn merge_recipe_keeps_command_line_values_and_reports_them() {
        let mut overrides = ServeOverrides {
            engine: Some("vllm".to_owned()),
            args: BTreeMap::from([("c".to_owned(), "4096".to_owned())]),
            ..ServeOverrides::default()
        };
        let reported = overrides.merge_recipe(&sample());

        assert_eq!(overrides.engine.as_deref(), Some("vllm"));
        assert_eq!(overrides.args["c"], "4096");
        // Unrelated recipe values still apply.
        assert_eq!(
            overrides.args["--device"], "Vulkan0",
            "engine selector must ride the passthrough"
        );
        assert_eq!(
            overrides.device, None,
            "serve --device is a policy and must stay unset"
        );
        assert_eq!(overrides.args["ngl"], "999");

        assert_eq!(reported.len(), 2, "both overrides reported: {reported:?}");
        assert!(
            reported
                .iter()
                .any(|line| line.contains("--engine = vllm") && line.contains("lemonade")),
            "{reported:?}"
        );
        assert!(
            reported
                .iter()
                .any(|line| line.contains("--engine-arg c = 4096") && line.contains("8192")),
            "{reported:?}"
        );
    }

    #[test]
    fn merge_recipe_reports_nothing_when_the_command_line_agrees() {
        let mut overrides = ServeOverrides {
            engine: Some("lemonade".to_owned()),
            ..ServeOverrides::default()
        };
        assert!(overrides.merge_recipe(&sample()).is_empty());
    }

    #[test]
    fn drift_names_the_stale_build_id() {
        let host = HostFacts {
            engine_build_id: Some("rocmfpx-b1400".to_owned()),
            gfx: Some("gfx1201".to_owned()),
            rocm_runtime: Some("nightly-7-14-0".to_owned()),
        };
        assert_eq!(drifted_fields(&sample(), &host), ["engine.build_id"]);
        let warning = staleness_warning(&sample(), &host).expect("stale recipe warns");
        assert!(warning.contains("engine.build_id"), "{warning}");
        assert!(warning.contains("qwen3-8b-interactive"), "{warning}");
    }

    #[test]
    fn drift_names_the_stale_gfx() {
        let host = HostFacts {
            engine_build_id: Some("rocmfpx-b1305".to_owned()),
            gfx: Some("gfx1100".to_owned()),
            rocm_runtime: Some("nightly-7-14-0".to_owned()),
        };
        assert_eq!(drifted_fields(&sample(), &host), ["provenance.host_gfx"]);
    }

    #[test]
    fn drift_reports_every_field_that_moved() {
        let host = HostFacts {
            engine_build_id: Some("rocmfpx-b1400".to_owned()),
            gfx: Some("gfx1100".to_owned()),
            rocm_runtime: Some("nightly-8-0-0".to_owned()),
        };
        assert_eq!(
            drifted_fields(&sample(), &host),
            [
                "engine.build_id",
                "provenance.host_gfx",
                "provenance.rocm_runtime"
            ]
        );
    }

    #[test]
    fn an_unobservable_fact_is_never_drift() {
        // Nothing observed at all: a recipe cannot be called stale on no evidence.
        assert!(drifted_fields(&sample(), &HostFacts::default()).is_empty());
        assert!(staleness_warning(&sample(), &HostFacts::default()).is_none());
    }

    #[test]
    fn matching_facts_are_not_stale() {
        let host = HostFacts {
            engine_build_id: Some("rocmfpx-b1305".to_owned()),
            gfx: Some("gfx1201".to_owned()),
            rocm_runtime: Some("nightly-7-14-0".to_owned()),
        };
        assert!(drifted_fields(&sample(), &host).is_empty());
    }

    #[test]
    fn engine_argv_gives_every_bare_key_exactly_one_dash() {
        // llama.cpp's tuning flags are single-dash short options of one to three
        // characters. A length heuristic spells `fa` as `--fa`, which the engine
        // rejects outright with `error: invalid argument: --fa`.
        let args = BTreeMap::from([
            ("c".to_owned(), "8192".to_owned()),
            ("fa".to_owned(), "on".to_owned()),
            ("ngl".to_owned(), "999".to_owned()),
            ("no-mmap".to_owned(), String::new()),
            ("--already-dashed".to_owned(), "1".to_owned()),
        ]);
        assert_eq!(
            engine_argv(&args),
            [
                "--already-dashed",
                "1",
                "-c",
                "8192",
                "-fa",
                "on",
                "-ngl",
                "999",
                "-no-mmap",
            ]
        );
    }

    #[test]
    fn bench_and_serve_spell_a_recipe_identically() {
        // The recipe's only promise is that what was measured is what gets served.
        // `rocm bench run` renders through this same function; if that ever stops
        // being true, the promise breaks silently.
        let measured = BTreeMap::from([
            ("c".to_owned(), "4096".to_owned()),
            ("fa".to_owned(), "on".to_owned()),
            ("ngl".to_owned(), "999".to_owned()),
            ("--device".to_owned(), "ROCm0".to_owned()),
        ]);
        assert_eq!(
            engine_argv(&measured),
            [
                "--device", "ROCm0", "-c", "4096", "-fa", "on", "-ngl", "999"
            ]
        );
    }

    #[test]
    fn engine_arg_parsing_splits_on_the_first_equals() {
        assert_eq!(
            parse_engine_arg("c=8192"),
            Ok(("c".to_owned(), "8192".to_owned()))
        );
        // Values may contain `=` themselves (chat templates, key/value payloads).
        assert_eq!(
            parse_engine_arg("override-kv=tokenizer.ggml.pre=str:llama3"),
            Ok((
                "override-kv".to_owned(),
                "tokenizer.ggml.pre=str:llama3".to_owned()
            ))
        );
        assert_eq!(
            parse_engine_arg("no-mmap"),
            Ok(("no-mmap".to_owned(), String::new()))
        );
        assert!(parse_engine_arg("=8192").is_err());
    }

    #[test]
    fn a_bare_name_resolves_under_the_cli_recipes_directory() {
        let paths = AppPaths {
            config_dir: PathBuf::from("/cfg"),
            data_dir: PathBuf::from("/data"),
            cache_dir: PathBuf::from("/cache"),
        };
        assert_eq!(
            resolve_path(&paths, "qwen3-8b-interactive"),
            PathBuf::from("/cfg/recipes/qwen3-8b-interactive.toml")
        );
        // Dotted names are still names, not paths.
        assert_eq!(
            resolve_path(&paths, "qwen3.5-4b"),
            PathBuf::from("/cfg/recipes/qwen3.5-4b.toml")
        );
        assert_eq!(
            resolve_path(&paths, "./tuned.toml"),
            PathBuf::from("./tuned.toml")
        );
        assert_eq!(
            resolve_path(&paths, "tuned.toml"),
            PathBuf::from("tuned.toml")
        );
    }

    #[test]
    fn engine_build_id_reads_the_sidecar_next_to_the_binary() {
        let dir = std::env::temp_dir().join(format!("rocm-recipe-build-id-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir");
        let binary = dir.join("llama-server");
        fs::write(&binary, b"#!/bin/true\n").expect("binary stub");
        assert_eq!(engine_build_id(&binary), None);
        fs::write(dir.join("llama-server.build_id"), "  rocmfpx-b1305\n").expect("sidecar");
        assert_eq!(engine_build_id(&binary), Some("rocmfpx-b1305".to_owned()));
        fs::remove_dir_all(&dir).ok();
    }
}
