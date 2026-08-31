// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Lemonade Server collector — pure parsers over the Lemonade REST API.
//!
//! Schema verified against the official docs (lemonade-server.ai/docs/api):
//! - `GET /api/v1/stats` → per-request performance metrics:
//!   `time_to_first_token`, `tokens_per_second`, `input_tokens`, `output_tokens`,
//!   `prompt_tokens`, `decode_token_times[]`. (No KV-cache / running / waiting —
//!   Lemonade does not expose those; those `InstanceSample` fields stay `None`.)
//! - `GET /api/v1/health` → `status` + `model_loaded` (most-recent model name).
//!
//! The parsers (`parse_stats`, `parse_health_model`) are **pure** (serde only)
//! and the deterministic, fixture-tested anchor. The async scrape
//! ([`LemonadeCollector`]) fetches those bodies over HTTP and degrades to
//! "not reachable" (never panics) so a host with no Lemonade endpoint is a no-op.

use std::time::Duration;

use reqwest::Client;
use rocm_dash_core::metrics::Instance;
use rocm_dash_core::traits::{
    CollectorError, DiscoveredService, InstanceSample, Result, merge_instance,
};
use serde::Deserialize;

/// Lemonade's default OpenAI-compatible port. Mirrors `rocm_dash_tui::skills`
/// (defined locally to avoid a cross-crate dep from collectors → tui).
pub const LEMONADE_PORT: u16 = 13305;
/// The runtime-stats path on the canonical `/api/v1` base.
pub const LEMONADE_STATS_PATH: &str = "/api/v1/stats";
/// The health/liveness path on the canonical `/api/v1` base.
pub const LEMONADE_HEALTH_PATH: &str = "/api/v1/health";
const LLAMA_METRICS_PATH: &str = "/metrics";
const LLAMA_KEY_RUNNING: &str = "llamacpp:requests_processing";
const LLAMA_KEY_WAITING: &str = "llamacpp:requests_deferred";
const LLAMA_KEY_GEN_TOKENS: &str = "llamacpp:tokens_predicted_total";
const LLAMA_KEY_GEN_SECONDS: &str = "llamacpp:tokens_predicted_seconds_total";

/// `/api/v1/stats` response — performance metrics from the most recent request.
/// Every field is optional: the endpoint only populates them after an inference.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LemonadeStats {
    #[serde(default)]
    pub time_to_first_token: Option<f64>,
    #[serde(default)]
    pub tokens_per_second: Option<f64>,
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub decode_token_times: Option<Vec<f64>>,
}

/// `/api/v1/health` response — liveness + the most-recently-loaded model name.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LemonadeHealth {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub model_loaded: Option<String>,
}

/// PURE: parse a `/api/v1/stats` body into an [`InstanceSample`].
///
/// Lemonade reports point-in-time request statistics. Encode each latency as
/// one observation so the runner's cumulative-histogram helper always falls
/// back to that request's average instead of differencing unrelated requests.
pub fn parse_stats(body: &str) -> InstanceSample {
    sample_from_stats(parse_stats_struct(body))
}

fn sample_from_stats(stats: LemonadeStats) -> InstanceSample {
    let valid = |value: f64| value.is_finite() && value >= 0.0;
    let ttft = stats.time_to_first_token.filter(|value| valid(*value));
    let gen_tps = stats.tokens_per_second.filter(|value| valid(*value));
    let (decode_sum, decode_count) = stats
        .decode_token_times
        .unwrap_or_default()
        .into_iter()
        .filter(|value| valid(*value))
        .fold((0.0, 0_u64), |(sum, count), value| (sum + value, count + 1));
    let mean_decode_time = (decode_count > 0).then(|| decode_sum / decode_count as f64);
    InstanceSample {
        kv_cache_usage_pct: None,
        running_reqs: None,
        waiting_reqs: None,
        gen_tokens_total: None,
        gen_tps,
        ttft_sum_s: ttft,
        ttft_count: ttft.map(|_| 1.0),
        tpot_sum_s: mean_decode_time,
        tpot_count: mean_decode_time.map(|_| 1.0),
    }
}

/// PURE: parse the structured `/api/v1/stats` body. Returns the full Lemonade
/// metric set (for callers that want TTFT / token counts). Malformed → default.
pub fn parse_stats_struct(body: &str) -> LemonadeStats {
    object_or_default(body)
}

fn parse_stats_response(body: &str) -> Result<InstanceSample> {
    const FIELDS: &[&str] = &[
        "time_to_first_token",
        "tokens_per_second",
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "decode_token_times",
    ];
    let value = known_object(body, FIELDS, "Lemonade stats")?;
    let stats = serde_json::from_value(value)
        .map_err(|e| CollectorError::Parse(format!("Lemonade stats: {e}")))?;
    Ok(sample_from_stats(stats))
}

/// Parse metrics exported by Lemonade's packaged llama.cpp server.
///
/// llama.cpp exposes cumulative generation tokens and seconds plus live queue
/// gauges. The daemon differences the token counter into tok/s and treats
/// generation seconds/tokens as the cumulative TPOT histogram pair.
pub fn parse_llama_metrics(body: &str) -> InstanceSample {
    let gen_tokens_total = extract_prometheus(body, LLAMA_KEY_GEN_TOKENS);
    InstanceSample {
        kv_cache_usage_pct: None,
        running_reqs: extract_prometheus(body, LLAMA_KEY_RUNNING).map(|v| v.round() as u32),
        waiting_reqs: extract_prometheus(body, LLAMA_KEY_WAITING).map(|v| v.round() as u32),
        gen_tokens_total,
        gen_tps: None,
        ttft_sum_s: None,
        ttft_count: None,
        tpot_sum_s: extract_prometheus(body, LLAMA_KEY_GEN_SECONDS),
        tpot_count: gen_tokens_total,
    }
}

fn extract_prometheus(body: &str, metric: &str) -> Option<f64> {
    body.lines().find_map(|line| {
        let line = line.trim_start();
        if line.starts_with('#') {
            return None;
        }
        let rest = line.strip_prefix(metric)?;
        let value = match rest.chars().next() {
            Some('{') => rest.get(rest.find('}')? + 1..)?.trim_start(),
            Some(c) if c.is_whitespace() => rest.trim_start(),
            _ => return None,
        };
        value
            .split_whitespace()
            .next()?
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite() && *value >= 0.0)
    })
}

fn known_object(body: &str, fields: &[&str], schema: &str) -> Result<serde_json::Value> {
    let value: serde_json::Value =
        serde_json::from_str(body).map_err(|e| CollectorError::Parse(format!("{schema}: {e}")))?;
    let Some(object) = value.as_object() else {
        return Err(CollectorError::Parse(format!("{schema}: expected object")));
    };
    if !fields.iter().any(|field| object.contains_key(*field)) {
        return Err(CollectorError::Parse(format!(
            "{schema}: no recognized fields"
        )));
    }
    Ok(value)
}

fn parse_health_response(body: &str) -> Result<Option<String>> {
    let value = known_object(body, &["status", "model_loaded"], "Lemonade health")?;
    let health: LemonadeHealth = serde_json::from_value(value)
        .map_err(|e| CollectorError::Parse(format!("Lemonade health: {e}")))?;
    Ok(health.model_loaded.filter(|model| !model.is_empty()))
}

/// PURE: extract the loaded model name from a `/api/v1/health` body, if present
/// and non-empty. Malformed JSON → `None`, never panics.
pub fn parse_health_model(body: &str) -> Option<String> {
    let health: LemonadeHealth = object_or_default(body);
    health.model_loaded.filter(|s| !s.is_empty())
}

/// Deserialize a JSON **object** body into `T`, returning `T::default()` for any
/// non-object/invalid input. serde's derived `Deserialize` decodes a struct from
/// a positional JSON *array* too — guarding on object shape prevents a wrong-shape
/// body (e.g. `[1,2,3]`) from being misread into fields.
fn object_or_default<T: serde::de::DeserializeOwned + Default>(body: &str) -> T {
    match serde_json::from_str::<serde_json::Value>(body) {
        Ok(value @ serde_json::Value::Object(_)) => {
            serde_json::from_value(value).unwrap_or_default()
        }
        _ => T::default(),
    }
}

/// A stable synthetic id for the local Lemonade endpoint (it is a server, not a
/// container, so it has no Docker id). Distinct from any vLLM/Docker container id.
pub fn lemonade_container_id(host: &str, port: u16) -> String {
    format!("lemonade-{host}-{port}")
}

/// PURE: build a `DiscoveredService` for a Lemonade endpoint. Lemonade is a
/// single local server (no tensor-parallel sharding metadata), so TP = 1.
///
/// Every call site only constructs this after `/api/v1/health` has already
/// answered (see `LemonadeCollector::discover`), so a successfully-built
/// service is by definition reachable and serving — mark it `Ready` rather
/// than the generic `Running`.
pub fn lemonade_service(host: &str, port: u16, model_name: &str) -> DiscoveredService {
    DiscoveredService {
        container_id: lemonade_container_id(host, port),
        container_name: "lemonade".to_string(),
        model_name: if model_name.is_empty() {
            "lemonade".to_string()
        } else {
            model_name.to_string()
        },
        port: Some(port),
        tensor_parallel_size: 1,
        status: rocm_dash_core::metrics::InstanceStatus::Ready,
        ..Default::default()
    }
}

/// PURE: build a finished `Instance` from a Lemonade `/health` model name + a `/stats` body.
///
/// This is the fixture→Instance anchor (no network). `gen_tps` flows from
/// `tokens_per_second`; KV/req fields stay `None` (Lemonade does not report them).
pub fn lemonade_instance(host: &str, port: u16, model_name: &str, stats_body: &str) -> Instance {
    let svc = lemonade_service(host, port, model_name);
    let sample = parse_stats(stats_body);
    merge_instance(&svc, &sample, 0, 0)
}

/// Async scrape of a local Lemonade endpoint. Network/parse failure degrades to
/// `Err`/`None` ("not reachable") — never a panic. The pure parsers above do the
/// actual mapping; this only does I/O.
#[derive(Debug, Clone)]
pub struct LemonadeCollector {
    host: String,
    port: u16,
    client: Client,
}

impl LemonadeCollector {
    pub fn new(host: impl Into<String>, port: u16, timeout: Duration) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            host: host.into(),
            port,
            client,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("http://{}:{}{path}", self.host, self.port)
    }

    async fn get_text(&self, path: &str) -> Result<String> {
        let url = self.url(path);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| CollectorError::Transport(format!("GET {url}: {e}")))?;
        if !resp.status().is_success() {
            return Err(CollectorError::Transport(format!(
                "GET {url}: status {}",
                resp.status()
            )));
        }
        resp.text()
            .await
            .map_err(|e| CollectorError::Transport(format!("body {url}: {e}")))
    }

    /// Scrape `/api/v1/stats` → an `InstanceSample` (rate → `gen_tps`).
    pub async fn fetch_stats(&self) -> Result<InstanceSample> {
        parse_stats_response(&self.get_text(LEMONADE_STATS_PATH).await?)
    }

    /// Scrape `/api/v1/health` → the loaded model name (if any).
    pub async fn fetch_health_model(&self) -> Result<Option<String>> {
        parse_health_response(&self.get_text(LEMONADE_HEALTH_PATH).await?)
    }

    /// Probe the endpoint: if `/api/v1/health` answers, return a `DiscoveredService`
    /// for it (model name from health, falling back to "lemonade"); otherwise
    /// `None` (endpoint absent/unreachable) — a clean no-op, no panic.
    pub async fn discover(&self) -> Option<DiscoveredService> {
        match self.fetch_health_model().await {
            Ok(model) => Some(lemonade_service(
                &self.host,
                self.port,
                model.as_deref().unwrap_or(""),
            )),
            Err(_) => None,
        }
    }
}

/// Scraper for managed Lemonade services.
///
/// Direct GGUF serving uses llama.cpp `/metrics`; routed Lemonade serving uses
/// `/api/v1/stats`. Try them in that order so one managed-service path covers
/// both backends without requiring engine-state coupling in the daemon.
#[derive(Debug, Clone)]
pub struct ManagedLemonadeCollector {
    host: String,
    client: Client,
}

impl ManagedLemonadeCollector {
    pub fn new(host: impl Into<String>, timeout: Duration) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            host: host.into(),
            client,
        }
    }

    async fn get_text(&self, port: u16, path: &str) -> Result<String> {
        let url = format!("http://{}:{port}{path}", self.host);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| CollectorError::Transport(format!("GET {url}: {e}")))?;
        if !resp.status().is_success() {
            return Err(CollectorError::Transport(format!(
                "GET {url}: status {}",
                resp.status()
            )));
        }
        resp.text()
            .await
            .map_err(|e| CollectorError::Transport(format!("body {url}: {e}")))
    }

    pub async fn fetch_async(&self, svc: &DiscoveredService) -> Result<InstanceSample> {
        let port = svc
            .port
            .ok_or_else(|| CollectorError::Unsupported("instance has no port".into()))?;
        let metrics = self.get_text(port, LLAMA_METRICS_PATH).await;
        if let Ok(body) = &metrics {
            let sample = parse_llama_metrics(body);
            if sample.gen_tokens_total.is_some()
                || sample.running_reqs.is_some()
                || sample.waiting_reqs.is_some()
                || sample.tpot_sum_s.is_some()
            {
                return Ok(sample);
            }
        }
        let stats = self.get_text(port, LEMONADE_STATS_PATH).await;
        match stats {
            Ok(body) => parse_stats_response(&body),
            Err(stats_error) => Err(CollectorError::Other(format!(
                "Lemonade metrics unavailable ({}); stats unavailable ({stats_error})",
                metrics
                    .err()
                    .map_or_else(|| "no recognized fields".to_string(), |e| e.to_string())
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Representative `/api/v1/stats` response (lemonade-server.ai/docs/api).
    const STATS_FIXTURE: &str = r#"{
        "time_to_first_token": 2.14,
        "tokens_per_second": 33.33,
        "input_tokens": 128,
        "output_tokens": 5,
        "prompt_tokens": 133,
        "decode_token_times": [0.03, 0.03, 0.03, 0.03, 0.03]
    }"#;

    // Representative `/api/v1/health` response.
    const HEALTH_FIXTURE: &str = r#"{
        "status": "ok",
        "version": "9.3.3",
        "model_loaded": "Llama-3.2-1B-Instruct-Hybrid",
        "all_models_loaded": []
    }"#;

    #[test]
    fn parse_stats_maps_tokens_per_second_to_gen_tps() {
        let sample = parse_stats(STATS_FIXTURE);
        assert_eq!(sample.gen_tps, Some(33.33));
        // Lemonade does not expose KV/queue metrics — they must stay None.
        assert_eq!(sample.kv_cache_usage_pct, None);
        assert_eq!(sample.running_reqs, None);
        assert_eq!(sample.waiting_reqs, None);
        assert_eq!(sample.gen_tokens_total, None);
        assert_eq!(sample.tpot_sum_s, Some(0.03));
        assert_eq!(sample.tpot_count, Some(1.0));
        assert_eq!(sample.ttft_sum_s, Some(2.14));
        assert_eq!(sample.ttft_count, Some(1.0));
    }

    #[test]
    fn parses_packaged_llama_metrics() {
        let sample = parse_llama_metrics(
            "llamacpp:tokens_predicted_total 120\n\
             llamacpp:tokens_predicted_seconds_total 2.4\n\
             llamacpp:requests_processing 3\n\
             llamacpp:requests_deferred 2\n",
        );
        assert_eq!(sample.gen_tokens_total, Some(120.0));
        assert_eq!(sample.running_reqs, Some(3));
        assert_eq!(sample.waiting_reqs, Some(2));
        assert_eq!(sample.tpot_sum_s, Some(2.4));
        assert_eq!(sample.tpot_count, Some(120.0));
        assert_eq!(sample.ttft_sum_s, None);
    }

    #[test]
    fn stale_payloads_are_not_accepted_as_telemetry() {
        assert!(parse_stats_response("{}").is_err());
        assert!(parse_stats_response(r#"{"old_tokens_per_second": 12}"#).is_err());
        assert!(parse_health_response(r#"{"healthy": true}"#).is_err());
    }

    #[test]
    fn parse_stats_struct_captures_full_metric_set() {
        let stats = parse_stats_struct(STATS_FIXTURE);
        assert_eq!(stats.time_to_first_token, Some(2.14));
        assert_eq!(stats.tokens_per_second, Some(33.33));
        assert_eq!(stats.input_tokens, Some(128));
        assert_eq!(stats.output_tokens, Some(5));
        assert_eq!(stats.prompt_tokens, Some(133));
        assert_eq!(stats.decode_token_times.unwrap().len(), 5);
    }

    #[test]
    fn parse_health_model_extracts_loaded_model() {
        assert_eq!(
            parse_health_model(HEALTH_FIXTURE).as_deref(),
            Some("Llama-3.2-1B-Instruct-Hybrid")
        );
    }

    #[test]
    fn malformed_or_empty_input_is_graceful_not_panic() {
        // Garbage / empty / wrong-shape → all-None sample, no panic.
        for body in [
            "",
            "not json",
            "{}",
            "[1,2,3]",
            r#"{"tokens_per_second": "x"}"#,
        ] {
            let sample = parse_stats(body);
            assert_eq!(sample.gen_tps, None, "body {body:?}");
            assert!(sample.kv_cache_usage_pct.is_none());
        }
        assert_eq!(parse_health_model("not json"), None);
        assert_eq!(parse_health_model(r#"{"model_loaded": ""}"#), None);
    }

    #[test]
    fn fixture_stats_plus_health_becomes_an_instance() {
        // The fixture→Instance anchor: no network. Health gives the model name,
        // /stats gives the live rate → an Instances-tab row.
        let model = parse_health_model(HEALTH_FIXTURE).expect("model");
        let inst = lemonade_instance("127.0.0.1", LEMONADE_PORT, &model, STATS_FIXTURE);
        assert_eq!(inst.model_name, "Llama-3.2-1B-Instruct-Hybrid");
        assert_eq!(inst.container_name, "lemonade");
        assert_eq!(inst.container_id, "lemonade-127.0.0.1-13305");
        assert_eq!(inst.port, Some(13305));
        assert_eq!(inst.gen_tps, Some(33.33));
        assert_eq!(inst.status, rocm_dash_core::metrics::InstanceStatus::Ready);
        // Lemonade exposes no KV/req metrics → those stay None on the Instance.
        assert_eq!(inst.kv_cache_usage_pct, None);
        assert_eq!(inst.running_reqs, None);
    }

    #[tokio::test]
    async fn discover_is_clean_noop_when_endpoint_absent() {
        // Probe a port with no server → connection refused → None, no panic.
        // (Port 0 is never a live listener; the GET fails fast.)
        let collector = LemonadeCollector::new("127.0.0.1", 1, Duration::from_millis(200));
        assert!(collector.discover().await.is_none());
        assert!(collector.fetch_stats().await.is_err());
    }

    /// Integration-gated: scrape a REAL local Lemonade server (start it first via
    /// `rocm skill run install-lemonade --apply`). Not run in CI.
    #[tokio::test]
    #[ignore = "requires a running Lemonade server on :13305"]
    async fn live_lemonade_scrape_produces_instance() {
        let collector =
            LemonadeCollector::new("127.0.0.1", LEMONADE_PORT, Duration::from_millis(1500));
        let svc = collector.discover().await.expect("lemonade reachable");
        assert_eq!(svc.port, Some(LEMONADE_PORT));
        // A stats scrape returns a sample (gen_tps may be None until an inference).
        let _sample = collector.fetch_stats().await.expect("stats scrape");
    }
}
