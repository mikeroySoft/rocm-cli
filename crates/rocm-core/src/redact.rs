// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Removing secrets and identity from anything a user might paste into a
//! public issue.
//!
//! # Two lines of defence, in this order
//!
//! 1. **An allowlist decides what is collected at all.** A support bundle names
//!    the files and the fields it contains; anything not named is absent. That
//!    is the line that actually holds, because it fails closed: a field added to
//!    a config struct next year is not exported until someone declares it.
//! 2. **This module scrubs what did get collected.** Free text — log lines,
//!    error messages, command output — cannot be allowlisted field by field, so
//!    it is filtered. That fails *open* by nature: a shape nobody anticipated
//!    survives it.
//!
//! Anything that relies on step 2 alone is a leak waiting for a new secret
//! format.
//!
//! # Over-redaction is also a bug
//!
//! A bundle whose every string reads `[redacted]` tells a support engineer
//! nothing and the user files the issue anyway, with a screenshot. So the rules
//! are narrow and their *negative* cases are tested as carefully as the positive
//! ones: `active_runtime_key`, `key_source`, `public_key_pem`, and
//! `tokens_per_second` must all survive intact.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

/// What every removed value is replaced with. One constant, so a test can
/// assert a whole document contains no secret by looking for what is left.
pub const PLACEHOLDER: &str = "[redacted]";

/// Shortest user name or host name this will substitute.
///
/// A two-character name (`ab`, or a user literally called `mike` on a machine
/// called `mi`) matches so much ordinary text that the result is unreadable.
/// Below this length identity is left in place and the omission is reported
/// rather than silently traded for a mangled document.
const MIN_IDENTITY_LEN: usize = 3;

// ---------------------------------------------------------------------------
// Key classification
// ---------------------------------------------------------------------------

/// Field names whose *value* is never exportable, whatever its type.
const SENSITIVE_EXACT: &[&str] = &[
    "api_key",
    "apikey",
    "auth",
    "auth_header",
    "authorization",
    "bearer",
    "boot_id",
    "chat_api_key",
    "chat_auth_header",
    "cookie",
    "credential",
    "credentials",
    "endpoint_key",
    "host_name",
    "hostname",
    "machine_id",
    "password",
    "provider_key",
    "secret",
    "session_id",
    "set_cookie",
    "token",
    "user",
    "user_name",
    "username",
];

/// Suffixes that make a field name sensitive however it is prefixed.
///
/// Suffix rather than substring so `active_runtime_key` and `key_source`
/// survive: neither *ends* in a credential word.
const SENSITIVE_SUFFIX: &[&str] = &[
    "_api_key",
    "_apikey",
    "_auth_header",
    "_authorization",
    "_bearer",
    "_cookie",
    "_credential",
    "_password",
    "_secret",
    "_token",
];

/// Normalise a field name to lowercase `snake_case`, so `chatApiKey`,
/// `chat-api-key`, and `chat_api_key` classify identically.
fn normalise_key(key: &str) -> String {
    let mut out = String::with_capacity(key.len() + 4);
    let mut previous_lower = false;
    for ch in key.chars() {
        if ch == '-' || ch == ' ' || ch == '.' {
            out.push('_');
            previous_lower = false;
        } else if ch.is_ascii_uppercase() {
            if previous_lower {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            previous_lower = false;
        } else {
            out.push(ch);
            previous_lower = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        }
    }
    out
}

/// Whether a field's value must be removed because of what the field is called.
#[must_use]
pub fn key_is_sensitive(key: &str) -> bool {
    let key = normalise_key(key);
    SENSITIVE_EXACT.contains(&key.as_str())
        || SENSITIVE_SUFFIX.iter().any(|s| key.ends_with(s))
        || key.contains("api_key")
}

/// Environment variables that may appear in an exported document.
///
/// Exactly the set the diagnosis catalog reads, and nothing else. An
/// environment map is the classic accidental credential dump — one
/// `AWS_SECRET_ACCESS_KEY` in a pasted bundle is a breach — so this is an
/// allowlist, never a denylist. `PATH` and `LD_LIBRARY_PATH` are on it because
/// `fix-6` is *about* `PATH`; their home-directory components are rewritten by
/// [`Redactor::text`] rather than the whole variable dropped.
const EXPORTABLE_ENV: &[&str] = &[
    "AMDGPU_TARGETS",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HCC_AMDGPU_TARGET",
    "HIP_PATH",
    "HIP_PLATFORM",
    "HIP_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "LD_LIBRARY_PATH",
    "PATH",
    "PYTORCH_ROCM_ARCH",
    "ROCM_HOME",
    "ROCM_PATH",
    "ROCR_VISIBLE_DEVICES",
];

/// Whether an environment variable may be exported at all.
#[must_use]
pub fn env_is_exportable(name: &str) -> bool {
    EXPORTABLE_ENV.contains(&name)
}

// ---------------------------------------------------------------------------
// Text rules
// ---------------------------------------------------------------------------

/// `Authorization: Bearer …`, `Cookie: …`, `X-API-Key: …`.
///
/// The value runs to the closing quote or the end of the line, not to the next
/// space. A header value has internal structure — `Bearer <token>`, or a
/// semicolon-separated cookie jar where *every* entry is a secret — so a
/// one-token replacement leaves the credential sitting next to the word that
/// was removed.
static HEADER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)\b(authorization|proxy-authorization|set-cookie|cookie|x-api-key|api-key)("?\s*[:=]\s*"?)[^"\r\n]*"#,
    )
    .expect("header pattern")
});

/// `api_key=…`, `"token": "…"`, `chat_auth_header: …`.
///
/// Two deliberate choices. The name must *end* in a credential word immediately
/// before the separator, which is what keeps `tokens_per_second: 42` and
/// `active_runtime_key=nightly` out of it. And the value may contain spaces, so
/// a scheme-prefixed value is consumed whole; the cost is that a credential
/// assignment followed by unrelated `k=v` pairs on one line takes them with it,
/// which is the right direction to err.
static ASSIGNMENT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)\b([a-z0-9_.-]*(?:api[_-]?key|token|secret|password|credential|auth[_-]?header|bearer))("?\s*[:=]\s*"?)[^",;}\]\r\n]*"#,
    )
    .expect("assignment pattern")
});

/// Credential shapes that carry no field name at all.
static OPAQUE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?x)
        \b(?:sk|rk)-[A-Za-z0-9_-]{16,}
      | \bhf_[A-Za-z0-9]{16,}
      | \bgh[pousr]_[A-Za-z0-9]{20,}
      | \bxox[baprs]-[A-Za-z0-9-]{10,}
      | \bAKIA[0-9A-Z]{16}\b
      | \beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}
      | (?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}
        ",
    )
    .expect("opaque pattern")
});

/// Prose that names a credential and then just prints it: `stored token
/// AbCd1234…`, `using api key AbCd1234…`.
///
/// Found by a planted-secret audit of a real bundle, where an audit-log line
/// read `stored token <24 chars>` and cleared every other rule: there is no
/// `=` for [`ASSIGNMENT`] and no recognisable prefix for [`OPAQUE`].
///
/// The value must be **at least 16 unbroken credential-shaped characters**, so
/// `token expired`, `token refresh failed`, and `password required` survive
/// intact. That length threshold is the whole reason this rule is safe to apply
/// to prose.
static NAMED_ADJACENT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(api[ _-]?key|token|secret|password|credential)(\s+)[A-Za-z0-9_.+/=-]{16,}\b",
    )
    .expect("adjacent pattern")
});

/// `scheme://user:pass@host` — credentials inside a URL's authority.
static URL_USERINFO: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s@]+@").expect("userinfo"));

/// Everything after `?` or `#` in a URL. A pre-signed download URL *is* the
/// credential, so the query is removed whole rather than parsed for known
/// parameter names.
static URL_QUERY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"([a-zA-Z][a-zA-Z0-9+.-]*://[^\s`'\x22<>]*?)[?#][^\s`'\x22<>]*")
        .expect("query pattern")
});

/// Removes secrets and identity from text and JSON.
///
/// Identity is instance data, not a compile-time constant, so it is carried
/// here rather than read inside [`Redactor::text`]. That also makes the whole
/// module testable: a test names a home directory and a user, and the result is
/// the same on every machine.
#[derive(Debug, Clone, Default)]
pub struct Redactor {
    home: Vec<String>,
    user: Option<String>,
    host: Option<String>,
    /// Identity this refused to substitute because it was too short to
    /// substitute safely. Reported, never silently dropped.
    pub skipped_identity: Vec<String>,
}

impl Redactor {
    /// Build from explicit identity. `home` may name several roots (a POSIX
    /// home plus a Windows profile, or a symlinked one).
    #[must_use]
    pub fn with(home: &[&str], user: Option<&str>, host: Option<&str>) -> Self {
        let mut skipped = Vec::new();
        let mut usable = |value: Option<&str>| -> Option<String> {
            let value = value?.trim();
            if value.len() < MIN_IDENTITY_LEN {
                if !value.is_empty() {
                    skipped.push(value.to_owned());
                }
                return None;
            }
            Some(value.to_owned())
        };
        let user = usable(user);
        let host = usable(host);
        // Longest first: a nested root must not be rewritten by its parent and
        // leave a stub behind.
        let mut home: Vec<String> = home
            .iter()
            .map(|h| h.trim_end_matches(['/', '\\']).to_owned())
            .filter(|h| h.len() >= MIN_IDENTITY_LEN)
            .collect();
        home.sort_by_key(|h| std::cmp::Reverse(h.len()));
        Self {
            home,
            user,
            host,
            skipped_identity: skipped,
        }
    }

    /// Build from this host's own identity.
    #[must_use]
    pub fn from_host() -> Self {
        let home = crate::runtime::runtime_home_dir().map(|p| p.display().to_string());
        let user = std::env::var("USER")
            .or_else(|_| std::env::var("LOGNAME"))
            .or_else(|_| std::env::var("USERNAME"))
            .ok();
        let host = hostname_from(|name| std::env::var(name).ok());
        let roots: Vec<&str> = home.as_deref().into_iter().collect();
        Self::with(&roots, user.as_deref(), host.as_deref())
    }

    /// Scrub one string.
    ///
    /// Rule order is load-bearing: named and shaped credentials go first, so a
    /// token that happens to look like a path is already gone before the
    /// home-directory rewrite could make it look ordinary.
    #[must_use]
    pub fn text(&self, input: &str) -> String {
        // Shaped credentials before named ones: an `api_key=` value already
        // reduced to the placeholder is harmless, but a bare `sk-…` left beside
        // a removed field name is not.
        let out = HEADER.replace_all(input, format!("${{1}}${{2}}{PLACEHOLDER}"));
        let out = OPAQUE.replace_all(&out, PLACEHOLDER);
        let out = ASSIGNMENT.replace_all(&out, format!("${{1}}${{2}}{PLACEHOLDER}"));
        let out = NAMED_ADJACENT.replace_all(&out, format!("${{1}}${{2}}{PLACEHOLDER}"));
        let out = URL_USERINFO.replace_all(&out, format!("${{1}}{PLACEHOLDER}@"));
        let mut out = URL_QUERY
            .replace_all(&out, format!("${{1}}?{PLACEHOLDER}"))
            .into_owned();

        for root in &self.home {
            // Both separators: a Windows profile path reaches logs in either
            // form depending on which layer printed it.
            out = out.replace(root, "~");
            out = out.replace(&root.replace('\\', "/"), "~");
        }
        if let Some(user) = &self.user {
            out = replace_word(&out, user, "[user]");
        }
        if let Some(host) = &self.host {
            out = replace_word(&out, host, "[host]");
        }
        out
    }

    /// Scrub a JSON document in place.
    ///
    /// Field-name classification wins over content: a value under a sensitive
    /// key is removed whatever it looks like, because an empty-looking token is
    /// still a token.
    pub fn json(&self, value: &mut Value) {
        match value {
            Value::String(text) => *text = self.text(text),
            Value::Array(items) => {
                for item in items {
                    self.json(item);
                }
            }
            Value::Object(map) => {
                let keys: Vec<String> = map.keys().cloned().collect();
                for key in keys {
                    if key_is_sensitive(&key) {
                        map.insert(key, Value::String(PLACEHOLDER.to_owned()));
                        continue;
                    }
                    if normalise_key(&key) == "env" {
                        if let Some(env) = map.get_mut(&key).and_then(Value::as_object_mut) {
                            let dropped: Vec<String> = env
                                .keys()
                                .filter(|name| !env_is_exportable(name))
                                .cloned()
                                .collect();
                            for name in dropped {
                                env.remove(&name);
                            }
                            for entry in env.values_mut() {
                                self.json(entry);
                            }
                        }
                        continue;
                    }
                    if let Some(entry) = map.get_mut(&key) {
                        self.json(entry);
                    }
                }
            }
            Value::Null | Value::Bool(_) | Value::Number(_) => {}
        }
    }

    /// Scrub a serializable value, returning JSON.
    pub fn value<T: serde::Serialize>(&self, value: &T) -> Result<Value, serde_json::Error> {
        let mut json = serde_json::to_value(value)?;
        self.json(&mut json);
        Ok(json)
    }
}

/// Resolve this machine's name from an environment lookup, falling back to
/// the Linux kernel file.
///
/// `COMPUTERNAME` is in the chain because native Windows sets neither
/// `HOSTNAME` nor `/proc/sys/kernel/hostname`; without it every Windows
/// support bundle leaked the machine name. The lookup is a parameter so the
/// Windows path is testable without mutating process-global environment.
fn hostname_from(env: impl Fn(&str) -> Option<String>) -> Option<String> {
    env("HOSTNAME")
        .or_else(|| env("COMPUTERNAME"))
        .or_else(|| std::fs::read_to_string("/proc/sys/kernel/hostname").ok())
        .map(|h| h.trim().to_owned())
        .filter(|h| !h.is_empty())
}

/// Replace `needle` only where it is a whole token.
///
/// A plain `replace` on a user name of `pi` rewrites every `pi` in the
/// document, including `mapi`, `pip`, and `PYTORCH_ROCM_ARCH=gfx1201`. Word
/// boundaries here are "not an ASCII alphanumeric or underscore", which keeps
/// `/home/mike/x` and `mike@host` matching while leaving `mikeroysoft` alone.
fn replace_word(haystack: &str, needle: &str, replacement: &str) -> String {
    let boundary = |c: Option<char>| c.is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_');
    let mut out = String::with_capacity(haystack.len());
    let mut rest = haystack;
    while let Some(at) = rest.find(needle) {
        let before = rest[..at].chars().next_back();
        let after = rest[at + needle.len()..].chars().next();
        out.push_str(&rest[..at]);
        if boundary(before) && boundary(after) {
            out.push_str(replacement);
        } else {
            out.push_str(needle);
        }
        rest = &rest[at + needle.len()..];
    }
    out.push_str(rest);
    out
}

/// Keep only the exportable environment variables, scrubbing their values.
#[must_use]
pub fn exportable_env(
    redactor: &Redactor,
    env: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    env.iter()
        .filter(|(name, _)| env_is_exportable(name))
        .map(|(name, value)| (name.clone(), redactor.text(value)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn redactor() -> Redactor {
        Redactor::with(
            &["/home/alice", "C:\\Users\\alice"],
            Some("alice"),
            Some("workstation"),
        )
    }

    // -----------------------------------------------------------------------
    // Key classification
    // -----------------------------------------------------------------------

    #[test]
    fn redact_classifies_credential_field_names() {
        for key in [
            "api_key",
            "apiKey",
            "API-KEY",
            "openai_api_key",
            "token",
            "hf_token",
            "chat_auth_header",
            "authorization",
            "Cookie",
            "password",
            "endpoint_key",
            "provider_key",
            "session_id",
            "user_name",
            "hostname",
            "boot_id",
        ] {
            assert!(key_is_sensitive(key), "{key} should be sensitive");
        }
    }

    /// The other half of the rule, and the one that makes a bundle useful. A
    /// redactor that eats these produces a document nobody can diagnose from.
    #[test]
    fn redact_leaves_ordinary_field_names_alone() {
        for key in [
            "active_runtime_key",
            "previous_runtime_key",
            "runtime_key",
            "key_source",
            "public_key_pem",
            "tokens_per_second",
            "keyboard_layout",
            "author",
            "authoritative",
            "rocm_version",
            "install_root",
            "gpu_utilization_pct",
        ] {
            assert!(!key_is_sensitive(key), "{key} must not be redacted");
        }
    }

    #[test]
    fn redact_environment_allowlist_is_closed() {
        for name in [
            "PATH",
            "LD_LIBRARY_PATH",
            "HSA_OVERRIDE_GFX_VERSION",
            "ROCM_PATH",
        ] {
            assert!(env_is_exportable(name), "{name} is diagnostic");
        }
        for name in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "HF_TOKEN",
            "AWS_SECRET_ACCESS_KEY",
            "HOME",
            "USER",
            "HOSTNAME",
            "SSH_AUTH_SOCK",
            "GITHUB_TOKEN",
            "ROCM_SERVE_API_KEY",
        ] {
            assert!(!env_is_exportable(name), "{name} must never be exported");
        }
    }

    // -----------------------------------------------------------------------
    // Text
    // -----------------------------------------------------------------------

    #[test]
    fn redact_removes_authorization_and_cookie_headers() {
        let scrubbed = redactor().text(
            "GET /v1/models\nAuthorization: Bearer sk-proj-abcdefghijklmnopqrstuvwxyz012345\n\
             Cookie: session=9f8e7d6c5b4a; theme=dark\nX-API-Key: 5f4e3d2c1b0a9988",
        );
        assert!(!scrubbed.contains("sk-proj"), "{scrubbed}");
        assert!(!scrubbed.contains("9f8e7d6c5b4a"), "{scrubbed}");
        assert!(!scrubbed.contains("5f4e3d2c1b0a9988"), "{scrubbed}");
        // The header names survive: knowing an Authorization header was sent is
        // the diagnostic fact.
        assert!(scrubbed.contains("Authorization"));
        assert!(scrubbed.contains("GET /v1/models"));
    }

    #[test]
    fn redact_removes_named_assignments_in_any_syntax() {
        for line in [
            "api_key=sk-live-1234567890abcdef",
            "\"token\": \"ghp_aaaabbbbccccddddeeeeffff\"",
            "OPENAI_API_KEY=sk-proj-zzzzzzzzzzzzzzzzzzzz",
            "chat_auth_header: Bearer abcdefghijklmnop",
            "password = hunter2hunter2",
            "provider_secret=abc123def456",
        ] {
            let scrubbed = redactor().text(line);
            assert!(
                scrubbed.contains(PLACEHOLDER),
                "no placeholder in {scrubbed}"
            );
            for secret in [
                "sk-live-1234567890abcdef",
                "ghp_aaaabbbbccccddddeeeeffff",
                "sk-proj-zzzzzzzzzzzzzzzzzzzz",
                "abcdefghijklmnop",
                "hunter2hunter2",
                "abc123def456",
            ] {
                assert!(
                    !scrubbed.contains(secret),
                    "{secret} survived in {scrubbed}"
                );
            }
        }
    }

    /// The negative case that matters most: a field whose name merely contains
    /// a credential word must keep its value.
    #[test]
    fn redact_keeps_values_whose_names_only_look_credential_like() {
        let scrubbed = redactor()
            .text("tokens_per_second=1420 active_runtime_key=nightly-wheel-gfx120x-all-7-14-0");
        assert!(scrubbed.contains("1420"), "{scrubbed}");
        assert!(
            scrubbed.contains("nightly-wheel-gfx120x-all-7-14-0"),
            "{scrubbed}"
        );
        assert!(!scrubbed.contains(PLACEHOLDER), "{scrubbed}");
    }

    #[test]
    fn redact_removes_opaque_credentials_with_no_field_name() {
        for secret in [
            "sk-abcdefghijklmnopqrstuvwx",
            "hf_AbCdEfGhIjKlMnOpQrStUv",
            "ghp_0123456789abcdefghijklmnopqrstuv",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N",
        ] {
            let scrubbed = redactor().text(&format!("failed with {secret} at the gateway"));
            assert!(!scrubbed.contains(secret), "{secret} survived: {scrubbed}");
            assert!(scrubbed.contains("at the gateway"));
        }
    }

    /// Found by a planted-secret audit of a real bundle, not by reasoning: an
    /// audit-log line read `stored token <24 chars>`, which has no `=` for the
    /// assignment rule and no recognisable prefix for the opaque one.
    #[test]
    fn redact_removes_a_credential_named_in_prose_then_printed() {
        for line in [
            "stored token PLANTEDAUDITTOKEN0123456",
            "using api key AbCdEfGhIjKlMnOpQrSt",
            "wrote secret 0123456789abcdef0123",
            "rejected password hunter2hunter2hunter2",
        ] {
            let scrubbed = redactor().text(line);
            assert!(scrubbed.contains(PLACEHOLDER), "not redacted: {scrubbed}");
            for secret in [
                "PLANTEDAUDITTOKEN0123456",
                "AbCdEfGhIjKlMnOpQrSt",
                "0123456789abcdef0123",
                "hunter2hunter2hunter2",
            ] {
                assert!(!scrubbed.contains(secret), "{secret} survived: {scrubbed}");
            }
        }
    }

    /// The threshold that makes the prose rule safe. Ordinary sentences about
    /// credentials must survive, or every log line mentioning a token becomes
    /// unreadable.
    #[test]
    fn redact_leaves_prose_about_credentials_readable() {
        for line in [
            "token expired",
            "token refresh failed after 3 attempts",
            "password required",
            "api key missing from the request",
            "secret not found",
        ] {
            assert_eq!(redactor().text(line), line, "over-redacted: {line}");
        }
    }

    /// The documented limit, pinned so nobody mistakes this filter for a
    /// guarantee. An opaque string with no field name, no header, and no
    /// recognisable prefix survives, and it must: catching it needs an entropy
    /// heuristic, and an entropy heuristic eats `nightly-wheel-gfx120x-all-7-14-0`,
    /// `7.14.0a20260611`, and every SHA-256 in a manifest. The allowlist that
    /// decides what is collected is what actually keeps a bundle safe; this is
    /// the net underneath it.
    #[test]
    fn redact_cannot_catch_an_unmarked_opaque_string() {
        let line = "read failed near PLANTEDBINARYTOKEN012345";
        assert_eq!(redactor().text(line), line);

        // And the reason it must not try: these are the same shape and are the
        // whole point of collecting a bundle.
        for keep in [
            "nightly-wheel-gfx120x-all-7-14-0",
            "7.14.0a20260611",
            "gfx1201",
            "d3b199cd5e162e886c35cbf187f5c176c1bf5d65eae07b64c269b514cca1b972",
        ] {
            assert_eq!(redactor().text(keep), keep, "{keep} must survive");
        }
    }

    #[test]
    fn redact_removes_url_query_fragments_and_userinfo() {
        let scrubbed = redactor().text(
            "downloading https://example.invalid/artifact.tar.gz?X-Amz-Signature=deadbeefcafe&e=1 \
             via https://bob:s3cret@proxy.invalid/",
        );
        assert!(!scrubbed.contains("X-Amz-Signature"), "{scrubbed}");
        assert!(!scrubbed.contains("deadbeefcafe"), "{scrubbed}");
        assert!(!scrubbed.contains("s3cret"), "{scrubbed}");
        // The host and path stay: which artifact was fetched is the fact.
        assert!(scrubbed.contains("https://example.invalid/artifact.tar.gz"));
    }

    #[test]
    fn redact_keeps_a_plain_url_intact() {
        let url = "https://rocm.nightlies.amd.com/v2/gfx120X-all/";
        assert_eq!(redactor().text(url), url);
    }

    // -----------------------------------------------------------------------
    // Identity
    // -----------------------------------------------------------------------

    #[test]
    fn redact_rewrites_home_user_and_host() {
        let scrubbed = redactor().text(
            "runtime at /home/alice/.rocm/runtimes/nightly, windows copy at \
             C:\\Users\\alice\\AppData, reported by alice@workstation",
        );
        assert!(!scrubbed.contains("/home/alice"), "{scrubbed}");
        assert!(!scrubbed.contains("Users\\alice"), "{scrubbed}");
        assert!(!scrubbed.contains("alice@"), "{scrubbed}");
        assert!(!scrubbed.contains("workstation"), "{scrubbed}");
        // The part a support engineer needs survives.
        assert!(scrubbed.contains("~/.rocm/runtimes/nightly"), "{scrubbed}");
    }

    /// A user name that is also an ordinary substring must not shred the
    /// document.
    #[test]
    fn redact_substitutes_identity_only_at_word_boundaries() {
        let scrubbed = Redactor::with(&[], Some("ann"), None)
            .text("ann ran annotate for /opt/annex with PYTORCH_ROCM_ARCH=gfx1201");
        assert!(scrubbed.starts_with("[user] ran annotate"), "{scrubbed}");
        assert!(scrubbed.contains("/opt/annex"), "{scrubbed}");
        assert!(scrubbed.contains("gfx1201"), "{scrubbed}");
    }

    /// Refusing is better than mangling, but it must be *reported* refusing.
    #[test]
    fn redact_reports_identity_too_short_to_substitute() {
        let redactor = Redactor::with(&[], Some("pi"), None);
        assert_eq!(redactor.skipped_identity, vec!["pi".to_owned()]);
        assert_eq!(redactor.text("pip install torch"), "pip install torch");
    }

    #[test]
    fn redact_rewrites_the_longest_home_root_first() {
        let redactor = Redactor::with(&["/home/alice", "/home/alice/work"], None, None);
        assert_eq!(redactor.text("/home/alice/work/x"), "~/x");
    }

    /// Native Windows sets `COMPUTERNAME`, not `HOSTNAME` and not
    /// `/proc/sys/kernel/hostname` — before the fallback existed, every
    /// Windows support bundle leaked the machine name.
    #[test]
    fn redact_resolves_a_windows_computername_and_rewrites_it() {
        let host =
            hostname_from(|name| (name == "COMPUTERNAME").then(|| "WIN-SUPPORT7".to_owned()));
        assert_eq!(host.as_deref(), Some("WIN-SUPPORT7"));

        let redactor = Redactor::with(&[], None, host.as_deref());
        assert_eq!(
            redactor.text("bundle exported from WIN-SUPPORT7 by operator"),
            "bundle exported from [host] by operator"
        );

        // HOSTNAME still wins where both are set (a POSIX shell on Windows).
        let both = hostname_from(|name| match name {
            "HOSTNAME" => Some("penguin".to_owned()),
            "COMPUTERNAME" => Some("WIN-SUPPORT7".to_owned()),
            _ => None,
        });
        assert_eq!(both.as_deref(), Some("penguin"));
    }

    // -----------------------------------------------------------------------
    // JSON
    // -----------------------------------------------------------------------

    #[test]
    fn redact_json_removes_values_by_field_name_whatever_the_type() {
        let mut value = serde_json::json!({
            "token": 12345,
            "nested": { "chatApiKey": "sk-abcdefghijklmnopqrstuv", "model": "gpt-4o" },
            "list": [{ "authorization": "Bearer aaaaaaaaaaaaaaaa" }],
            "activeRuntimeKey": "nightly-wheel-gfx120x-all-7-14-0",
        });
        redactor().json(&mut value);
        assert_eq!(value["token"], PLACEHOLDER);
        assert_eq!(value["nested"]["chatApiKey"], PLACEHOLDER);
        assert_eq!(value["nested"]["model"], "gpt-4o");
        assert_eq!(value["list"][0]["authorization"], PLACEHOLDER);
        assert_eq!(
            value["activeRuntimeKey"],
            "nightly-wheel-gfx120x-all-7-14-0"
        );
    }

    #[test]
    fn redact_json_reduces_an_env_map_to_the_allowlist() {
        let mut value = serde_json::json!({
            "env": {
                "PATH": "/home/alice/.local/bin:/usr/bin",
                "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
                "OPENAI_API_KEY": "sk-abcdefghijklmnopqrstuv",
                "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI",
                "SSH_AUTH_SOCK": "/run/user/1000/keyring/ssh",
            }
        });
        redactor().json(&mut value);
        let env = value["env"].as_object().expect("env survives as a map");
        assert_eq!(env.len(), 2, "{env:?}");
        assert_eq!(env["HSA_OVERRIDE_GFX_VERSION"], "11.0.0");
        // Kept, because fix-6 is about PATH -- but with identity rewritten.
        assert_eq!(env["PATH"], "~/.local/bin:/usr/bin");
    }

    #[test]
    fn redact_json_scrubs_free_text_inside_ordinary_fields() {
        let mut value = serde_json::json!({
            "message": "auth failed for alice using api_key=sk-abcdefghijklmnopqrst"
        });
        redactor().json(&mut value);
        let message = value["message"].as_str().expect("string");
        assert!(!message.contains("sk-abc"), "{message}");
        assert!(!message.contains("alice"), "{message}");
        assert!(message.contains("auth failed"), "{message}");
    }

    #[test]
    fn redact_exportable_env_filters_and_scrubs() {
        let env: BTreeMap<String, String> = [
            ("PATH", "/home/alice/bin"),
            ("HF_TOKEN", "hf_AbCdEfGhIjKlMnOpQrStUv"),
            ("ROCM_PATH", "/opt/rocm"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect();
        let out = exportable_env(&redactor(), &env);
        assert_eq!(out.len(), 2);
        assert_eq!(out["PATH"], "~/bin");
        assert_eq!(out["ROCM_PATH"], "/opt/rocm");
        assert!(!out.contains_key("HF_TOKEN"));
    }
}
