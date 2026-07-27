<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Support bundles and redaction

`rocm app-support-bundle` writes one `.tar.gz` a user can attach to a public
issue. Everything in this document exists because that sentence has to stay true
on a machine whose owner never reads it.

The bundle is written to disk and nowhere else. Nothing in this path uploads,
posts, or phones home; the user chooses where the file goes and who sees it.

## Two lines of defence, in this order

1. **An allowlist decides what is collected.** The bundle names its members and,
   for configuration, names its fields. Anything not named is absent. This is the
   line that actually holds, because it fails **closed**: a field added to
   `RocmCliConfig` next year is not exported until someone declares it, and the
   test that pins the member set fails the day a new file appears.

2. **`rocm_core::redact` scrubs what did get collected.** Free text — log lines,
   error messages, captured command output — cannot be allowlisted field by
   field, so it is filtered instead. This line fails **open** by nature: a secret
   shape nobody anticipated survives it.

Anything that relies on step 2 alone is a leak waiting for a new credential
format. Step 2 is the safety net, not the plan.

## What a bundle contains

Exactly these members, plus one `logs/<source>.log` per log source that exists:

| Member | Contents |
|---|---|
| `manifest.json` | every other member's name, byte length, and SHA-256 |
| `versions.json` | producer identity and component versions |
| `examination.json` | the redacted `Examination` |
| `diagnosis.json` | the closed catalog's findings for this machine |
| `health.json` | the `rocm app-snapshot` payload, redacted |
| `config.json` | allowlisted configuration fields only |
| `reproduction.json` | OS, architecture, symptom text, and timestamps |

`manifest.json` cannot hash itself, so its own row is omitted; the archive's own
SHA-256 covers the finished file and is printed alongside it.

A test enumerates the finished archive and asserts the member set is exactly this
list plus its `logs/` members, and that no member name contains `..` or begins
with `/`. A bundle that can write outside its own root is a different bug with
the same blast radius.

## What is never collected

| Category | Why |
|---|---|
| Provider API keys, endpoint keys, service tokens | The whole reason a bundle is dangerous. Keys live in the OS keyring or a `0600` file and neither is read here. |
| `Authorization`, `Proxy-Authorization`, `Cookie`, `Set-Cookie` values | A single captured header is a working credential. |
| URL query strings and fragments | A pre-signed download URL *is* the credential. The scheme, host, and path survive; everything after `?` or `#` does not. |
| URL userinfo (`scheme://user:pass@host`) | Same. |
| Unrelated environment variables | An environment dump is the classic accidental credential leak. Only the fourteen variables the diagnosis catalog actually reads are exportable, and that list is an allowlist. |
| Home directory, user name, host name, machine and boot ids | Identity, not diagnosis. Paths are rewritten to `~`, which keeps them readable. |

`PATH` and `LD_LIBRARY_PATH` **are** exported, because `fix-6-path` is about
`PATH` and a bundle that omits it cannot answer the question it was collected to
answer. Their home-directory components are rewritten rather than the whole
variable dropped.

## Over-redaction is also a bug

A bundle whose every string reads `[redacted]` tells a support engineer nothing,
and the user files the issue anyway with a screenshot. So the rules are narrow,
and their negative cases are tested as carefully as the positive ones:

- `active_runtime_key`, `previous_runtime_key`, and `key_source` survive intact.
  They end in `key` but are not credentials, and a diagnosis without the active
  runtime key is unactionable.
- `public_key_pem` survives. A public key is public.
- `tokens_per_second` survives. It contains `token` and is a throughput number.
- `token expired` and `password required` survive. The prose rule only fires on
  a value of sixteen or more unbroken credential-shaped characters.
- A plain `https://rocm.nightlies.amd.com/v2/gfx120X-all/` survives whole. Which
  index was used is the fact.

The classifier therefore matches a field name that **ends** in a credential word,
not one that contains one anywhere.

## What redaction cannot catch

A secret printed into a log with **no marker at all** — no field name, no
`Authorization:`, no recognisable prefix, just an opaque string in a sentence —
survives. A planted-secret audit of a real bundle confirms it, and
`redact_cannot_catch_an_unmarked_opaque_string` pins the limit so nobody later
mistakes the filter for a guarantee.

Catching that case needs an entropy heuristic, and an entropy heuristic eats the
things a bundle exists to carry: `nightly-wheel-gfx120x-all-7-14-0`,
`7.14.0a20260611`, `gfx1201`, and every SHA-256 in the manifest are all
high-entropy and all load-bearing. A filter that removes them produces a bundle
nobody can diagnose from, which is the failure mode in the other direction.

So the rule stands: an unmarked secret in a log line is a bug in whatever wrote
it, and the allowlist — not the filter — is what keeps the bundle safe.

## Identity too short to substitute

A user called `pi` on a host called `mi` cannot be substituted by whole-word
replacement without shredding `pip`, `mirror`, and half the log. Below three
characters the substitution is refused and the refusal is **reported** in the
manifest's `redaction.identitySkipped`, rather than silently traded for a
document nobody can read. A reviewer can then decide.

## Verifying the policy

```bash
cargo test -p rocm-core redact           # the rules, positive and negative
cargo test -p rocm --bin rocm logs       # bounded reads, allowlist, manifest
```

The archive test plants known secrets in every input it can reach — a config
value, a log line, an environment variable, a URL query — and then asserts the
finished archive's bytes contain none of them. Asserting on the *bytes* rather
than on the structs is deliberate: it catches a field that was serialized on a
path nobody thought to redact.

## Reading a bundle

```bash
tar -tzf rocm-support-*.tar.gz              # member list
tar -xOzf rocm-support-*.tar.gz manifest.json | python3 -m json.tool
sha256sum rocm-support-*.tar.gz             # compare with the printed digest
```

Every member's recorded SHA-256 can be checked independently:

```bash
tar -xOzf rocm-support-*.tar.gz health.json | sha256sum
```

A mismatch means the archive was edited after it was written, which is worth
knowing before trusting anything else in it.
