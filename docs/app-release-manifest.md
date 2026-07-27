<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# ROCm App release manifest

`rocm install app` is the **only** command that installs ROCm App. Installing
the CLI — through `install.sh`, `install.ps1`, `rocm install sdk`, `rocm update`,
or first-run setup — never installs the app. That asymmetry is enforced by
`install_app_is_not_reachable_from_any_other_install_path`, which scans the
repository rather than trusting convention.

## Manifest contract

Schema version **1**. Unknown fields are rejected: a field this CLI does not
understand may change what gets installed, so ignoring it is not safe on an
install path.

```json
{
  "schemaVersion": 1,
  "appVersion": "0.1.0",
  "compatibleCli": { "min": "0.1.0", "max": "0.2.0" },
  "publishedAtUnixMs": 1767225600000,
  "releaseNotesUrl": "https://github.com/mikeroysoft/rocm-app/releases/tag/v0.1.0",
  "assets": [
    {
      "os": "linux",
      "arch": "x86_64",
      "url": "https://example.invalid/rocm-app_0.1.0_amd64.deb",
      "fileName": "rocm-app_0.1.0_amd64.deb",
      "sizeBytes": 26,
      "sha256": "53f63fd1…",
      "signatureB64": "…"
    }
  ]
}
```

| Field | Rule |
|---|---|
| `schemaVersion` | Must equal 1. A newer manifest is refused, not best-effort parsed. |
| `appVersion` | Non-empty. |
| `compatibleCli` | Inclusive CLI version range this app build pairs with. |
| `assets[].os` | `windows` or `linux`. Anything else is rejected at parse time. |
| `assets[].arch` | `x86_64`. Other architectures are out of scope for v1. |
| `assets[].fileName` | A plain file name. No `/`, `\`, or `..` — it is joined onto a temporary directory. |
| `assets[].sizeBytes` | Non-zero, and must match the download exactly. |
| `assets[].sha256` | 64 lowercase hex characters. |
| `assets[].signatureB64` | Base64 RSASSA-PKCS#1 v1.5 SHA-256 over the asset bytes. |

## Verification order

Platform → manifest schema → target match → download → size → digest →
signature → execute.

Every check that needs no network runs first, so an unsupported host or a
malformed manifest costs nothing and reveals nothing to a download server.
Size is checked before the digest because there is no reason to hash bytes
already known to be the wrong length. **Nothing is executed until all checks
pass**, and `install_app_apply_never_executes_an_unverified_asset` asserts it.

The installer is launched by an exact argv from a resolved temporary path
(`argv_for_installer`), never through shell text. The temporary directory
removes itself on drop, so every failure path cleans up without each branch
having to remember to.

## Trust inputs (owner-controlled)

Production trust roots are supplied at runtime. **This repository contains no
private key and no production key material** —
`install_app_repository_contains_no_private_key` enforces that, and
`install_app_private_key_scan_detects_real_key_material` proves the scan can
actually fail.

| Variable | Meaning |
|---|---|
| `ROCM_CLI_APP_PUBLIC_KEY_PATH` | Path to the trusted public key PEM. |
| `ROCM_CLI_APP_PUBLIC_KEY_PEM` | The PEM inline, for environments without a writable path. |
| `ROCM_CLI_APP_REQUIRE_SIGNATURE` | `0`/`false`/`no` disables the signature requirement. |

Signatures are **required by default**. A default of optional would mean a
misconfigured host silently accepts an unsigned installer, which is precisely
the failure worth being loud about. When a signature is required and no key is
configured, the install fails with that reason rather than proceeding.

Tests use ephemeral keys generated per run via
`rocm_core::generate_rsa_signing_keypair`. See
[release-trust.md](release-trust.md) for the wider release signing policy.

## Platform support

Native Windows x86_64 and native Linux x86_64. WSL, macOS, and non-x86_64 are
refused **before any network access**, each with a plain reason and, for WSL, a
next action ("run ROCm App on your Windows desktop instead").

## Approval

`--dry-run` and apply render the *same* plan text, so what a user reviews is
exactly what they approve. Apply then requires either an interactive `y`/`yes`
or the explicit `--yes` flag; a non-interactive run without `--yes` fails with
guidance instead of proceeding.

No driver is installed, updated, or modified by this command. rocm-cli's
separate `rocm install driver` flow is unchanged and unreachable from here.
