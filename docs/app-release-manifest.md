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
      "format": "deb",
      "url": "https://example.invalid/rocm-app_0.1.0_amd64.deb",
      "fileName": "rocm-app_0.1.0_amd64.deb",
      "sizeBytes": 26,
      "sha256": "53f63fd1…",
      "signatureB64": "…"
    },
    {
      "os": "linux",
      "arch": "x86_64",
      "format": "rpm",
      "url": "https://example.invalid/rocm-app-0.1.0-1.x86_64.rpm",
      "fileName": "rocm-app-0.1.0-1.x86_64.rpm",
      "sizeBytes": 26,
      "sha256": "53f63fd1…",
      "signatureB64": "…"
    },
    {
      "os": "windows",
      "arch": "x86_64",
      "format": "nsis",
      "url": "https://example.invalid/rocm-app_0.1.0_x64-setup.exe",
      "fileName": "rocm-app_0.1.0_x64-setup.exe",
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
| `assets[].arch` | `x86_64`. Anything else is rejected at parse time. |
| `assets[].format` | `deb`, `rpm`, `nsis`, or `unspecified`. Optional; defaults to `unspecified`. Any other value is rejected at parse time. |
| `assets[].fileName` | A plain file name. No `/`, `\`, or `..` — it is joined onto a temporary directory. |
| `assets[].sizeBytes` | Non-zero, and must match the download exactly. |
| `assets[].sha256` | Exactly 64 **lowercase** hex characters. Uppercase is rejected at parse time. |
| `assets[].signatureB64` | Base64 RSASSA-PKCS#1 v1.5 SHA-256 over the asset bytes. |

The digest is required lowercase because a manifest is emitted by a release
tool; mixed case means the file was hand-edited, which is worth refusing on an
install path. The comparison against the downloaded bytes stays
case-insensitive, so a correct manifest is never rejected on a technicality.

## Asset selection

`os` and `arch` alone cannot separate a `.deb` from an `.rpm` — both are
`linux`/`x86_64` — so `format` is what makes a release able to ship both.

Selection filters the assets to those matching the host's `os` and `arch`, then
takes the first whose `format` the host can actually install:

| Host | Formats tried, in order |
|---|---|
| Windows | `nsis`, then `unspecified` |
| Linux with `dpkg` (`/usr/bin/dpkg` or `/etc/debian_version`) | `deb`, then `unspecified` |
| Linux with `rpm` (`/usr/bin/rpm` or `/etc/redhat-release`) | `rpm`, then `unspecified` |
| Linux with both | `deb`, `rpm`, then `unspecified` |
| Linux with neither | `unspecified` |

deb wins on a host that has both tools: such a host is almost always a Debian
derivative with `rpm` installed as a conversion utility, and installing the rpm
there leaves the app invisible to `apt`.

`unspecified` is always tried last, so a manifest written before `format`
existed still installs, while a typed asset always beats an untyped one.

When os+arch matches exist but none is installable, the failure names both the
formats the release offers and the formats this host can install. Reporting a
bare "no asset published" for a machine that *has* an asset it merely cannot
install is the failure mode this avoids.

### Compatibility runs one way

`format` is optional, so a manifest written before the field existed parses
here: `deny_unknown_fields` rejects *unknown* keys, not absent optional ones.
The reverse is **not** true — an older CLI reading a manifest that carries
`format` rejects the whole manifest. That is acceptable only because nothing is
released yet; once a build is in the wild, adding a field to the manifest is a
breaking change.

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
