<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Review Log: `feat/system-runtime-adoption`

## Purpose

This branch combines the local-agent harness work already committed on the branch with system ROCm SDK adoption, managed Lemonade serving corrections, dashboard telemetry improvements, and portable repository agent tooling. It is pushed to the fork for review and further preparation; it is not yet proposed upstream.

## Upstream state

Pre-push remote refresh on 2026-08-30:

- comparison base: `upstream/main`
- divergence before the remaining commits: 15 upstream commits behind, 3 branch commits ahead
- action required before an upstream pull request: rebase onto the then-current `upstream/main`, resolve conflicts, rerun every gate below, and refresh the leak scan

No upstream pull request, issue, comment, or review reply was created as part of this work.

## Commit structure

Already committed:

- `d0f25d8 feat: configure local agent harnesses`
- `3524d02 feat: add Pi and OMP agent support`
- `961e6df feat: prompt before selecting OMP default`

The remaining work is intentionally split into these review scopes:

1. `feat(runtime): adopt system ROCm SDKs as read-only runtimes`
   - CLI adoption, activation, update reporting, storage accounting, environment resolution, documentation, and tests
   - hermetic dependency-removal guard coverage across Linux distributions
2. `feat(models): harden managed Lemonade serving`
   - stale ComfyUI PID handling, Ornith catalog entry, active-runtime propagation, managed Lemonade backend installation/detection, telemetry, and serving scenarios
3. `feat(dash): collect managed Lemonade telemetry`
   - collector fallback, daemon registry/runner integration, hardware and Observe layouts, and end-to-end dashboard coverage
4. `chore(agents): add portable engineering skills setup`
   - repository agent pointers, portable skill copies, workflow documentation, and lock metadata
5. `docs: add branch review log`
   - this file

## Behavioral decisions

### System ROCm SDKs

- `rocm runtimes adopt-system` discovers supported Linux/WSL SDK layouts and registers a selected SDK instead of copying or modifying it.
- Adopted runtimes are read-only external records. Uninstall removes only the registry entry; the OS-owned SDK remains untouched.
- Activation uses the same runtime-selection state as managed TheRock installs.
- Engine launches consume the active runtime environment rather than inventing a second selection path.
- Update reporting marks system runtimes not applicable instead of treating them as managed update candidates.
- Strict GPU-required behavior remains intact; no CPU fallback was added.

### Managed Lemonade

- A reachable unrelated port cannot revive a stale managed ComfyUI PID record.
- Managed Lemonade backend inspection now queries the short-lived managed daemon using its actual host and port. This prevents a user-level daemon/cache from falsely reporting a backend installed in the managed runtime.
- Detection repairs manifests whose managed GPU backend is missing.
- Installation fails closed unless a GPU `llama-server` exists in the managed runtime.
- Tier-3 runtime-directory tests use trusted `/tmp` ancestry. Production ownership, mode, symlink, and descriptor-relative checks are unchanged.
- The catalog exposes the public Ornith Q4_K_M recommendation.

### Dashboard telemetry

- The Lemonade collector validates `/metrics` and falls back to schema-valid `/api/v1/stats` data.
- Direct generation throughput is consumed when available.
- Managed-service discovery keeps the newest record for duplicate IDs, validates GPU ordinals and IDs, and resets stale telemetry when endpoints or collectors change.
- Observe keeps a seven-row minimum for instances but uses extra terminal height; selection no longer disappears below a permanently fixed four-row viewport.
- CPU overflow reserves its own row and reports the actual number of hidden cores.

### Agent tooling

- Repository instructions point to the canonical issue tracker, triage-label, and domain documentation.
- Thirty-seven portable engineering skills are copied under `.agents/` and recorded by `skills-lock.json`.
- The copied skill tree contains no symlinks, absolute local paths, or credential material; only the expected script files are executable.

## Reviewer map

Start with these files by scope:

- runtime contract and registry: `crates/rocm-core/src/lib.rs`, `crates/rocm-core/src/system_sdk.rs`
- CLI behavior: `apps/rocm/src/main.rs`, `apps/rocm/src/therock.rs`
- engine integration: `engines/lemonade/src/lib.rs`
- model/catalog behavior: `apps/rocm/src/comfyui.rs`, `crates/rocm-core/src/model_catalog.json`
- dashboard collection: `crates/rocm-dash-collectors/src/lemonade.rs`
- dashboard state: `crates/rocm-dash-daemon/src/registry.rs`, `crates/rocm-dash-daemon/src/runner.rs`
- dashboard UI: `crates/rocm-dash-tui/src/ui/tabs/hardware.rs`, `crates/rocm-dash-tui/src/ui/tabs/observe.rs`
- user-observable scenarios: `tests/e2e-cucumber/features/`
- agent tooling: `AGENTS.md`, `docs/agents/`, `.agents/`, `skills-lock.json`

## Verification completed

Final post-fix local gates on Linux with an AMD Radeon AI PRO R9700 (`gfx1201`) and ROCm 7.2.4 present at `/opt/rocm`:

```text
cargo fmt --all -- --check
PASS

cargo check --workspace --all-targets
PASS

cargo test --workspace --all-targets
PASS: 2,295 tests across 29 suites; 8 ignored

cargo clippy --workspace --all-targets -- -D warnings
PASS

python scripts/smoke_local.py
PASS: workspace build, CLI inspection/config/planning, daemon bridge/sandbox, and GPU-only vLLM policy checks

cargo xtask e2e
PASS: 117 scenarios reconciled; 110 passed, 7 expected failures, 0 unexpected failures
PASS: 496 steps reconciled; 489 passed, 7 expected failures
```

The default E2E run included real managed Lemonade GPU serving. The canonical Hugging Face checkpoint scenario downloaded the managed ROCm backend, served `unsloth/Qwen3-0.6B-GGUF:Q4_0`, and completed inference successfully.

Focused regression evidence also passed for:

- system SDK discovery, validation, adoption, active environment, and update behavior
- dependency-removal refusal on a hermetic Ubuntu/apt fixture
- visibly marked engine shells
- stale ComfyUI PID/reused-port handling
- Lemonade telemetry collection and daemon fallback
- dashboard hardware/Observe rendering
- interactive OMP default selection and noninteractive default preservation

## Coverage intentionally not run

These lanes are outside the default local gate and must be considered before upstream submission:

- real installed Claude/Codex/Pi/OMP harness lane (`E2E_INCLUDE_REAL_AGENTS=1`)
- nightly and merge-queue-only scenarios
- installer lifecycle lane (`E2E_INCLUDE_LIFECYCLE=1`)
- Windows and WSL execution
- CI on dedicated hardware

The default suite exercised fake agent harnesses and the real local Lemonade GPU path. It did not mutate the host `/opt/rocm` installation.

## Upstream preflight

Before an upstream pull request:

1. Fetch and rebase onto current `upstream/main`.
2. Review conflicts carefully in `Cargo.lock`, `apps/rocm/src/main.rs`, runtime registry code, and E2E expectations.
3. Rerun formatting, workspace tests, clippy, smoke, and default E2E.
4. Run any newly applicable Windows/WSL, lifecycle, nightly, real-agent, and CI lanes.
5. Re-run the sensitive-content leak scan against the refreshed upstream base and manually inspect the PR title/body, branch name, commit messages, and logs.
6. Verify every commit remains SSH-signed and DCO-signed-off.
7. Obtain explicit approval before creating the upstream PR or posting any public comment.

## Local-only artifact

`btop-cpu.png` remains untracked. It is a local visual reference for the dashboard CPU layout, is not referenced by product or documentation code, and is intentionally excluded from the branch.
