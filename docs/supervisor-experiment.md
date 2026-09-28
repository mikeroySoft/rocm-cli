<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# ROCm supervisor: investigation and isolated experiment

**Status:** Proposal for team investigation. No implementation, hardware experiment, or production integration has been approved by this document.

**Audience:** rocm-cli maintainers, inference-engine engineers, and hardware/test owners.

**Decision requested:** Assign an experiment owner and reviewer, establish an isolated test environment, and authorize the initial non-GPU investigation. Authorize live GPU execution separately with the hardware owner.

## Executive summary

Investigate whether a small, resource-aware local supervisor would make independent AI workloads coexist more predictably on an AMD GPU host. The likely eventual home is `rocmd`, but the investigation should happen in a disposable lab outside the rocm-cli repository and outside Factory.

The motivating example is a host running an interactive language model, an embedding model, and an image-generation application. Each application should remain unaware of the others. However, independent engines can select the same GPU, reserve overlapping memory budgets, or degrade each other's responsiveness. A shared supervisor could coordinate admission and lifecycle while explaining why a workload is running, waiting, or degraded.

The proposed pattern is familiar from infrastructure control planes:

> Observe actual state → compare with declared intent and policy → take a bounded action → record the result → repeat.

The hypothesis is that this pattern is useful on a single host. It does **not** imply a need for Kubernetes, distributed consensus, custom resource frameworks, or a general-purpose scheduler.

**Recommended experiment:** Two independently configured model servers share one GPU under explicit conservative budgets. A third incompatible launch is refused before its engine starts. The controller survives interruption without duplicating workloads, and its decisions are understandable without an LLM.

**Estimated effort:** Approximately 5–9 engineering days for a bounded Linux experiment, assuming a working isolated engine environment and suitable hardware. An initial go/no-go signal should be available after 2–3 days. These are planning estimates, not measured implementation times; dependency builds and hardware compatibility issues can extend them.

## 1. Questions the experiment must answer

1. Can explicit resource budgets make two engine instances coexist usefully under representative concurrent traffic?
2. Can one admission authority prevent conflicting launches and reconstruct ownership after its own crash?
3. Can the recorded state explain capacity decisions and failures well enough to help users and, later, an LLM?

The experiment must allow a negative conclusion. If static engine configuration and clearer CLI preflight solve most of the problem, a more capable daemon may not be justified.

### Proposed operating contract

> Coordinate participating rocm-cli-managed workloads to prevent predictable overcommit, recover bounded lifecycle failures, and make host degradation observable.

Do not promise universal OOM prevention, hard GPU memory isolation, automatic driver repair, or guaranteed latency for arbitrary programs.

Applications need not communicate with one another, but their launchers or engines must participate in a shared resource contract for meaningful coordination. Observing an arbitrary GPU process does not make it safely controllable.

## 2. Existing rocm-cli foundations

Prior source inspection found relevant behavior in the following locations. Symbols are navigation anchors, not a compatibility guarantee; the team should confirm them at the experiment's pinned baseline.

| Area | Source anchors | Relevance |
|---|---|---|
| Automation loop | `apps/rocmd/src/lib.rs`: `run_daemon`, `evaluate_watchers` | Existing periodic observation and policy execution |
| Recovery | `apps/rocmd/src/lib.rs`: `find_recoverable_service`, `restart_managed_service` | Existing managed-service health and recovery paths |
| State records | `crates/rocm-core/src/lib.rs`: `ManagedServiceRecord` | Placement, process identity, readiness, and restart metadata |
| GPU selection | `apps/rocm/src/main.rs`: `auto_select_gpu_indices`, `select_auto_gpu_index` | Selection is not an authoritative reservation transaction; inspected code documents a concurrent-launch race and permits GPU 0 when all detected devices are assigned |
| Engine budgets | `apps/rocm/src/main.rs`: `engine_recipe_with_gpu_memory_utilization_override`; `docs/vllm.md` | Existing explicit vLLM memory-utilization configuration |
| Telemetry | `crates/rocm-dash-daemon/src/runner.rs`; `crates/rocm-dash-core/src/vram.rs` | GPU and engine metrics; attribution can fall back to device-wide values |
| Process termination | `crates/rocm-core/src/proc_lifecycle.rs` | Verified process identities and bounded termination |
| Assistant tools | `apps/rocmd/src/lib.rs`: `run_mcp_server`, `rocm_mcp_tools`; `docs/llm-tool-use.md` | Structured inspection and mutation interfaces |
| Directory isolation | `crates/rocm-core/src/lib.rs`: `AppPaths::discover`, `engine_envs_root` | Separate config, data, cache, and engine-environment roots |

This is source-based context, not evidence that a resource coordinator already exists or that its proposed guarantees have been tested. The dashboard telemetry loop and automation supervisor also have distinct lifecycles today. Reusing their code does not require immediately merging them.

## 3. Scope and non-goals

### In scope

- One Linux host and one operator's lab-owned workloads.
- One GPU and initially one serving engine with two model configurations.
- Explicit, conservative resource budgets derived from measurements.
- Admission before engine launch, including accounting for in-progress starts.
- Desired state, observed state, and eligibility as distinct concepts.
- Bounded retries, operator-stop precedence, and crash recovery.
- A small command interface and machine-readable decision timeline.
- Reproducible non-GPU scenarios and a separately authorized live demonstration.

### Out of scope

- Changes to production `rocmd`, `rocm serve`, dashboard, or engine protocol.
- Factory tickets, dispatch, CI modifications, or automatic merges.
- Windows/WSL parity, multi-user tenancy, or multi-host scheduling.
- Automatic eviction, model swapping, request routing, or dynamic budget tuning.
- Driver installation, GPU reset, repartitioning, power tuning, or reboot.
- Hard GPU-memory enforcement for arbitrary programs.
- Kernel profiling integration or autonomous LLM mutations.

Exclusive GPU assignment may be a useful policy, but it is not sufficient proof for this experiment: the motivating problem is useful coexistence on a shared GPU.

## 4. Isolation and operational safety

### Source and Factory isolation

Create an independent local lab repository, for example:

```text
~/dev/experiments/rocm-supervisor-lab/
    controller/
    scenarios/
    runs/
```

Do not use the active rocm-cli checkout or a Factory worktree. Do not modify its Git state, workspace, hooks, `.factory.toml`, or installed binaries. Do not register the lab with Factory or assign dispatch labels.

If existing Rust code is needed, use a separate checkout pinned to a recorded commit and a separate build-output directory. Avoid following ongoing Factory changes during a measurement campaign.

The inspected `.factory.toml` marks clippy, tests, and smoke checks as exclusive. This does not establish that an unrelated lab process participates in the same resource exclusion. Do not treat those flags as a hardware reservation for the experiment.

### State and runtime isolation

Use dedicated writable paths for state, logs, sockets, caches, engine environments, and model downloads. Use distinct loopback endpoints and treat bind failures as conflicts; never stop an existing listener to reclaim its port.

Existing rocm-cli overrides, if needed:

- `ROCM_CLI_CONFIG_DIR`
- `ROCM_CLI_DATA_DIR`
- `ROCM_CLI_CACHE_DIR`
- `ROCM_CLI_ENGINE_ENVS_ROOT`

These are isolation controls, **not a sandbox**. Inspect inherited environment variables, engine-specific caches, runtime registries, and any command that can auto-start a helper. Do not copy production configuration wholesale into the lab.

For initial live runs, prefer launching a known engine executable directly from a lab-owned environment. This avoids activating existing `rocm serve` automation/recovery paths. Do not update or install packages into an existing managed runtime. Model weights may be reused through a verified read-only arrangement; caches that require writes remain separate.

Run manually; install no boot/login service. Teardown may act only on positively identified lab-owned process groups. Never use broad process-name matching or generic kill commands to clean up.

### Hardware isolation

**Recommended:** develop non-GPU logic locally and run live inference on a separate test host not serving Factory or other useful workloads.

A separate directory, user, container, or device-selection variable does not isolate GPU memory, performance, or driver failures. A second GPU on the same host reduces contention but may still share host/driver failure modes.

If only the current shared host is available, obtain explicit hardware-owner approval for an exclusive test window and bounded load. This is reduced risk, not zero disruption. If uninterrupted operation is required and separate hardware is unavailable, complete non-GPU stages and report live coexistence as unverified.

Do not deliberately exhaust GPU memory or trigger driver faults on a shared machine. Simulate these conditions in the controller scenarios. Unexpected external GPU activity during a live test invalidates the controlled measurement: stop new admission, preserve evidence, and coordinate with the hardware owner rather than evicting the external workload.

## 5. Minimal experimental design

A small Python controller using standard-library process handling and SQLite is a reasonable disposable implementation. This is not a production language or storage decision. Avoid a new framework, plugin system, or UI.

### Managed object

Initially manage an **engine instance and its complete owned process group**, including its resolved model and serving configuration. A model name alone is insufficient: engines may create workers or change loaded models, and demand varies with context length and concurrency.

Expose only the operations needed to request a named instance, stop it, and inspect its state and decisions. Use stable workload and request identities so retrying a request does not create another instance.

### State

Keep these concepts distinct:

- **Desired:** what the operator requested, such as running or stopped.
- **Eligible:** whether current policy permits starting or continuing.
- **Observed:** process identity, readiness, measurements, and failures.
- **Operation:** a particular launch/stop attempt and its progress.

A pressure-suspended workload may still be desired but not eligible. It must not be interpreted as a crashed workload that should immediately restart.

Suggested precedence:

> Operator stop or maintenance → safety restrictions → admission eligibility → recovery → optimization.

Default to refusing insufficient-capacity requests rather than creating indefinite queues. If queueing is explored, specify expiration, cancellation, fairness, and whether permission remains valid when execution is delayed.

### Admission versus reconciliation

Admission is synchronous: inspect policy and reservations, reserve capacity transactionally, then launch. In-progress starts consume reservations. Do not wait for a periodic telemetry tick to prevent conflicting launches.

Reconciliation observes outcomes and repairs interrupted operations. Downloads, loading, health checks, and telemetry must not block the admission decision loop indefinitely. Bound operation durations and isolate slow probes.

Distinguish startup peaks from steady-state demand. It may be possible to admit two resident models only if startup is sequenced.

### Resource accounting and honest guarantees

| Term | Meaning |
|---|---|
| Admission | Permission to start |
| Reservation | Capacity promised in the controller's accounting |
| Enforcement | A verified engine/platform mechanism that constrains consumption |
| Observation | A measurement of actual consumption |

A database reservation does not reserve GPU allocator memory. vLLM memory-utilization configuration is not a universal hard process-memory limit. Record exactly which controls exist and which guarantees remain estimates.

Measure model weights, cache configuration, startup overhead, and representative request demand. Retain explicit headroom. Record telemetry timestamps and provenance. Unknown consumption must not become zero, and device-total attribution must not be summed as if it were independent per-instance consumption.

High occupancy alone is not a failure: engines may intentionally preallocate cache. Memory fit also does not imply acceptable latency under shared compute or bandwidth contention.

### Crash and ownership protocol

Persistence and process creation cannot be made atomic merely by using SQLite or atomic JSON writes. Exercise the gap between recording intent, launching a process, and recording its identity.

Require a stable launch identity, a rediscoverable ownership mechanism, and reconstruction before new admission after restart. Evaluate native process-group/cgroup or service-manager facilities where applicable; validate their child-lifetime and rediscovery behavior rather than assuming it.

Established inference should normally survive a controller crash. If ownership cannot be established after restart, mark it uncertain, retain conservative accounting, and refuse potentially duplicating launches. Do not kill an uncertain process.

Record boot identity where process identity can otherwise be confused across reboots. Confirm the whole owned group has exited before releasing its reservation. A successful signal operation is not proof that GPU-holding workers are gone.

## 6. Phased execution and exit gates

### Stage A — Establish isolation and the basic controller

**Estimate:** 1–2 days. No GPU access required.

1. Record the lab location, pinned source baseline, owner, reviewer, and writable roots.
2. Create the small controller and inspection interface.
3. Use controlled child processes that can delay readiness, exit, or ignore graceful shutdown.
4. Exercise concurrent admission, startup reservations, and explicit stop.
5. Verify that outputs remain within the lab and teardown leaves no unexplained children.

**Gate:** concurrent requests cannot spend the same capacity; explicit stop prevents restart; uncertain ownership never authorizes signaling. Child-process simulations are lifecycle evidence only, not GPU validation.

### Stage B — Exercise interrupted operations and uncertainty

**Estimate:** 1–2 days. No GPU access required.

Provide a deterministic scenario runner with controllable synchronization points rather than relying on timing sleeps to hit crash windows.

| Scenario | Required observation |
|---|---|
| Two incompatible simultaneous requests | Only one admitted; the other receives a capacity reason |
| Controller dies before launch | Accepted intent remains recoverable |
| Controller dies after launch before completion is recorded | Existing attempt recovered or marked uncertain; no duplicate |
| Established child survives controller restart | Ownership/accounting reconstructed before new admission |
| Telemetry unavailable or stale | No optimistic admission or speculative kill |
| Repeated child failure | Finite retry policy reaches an explicit blocked state |
| Stop during startup/recovery | No late resurrection |
| Graceful stop cannot confirm exit | Reservation retained; outcome reported truthfully |
| Unrelated process occupies an endpoint | No adoption or termination of the unrelated process |

**Gate:** repeatable transcripts demonstrate each transition and verified cleanup. Select one policy for retry count/backoff and document it; tuning those constants is not the goal.

**Early decision:** after approximately 2–3 days, review whether the state model is coherent and a suitable isolated live engine environment is available. Stop or narrow the experiment if either prerequisite fails.

### Stage C — Demonstrate useful shared-GPU inference

**Estimate:** 2–4 days. Hardware-owner authorization required.

Choose one working engine, likely vLLM if available, and two small but representative model configurations. Confirm those configurations can run independently before building further integration. Mixed-engine support is a later experiment.

Before execution, agree on load duration, input/context sizes, request concurrency, acceptable error rate and latency degradation, startup timeout, and abort criteria. Record numeric thresholds in the run manifest; do not choose success thresholds after seeing results.

1. Measure A alone: startup peak, steady memory, request latency, throughput, errors.
2. Measure B alone with the same measurement discipline.
3. Set explicit budgets and headroom from observations.
4. Start A and B through the controller, sequencing startup if necessary.
5. Exercise representative bounded concurrent traffic; repeat runs to distinguish noise from consistent interference.
6. Request C with a declared budget that cannot fit. Verify refusal before engine launch and continued service from A and B.
7. Explicitly stop B; confirm process-group exit and capacity recovery; resubmit C, selected to fit the new arrangement.
8. Restart the controller while established inference continues. Verify no duplicate engine and no unaccounted reservation.

Do not induce real OOM merely to prove that the refusal is useful. The refusal tests admission; successful representative concurrent inference tests whether the chosen budgets are useful. Neither proves safety for every input shape or external allocator.

**Gate:** both models serve within the predeclared envelope, the incompatible request is refused without disrupting established service, and controller recovery does not duplicate work.

### Stage D — Package the demonstration and decide

**Estimate:** ½–1 day.

Deliver:

- A documented single-command non-GPU scenario replay.
- A documented single-command live demonstration requiring explicit hardware opt-in.
- A teardown command with verified lab-only targeting and exit confirmation.
- Run manifests and bounded measurement/decision artifacts.
- A findings report separating observed results, estimates, limitations, and untested cases.
- A recommendation to integrate, narrow, extend the experiment, or stop.

These commands are deliverables to implement in the lab, not existing rocm-cli commands.

## 7. LLM and profiling: preserve the option without building it first

The deterministic controller must remain usable when inference is unavailable. A local assistant on the affected GPU may fail exactly when diagnosis is needed; it must not be on the admission, recovery, or safety path.

The first useful diagnostic interface is structured evidence answering:

- Why is this workload not running?
- Which reservations or observations prevent admission?
- What changed before degradation?
- Which operator-approved action could make progress possible?

A bounded incident bundle should contain resolved versions/configuration, decision timestamps, measurement provenance, exit reasons, and recent approved changes. Keep it local by default and avoid capturing prompt bodies or secrets unnecessarily. Provider upload requires separate approval; local read permission is not permission to disclose data externally.

Later profiling must be capability-specific. A profiler may require launch-time instrumentation, restart, privileges, exclusive counters, or significant overhead. Do not promise universal attachment to any PID. A proposed capture must state its target, disruption, duration, output size, and data sensitivity.

If LLM-assisted tuning is investigated later, select an explicit objective, change one controlled variable, compare representative before/after results, and retain a rollback configuration. Logs and trace contents are untrusted input. Validate authorization at every mutation entrypoint; tool annotations are not an enforcement mechanism.

## 8. Handoff ownership and decisions

Before starting, the receiving team should assign:

| Responsibility | Required decision |
|---|---|
| Experiment owner | Owns lab implementation, scenario execution, and findings |
| Reviewer | Checks lifecycle/accounting invariants and challenges conclusions |
| Hardware owner | Authorizes host, workload limits, execution window, and abort criteria |
| Engine specialist, if needed | Confirms isolated runtime and meaning of engine controls |

These roles can overlap. Progress should not depend on the original proposer being available. The bounded defaults in this document are sufficient for the initial investigation; broader scope or shared-hardware risk requires the receiving team's explicit decision.

Record unresolved questions in the lab findings rather than silently expanding scope:

- Which models, engine version, and load represent a useful coexistence case?
- Which budget controls are enforceable versus estimates?
- Which process ownership mechanism survives interruption reliably on the test platform?
- Does startup sequencing materially improve utilization?
- Is manual budget configuration acceptable for the intended user?
- Is a supervisor better than improved preflight and static serving recipes?

## 9. Go/no-go and a possible production path

**Proceed toward integration** if shared inference is useful, admission prevents predictable conflicts, recovery preserves ownership, and decision explanations materially help operation.

**Narrow the proposal** if fixed engine settings plus clearer CLI planning capture most of the value.

**Reconsider the contract** if demand variability makes conservative budgets unusable or available controls cannot support the intended guarantees.

**Stop** if useful coexistence cannot be demonstrated, isolated hardware is unavailable for the required evidence, or safe ownership cannot be established. Preserve the findings; do not merge a prototype because effort has already been spent.

If the result is positive, propose a separate production design and estimate. `rocmd` is the likely authority, but integration requires routing all relevant launch/stop/recovery paths through it, migrating lifecycle semantics, preserving authorization, validating platform behavior, and adding repository-standard user-observable scenarios. Reuse existing telemetry and process modules without putting slow probes in the critical control path. Avoid competing lifecycle ownership with other orchestrators.

Factory work should begin only after maintainers approve that production scope and split it into independently verifiable changes. The lab itself should remain outside Factory and out of the production workspace.

## 10. Evidence record for every live run

Retain enough information for another engineer to reproduce or reject the conclusion:

- Lab revision and any referenced rocm-cli baseline commit.
- Host OS/kernel, GPU identity, driver, runtime, and engine versions.
- Exact model revisions, launch flags, serving settings, and budget/headroom assumptions.
- Startup order, workload inputs, concurrency, duration, and predeclared success/abort thresholds.
- Measurement sources, timestamps, freshness, and attribution limitations.
- Admission decisions, operation identities, process-group lifecycle, and failure injections.
- Latency, throughput, errors, startup/steady memory, and repeated-run variability.
- External activity or deviations that invalidate the controlled comparison.
- Teardown outcome and any residual processes or resources.

Store bounded artifacts without credentials or unnecessary application data. The final report should say exactly what was exercised—not merely that tests passed.
