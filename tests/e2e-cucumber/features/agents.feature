@agents
Feature: Agent harness setup

  # Every scenario drives the built `rocm` CLI. Homes, XDG directories, ROCm
  # state, executable lookup, agent configs, and project files are isolated by
  # the agents fixture; no developer harness installation or setting is read.

  @id:agents-list-supported-harnesses
  Scenario: agents-01 - Help explains target selection and listing shows every supported harness
    Given an isolated agents environment
    And a supported fake Claude harness is installed
    When the user reads agents help and lists supported agent harnesses
    Then help describes managed-service detection and the ROCm fallback endpoint
    And every canonical harness and both installation states are listed

  @id:agents-inspect-visible-status
  Scenario: agents-02 - Inspecting one harness reports its executable and visible configuration
    Given an isolated agents environment
    And a configured fake Aider harness
    And supported fake Pi and OMP harnesses are installed
    When the user inspects the Aider harness
    And the user also inspects canonical Pi and OMP
    Then the Aider harness status is shown without changing its configuration
    And canonical Pi and OMP status reports their distinct executables and versions

  @id:agents-aliases-canonicalized
  Scenario: agents-03 - Familiar aliases resolve to canonical harness names
    Given an isolated agents environment
    And all supported fake agent harnesses are installed
    When the user inspects every documented agent alias
    And the user inspects both alias-free harnesses
    Then every alias reports its canonical harness
    And Pi and OMP report no aliases

  @id:agents-invalid-name-and-missing-agent
  Scenario: agents-04 - Unknown names and actions without an agent are rejected with guidance
    Given an isolated agents environment
    When the user names an unknown harness and requests setup without a harness
    And the user names plausible Pi and OMP aliases
    Then both agent invocations fail with valid-name guidance
    And the alias-free Pi and OMP names are rejected with canonical guidance

  @id:agents-target-unique-managed-service
  Scenario: agents-05 - A unique ready managed service supplies endpoint and exact model
    Given an isolated agents environment
    And one ready managed agent service
    When the user previews Aider setup without an explicit target
    Then the plan uses the unique managed endpoint and model

  @id:agents-target-ambiguous-and-model-filtered
  Scenario: agents-06 - Several managed services require a unique model selection
    Given an isolated agents environment
    And two ready managed agent services
    When the user previews setup without and then with a model filter
    Then ambiguity is refused and the exact matching service is selected

  @id:agents-target-explicit-single-model
  Scenario: agents-07 - An explicit unmanaged loopback endpoint supplies its single advertised model
    Given an isolated agents environment
    And an unmanaged agent endpoint advertising one model
    When the user previews setup with only that base URL
    Then the advertised model and normalized endpoint appear in the plan

  @id:agents-target-explicit-multiple-models
  Scenario: agents-08 - An endpoint advertising several models requires an exact model
    Given an isolated agents environment
    And an unmanaged agent endpoint advertising several models
    When the user previews setup without and then with an advertised model
    Then multiple models are refused until the exact model is supplied

  @id:agents-target-default-fallback
  Scenario: agents-09 - The ROCm serving default is the fallback when the model is explicit
    Given an isolated agents environment
    When the user applies offline Aider setup with only an explicit model
    Then the configuration uses the ROCm default loopback endpoint

  @id:agents-target-no-running-server
  Scenario: agents-10 - No managed server gives deterministic serve guidance instead of starting one
    Given an isolated agents environment
    When the user previews setup with no target and no managed server
    Then setup fails with deterministic rocm serve guidance

  @id:agents-target-explicit-unreachable
  Scenario: agents-11 - An explicit local endpoint with no running server is rejected
    Given an isolated agents environment
    When the user previews setup against a local endpoint with no server
    Then setup fails and identifies the unreachable local endpoint

  @id:agents-target-invalid-urls
  Scenario: agents-12 - Unsafe or malformed endpoint URLs are all rejected
    Given an isolated agents environment
    When the user previews setup with invalid endpoint forms
    Then every invalid endpoint is rejected before configuration is written

  @id:agents-plan-dry-run-and-approval
  Scenario: agents-13 - Dry run writes nothing and noninteractive setup requires approval
    Given an isolated agents environment
    And a representative Claude configuration
    When the user previews and then attempts unapproved Claude setup
    Then both commands leave the configuration unchanged and explain why

  @id:agents-plan-redaction-and-idempotence
  Scenario: agents-14 - Plans redact credentials and repeated setup performs no rewrite
    Given an isolated agents environment
    And a Claude configuration containing a credential
    When the user previews and applies the same Claude setup twice
    Then the credential is redacted and the second setup is a filesystem no-op

  @id:agents-persistence-all-adapters
  Scenario: agents-15 - Every adapter safely preserves unrelated global settings
    Given an isolated agents environment
    And representative global configurations for every harness
    And all supported fake agent harnesses are installed
    When the user applies offline setup to every supported harness
    Then every global config registers the exact local model and keeps unrelated settings

  @id:agents-write-refuses-symlink
  @requires-os:linux
  Scenario: agents-16 - Setup refuses a symlinked configuration
    Given an isolated agents environment
    And a symlinked Claude configuration
    And a symlinked Pi second configuration
    When the user attempts offline Claude setup
    And the user attempts offline Pi setup
    Then the symlink targets are unchanged and both setups explain the refusal

  @id:agents-write-refuses-stale-plan
  @requires-os:linux
  Scenario: agents-17 - An edit made during approval invalidates the setup plan
    Given an isolated agents environment
    And a representative Claude configuration
    And representative Pi and OMP configurations
    When the Claude configuration changes at the approval prompt
    And the OMP model registry changes at the approval prompt
    Then both stale plans are refused without losing either concurrent edit

  @id:agents-write-permissions-and-atomicity
  @requires-os:linux
  Scenario: agents-18 - Successful setup preserves permissions and leaves no replacement debris
    Given an isolated agents environment
    And a restricted Claude configuration
    When the user applies offline Claude setup
    Then its permissions are preserved and the atomic replacement is complete

  @id:agents-write-rollback-after-check
  Scenario: agents-19 - A failed protocol check restores the original configuration
    Given an isolated agents environment
    And an endpoint that only lists agent models
    And a representative Claude configuration
    And representative Pi and OMP configurations
    When checked Claude setup is applied to the incompatible endpoint
    And checked Pi and OMP setup is applied to the incompatible endpoint
    Then every checked setup fails and restores all original configuration files

  @id:agents-write-no-check-retained
  Scenario: agents-20 - No-check keeps offline configuration without a protocol request
    Given an isolated agents environment
    And an endpoint that only lists agent models
    And a representative Claude configuration
    When no-check Claude setup is applied to that endpoint
    Then setup is retained without sending a protocol request

  @id:agents-version-detected
  Scenario: agents-21 - A supported installed version is detected and reported
    Given an isolated agents environment
    And a supported fake Claude harness is installed
    And supported fake Pi and OMP harnesses are installed
    When the user previews Claude setup using detected version selection
    And the user previews Pi and OMP setup using detected version selection
    Then all three setup plans report their exact detected version source

  @id:agents-version-override-without-install
  Scenario: agents-22 - Explicit versions permit deterministic setup before installation while preserving managed policy
    Given an isolated agents environment
    When the user configures uninstalled harnesses by explicit version and retries in managed modes
    Then every override is visible direct setup succeeds and known managed modes are refused

  @id:agents-version-unsupported
  Scenario: agents-23 - Unsupported versions remain inspectable but cannot mutate settings
    Given an isolated agents environment
    And an unsupported fake Claude harness is installed
    And unsupported fake Pi and OMP harnesses are installed
    When the user inspects and then previews setup for that harness
    And the user inspects and previews unsupported Pi and OMP harnesses
    Then all unsupported versions remain inspectable and refuse mutation

  @id:agents-protocol-chat-completions
  Scenario: agents-24 - Chat adapters check the Chat Completions route with the exact model
    Given an isolated agents environment
    And a protocol-complete agent endpoint
    And all supported fake agent harnesses are installed
    When checked setup is applied to every Chat Completions harness
    Then every check reaches v1 chat completions with the exact model

  @id:agents-protocol-responses
  Scenario: agents-25 - Codex checks the Responses route with the exact model
    Given an isolated agents environment
    And a protocol-complete agent endpoint
    When checked Codex setup is applied
    Then the check reaches v1 responses with the exact model

  @id:agents-protocol-anthropic-messages
  Scenario: agents-26 - Claude checks the Anthropic Messages route with the exact model
    Given an isolated agents environment
    And a protocol-complete agent endpoint
    When checked Claude setup is applied
    Then the check reaches v1 messages with the exact model

  @id:agents-test-all-fake-harnesses
  Scenario: agents-27 - Every isolated fake harness reads the nonce without exposing the caller workspace
    Given an isolated agents environment
    And all supported fake agent harnesses are installed
    When the user configures and tests every fake harness
    Then every fake process proves safe arguments caller isolation nonce integrity and disposable workspace state

  @id:agents-test-failure-categories
  Scenario: agents-28 - Harness timeout exit failure and missing nonce are distinguished
    Given an isolated agents environment
    When fake Claude tests time out exit nonzero and omit the nonce
    Then each harness failure has its concise category

  @id:agents-test-setup-combinations
  Scenario: agents-29 - Setup-test and no-check-test keep their literal behavior
    Given an isolated agents environment
    And a protocol-complete agent endpoint
    And a supported fake Claude harness is installed
    When the user runs setup-test and then no-check-test
    Then the first checks the protocol and both invoke the isolated harness

  @id:agents-project-override-warning
  Scenario: agents-30 - A project override is warned about but never modified
    Given an isolated agents environment
    And an Aider project override
    And Pi and OMP project and overlay overrides
    When the user applies offline Aider setup
    And the user applies offline Pi and OMP setup with project overlays present
    Then all global setups warn about higher-precedence project and overlay files without changing them

  @id:agents-invalid-flag-combinations
  Scenario: agents-31 - Invalid setup flag combinations are rejected before mutation
    Given an isolated agents environment
    When the user requests invalid agents flag combinations
    Then every invalid flag combination fails without creating a harness config

  @id:agents-multifile-plan-idempotence
  Scenario: agents-32 - Pi updates both user files while OMP registers models without changing defaults
    Given an isolated agents environment
    And representative Pi and OMP configurations
    When the user previews and applies the same Pi and OMP setup twice
    Then dry runs write nothing and repeated setup rewrites no registered configuration

  @id:agents-multifile-partial-rollback
  @requires-os:linux
  Scenario: agents-33 - A second-file write failure restores the first Pi file without replacement debris
    Given an isolated agents environment
    And a Pi second configuration larger than the bounded writer
    When Pi setup hits a bounded second-file write failure
    Then the first Pi file is rolled back and the oversized second file is unchanged

  @id:agents-omp-config-root-and-profiles
  Scenario: agents-34 - OMP resolves relative config roots and normalized profile precedence under home
    Given an isolated agents environment
    When the user inspects OMP with a relative config directory and normalized profiles
    Then the relative OMP root and normalized profile precedence select exact files

  @id:agents-omp-legacy-models-refused
  Scenario: agents-35 - OMP refuses setup over an unmigrated legacy registry without changing it
    Given an isolated agents environment
    And a legacy OMP models.json without a YAML registry
    When the user attempts offline OMP setup with the legacy registry
    Then OMP setup refuses migration and preserves the legacy registry

  @id:agents-omp-default-declined
  Scenario: agents-36 - Interactive OMP setup can retain the existing default after registration
    Given an isolated agents environment
    And a protocol-complete agent endpoint
    And representative Pi and OMP configurations
    And supported fake Pi and OMP harnesses are installed
    When the user interactively registers OMP and declines the default
    Then OMP setup and test use the registered model without changing the existing default

  @id:agents-omp-default-accepted
  Scenario: agents-37 - Interactive OMP setup can select the registered model as the default
    Given an isolated agents environment
    And a protocol-complete agent endpoint
    And representative Pi and OMP configurations
    And supported fake Pi and OMP harnesses are installed
    When the user interactively registers OMP and accepts the default
    Then only the OMP default role changes after registration

  # Explicit opt-in is required in addition to the nightly AMD GPU/vLLM gates:
  # E2E_INCLUDE_NIGHTLY=1 E2E_INCLUDE_REAL_AGENTS=1. Default CI never requires
  # Claude Code or Codex to be installed.
  @id:agents-real-vllm-claude-codex @requires-gpu @requires-engine:vllm @requires-os:linux @nightly @real-agents
  Scenario: agents-38 - Real Claude Code and Codex pass setup protocol and nonce tests through vLLM
    Given a managed runtime is active
    And a model is being served on GPU
    And an isolated environment with real Claude Code and Codex
    When real Claude Code and Codex are configured and tested through agents
    Then both real harnesses pass their protocol and isolated nonce checks
