@agents
Feature: Agent harness setup

  # Every scenario drives the built `rocm` CLI. Homes, XDG directories, ROCm
  # state, executable lookup, agent configs, and project files are isolated by
  # the agents fixture; no developer harness installation or setting is read.

  @id:agents-list-supported-harnesses
  Scenario: 1 - Help explains target selection and listing shows every supported harness
    Given an isolated agents environment
    And a supported fake Claude harness is installed
    When the user reads agents help and lists supported agent harnesses
    Then help describes managed-service detection and the ROCm fallback endpoint
    And every canonical harness and both installation states are listed

  @id:agents-inspect-visible-status
  Scenario: 2 - Inspecting one harness reports its executable and visible configuration
    Given an isolated agents environment
    And a configured fake Aider harness
    When the user inspects the Aider harness
    Then the Aider harness status is shown without changing its configuration

  @id:agents-aliases-canonicalized
  Scenario: 3 - Familiar aliases resolve to canonical harness names
    Given an isolated agents environment
    And all supported fake agent harnesses are installed
    When the user inspects every documented agent alias
    Then every alias reports its canonical harness

  @id:agents-invalid-name-and-missing-agent
  Scenario: 4 - Unknown names and actions without an agent are rejected with guidance
    Given an isolated agents environment
    When the user names an unknown harness and requests setup without a harness
    Then both agent invocations fail with valid-name guidance

  @id:agents-target-unique-managed-service
  Scenario: 5 - A unique ready managed service supplies endpoint and exact model
    Given an isolated agents environment
    And one ready managed agent service
    When the user previews Aider setup without an explicit target
    Then the plan uses the unique managed endpoint and model

  @id:agents-target-ambiguous-and-model-filtered
  Scenario: 6 - Several managed services require a unique model selection
    Given an isolated agents environment
    And two ready managed agent services
    When the user previews setup without and then with a model filter
    Then ambiguity is refused and the exact matching service is selected

  @id:agents-target-explicit-single-model
  Scenario: 7 - An explicit unmanaged loopback endpoint supplies its single advertised model
    Given an isolated agents environment
    And an unmanaged agent endpoint advertising one model
    When the user previews setup with only that base URL
    Then the advertised model and normalized endpoint appear in the plan

  @id:agents-target-explicit-multiple-models
  Scenario: 8 - An endpoint advertising several models requires an exact model
    Given an isolated agents environment
    And an unmanaged agent endpoint advertising several models
    When the user previews setup without and then with an advertised model
    Then multiple models are refused until the exact model is supplied

  @id:agents-target-default-fallback
  Scenario: 9 - The ROCm serving default is the fallback when the model is explicit
    Given an isolated agents environment
    When the user applies offline Aider setup with only an explicit model
    Then the configuration uses the ROCm default loopback endpoint

  @id:agents-target-no-running-server
  Scenario: 10 - No running server gives serve guidance instead of starting one
    Given an isolated agents environment
    When the user previews setup with no service or model
    Then setup fails and tells the user to run rocm serve

  @id:agents-target-invalid-urls
  Scenario: 11 - Unsafe or malformed endpoint URLs are all rejected
    Given an isolated agents environment
    When the user previews setup with invalid endpoint forms
    Then every invalid endpoint is rejected before configuration is written

  @id:agents-plan-dry-run-and-approval
  Scenario: 12 - Dry run writes nothing and noninteractive setup requires approval
    Given an isolated agents environment
    And a representative Claude configuration
    When the user previews and then attempts unapproved Claude setup
    Then both commands leave the configuration unchanged and explain why

  @id:agents-plan-redaction-and-idempotence
  Scenario: 13 - Plans redact credentials and repeated setup performs no rewrite
    Given an isolated agents environment
    And a Claude configuration containing a credential
    When the user previews and applies the same Claude setup twice
    Then the credential is redacted and the second setup is a filesystem no-op

  @id:agents-persistence-all-adapters
  Scenario: 14 - Every adapter safely preserves unrelated global settings
    Given an isolated agents environment
    And representative global configurations for every harness
    And all supported fake agent harnesses are installed
    When the user applies offline setup to every supported harness
    Then every global config visibly selects the exact local model and keeps unrelated settings

  @id:agents-write-refuses-symlink
  @requires-os:linux
  Scenario: 15 - Setup refuses a symlinked configuration
    Given an isolated agents environment
    And a symlinked Claude configuration
    When the user attempts offline Claude setup
    Then the symlink target is unchanged and setup explains the refusal

  @id:agents-write-refuses-stale-plan
  @requires-os:linux
  Scenario: 16 - An edit made during approval invalidates the setup plan
    Given an isolated agents environment
    And a representative Claude configuration
    When the Claude configuration changes at the approval prompt
    Then the stale plan is refused without losing the concurrent edit

  @id:agents-write-permissions-and-atomicity
  @requires-os:linux
  Scenario: 17 - Successful setup preserves permissions and leaves no replacement debris
    Given an isolated agents environment
    And a restricted Claude configuration
    When the user applies offline Claude setup
    Then its permissions are preserved and the atomic replacement is complete

  @id:agents-write-rollback-after-check
  Scenario: 18 - A failed protocol check restores the original configuration
    Given an isolated agents environment
    And an endpoint that only lists agent models
    And a representative Claude configuration
    When checked Claude setup is applied to the incompatible endpoint
    Then setup fails and restores the original configuration

  @id:agents-write-no-check-retained
  Scenario: 19 - No-check keeps offline configuration without a protocol request
    Given an isolated agents environment
    And an endpoint that only lists agent models
    And a representative Claude configuration
    When no-check Claude setup is applied to that endpoint
    Then setup is retained without sending a protocol request

  @id:agents-version-detected
  Scenario: 20 - A supported installed version is detected and reported
    Given an isolated agents environment
    And a supported fake Claude harness is installed
    When the user previews Claude setup using detected version selection
    Then the setup plan reports the detected version source

  @id:agents-version-override-without-install
  Scenario: 21 - Explicit versions permit deterministic setup before installation while preserving managed policy
    Given an isolated agents environment
    When the user configures uninstalled harnesses by explicit version and retries in managed modes
    Then every override is visible direct setup succeeds and known managed modes are refused

  @id:agents-version-unsupported
  Scenario: 22 - Unsupported versions remain inspectable but cannot mutate settings
    Given an isolated agents environment
    And an unsupported fake Claude harness is installed
    When the user inspects and then previews setup for that harness
    Then inspection succeeds but mutation is refused as unsupported

  @id:agents-protocol-chat-completions
  Scenario: 23 - Chat adapters check the Chat Completions route with the exact model
    Given an isolated agents environment
    And a protocol-complete agent endpoint
    And all supported fake agent harnesses are installed
    When checked setup is applied to every Chat Completions harness
    Then every check reaches v1 chat completions with the exact model

  @id:agents-protocol-responses
  Scenario: 24 - Codex checks the Responses route with the exact model
    Given an isolated agents environment
    And a protocol-complete agent endpoint
    When checked Codex setup is applied
    Then the check reaches v1 responses with the exact model

  @id:agents-protocol-anthropic-messages
  Scenario: 25 - Claude checks the Anthropic Messages route with the exact model
    Given an isolated agents environment
    And a protocol-complete agent endpoint
    When checked Claude setup is applied
    Then the check reaches v1 messages with the exact model

  @id:agents-test-all-fake-harnesses
  Scenario: 26 - Every isolated fake harness reads the nonce without exposing the caller workspace
    Given an isolated agents environment
    And all supported fake agent harnesses are installed
    When the user configures and tests every fake harness
    Then every fake process proves safe arguments caller isolation nonce integrity and disposable workspace state

  @id:agents-test-failure-categories
  Scenario: 27 - Harness timeout exit failure and missing nonce are distinguished
    Given an isolated agents environment
    When fake Claude tests time out exit nonzero and omit the nonce
    Then each harness failure has its concise category

  @id:agents-test-setup-combinations
  Scenario: 28 - Setup-test and no-check-test keep their literal behavior
    Given an isolated agents environment
    And a protocol-complete agent endpoint
    And a supported fake Claude harness is installed
    When the user runs setup-test and then no-check-test
    Then the first checks the protocol and both invoke the isolated harness

  @id:agents-project-override-warning
  Scenario: 29 - A project override is warned about but never modified
    Given an isolated agents environment
    And an Aider project override
    When the user applies offline Aider setup
    Then global setup succeeds with an override warning and the project file is unchanged

  @id:agents-invalid-flag-combinations
  Scenario: 30 - Invalid setup flag combinations are rejected before mutation
    Given an isolated agents environment
    When the user requests invalid agents flag combinations
    Then every invalid flag combination fails without creating a harness config

  # Explicit opt-in is required in addition to the nightly AMD GPU/vLLM gates:
  # E2E_INCLUDE_NIGHTLY=1 E2E_INCLUDE_REAL_AGENTS=1. Default CI never requires
  # Claude Code or Codex to be installed.
  @id:agents-real-vllm-claude-codex @requires-gpu @requires-engine:vllm @requires-os:linux @nightly @real-agents
  Scenario: 31 - Real Claude Code and Codex pass setup protocol and nonce tests through vLLM
    Given a managed runtime is active
    And a model is being served on GPU
    And an isolated environment with real Claude Code and Codex
    When real Claude Code and Codex are configured and tested through agents
    Then both real harnesses pass their protocol and isolated nonce checks
