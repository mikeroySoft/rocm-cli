Feature: Update report

  # `rocm update` (with no arguments) prints an update report: which managed
  # runtimes have updates, plus the status of each update feed (CLI, engines,
  # model recipes, runtimes). The walkthrough verified it correctly distinguishes
  # published feeds from not-configured ones, but nothing pinned it. Run with no
  # managed runtimes so the report needs no network — mock lane, every PR.

  @id:update-report-distinguishes-feed-status
  Scenario: 1 - The update report distinguishes configured from not-configured feeds
    Given a machine with no managed runtimes
    When the user checks for updates
    Then the report shows there are no managed runtimes to update
    And it reports each update feed's status, marking unpublished feeds as not configured

  # The command name reads as a self-update. `--apply` only installs runtime
  # updates; the help must say so and name the installer as the CLI upgrade path.
  @id:update-help-names-runtime-only-apply-and-cli-upgrade-path
  Scenario: 2 - The update help explains --apply is runtime-only and how to upgrade the CLI
    When the user asks for update help
    Then the help states --apply does not update the CLI and points to the installer
