Feature: Update report

  # `rocm update` (with no arguments) prints an update report: which managed
  # runtimes have updates, plus the status of each update feed (CLI, engines,
  # model recipes, runtimes). The walkthrough verified it correctly distinguishes
  # published feeds from not-configured ones, but nothing pinned it. Run with no
  # managed runtimes so the report needs no network — mock lane, every PR.

  @id:update-report-distinguishes-feed-status
  Scenario: update-01 - The update report distinguishes configured from not-configured feeds
    Given a machine with no managed runtimes
    When the user checks for updates
    Then the report shows there are no managed runtimes to update
    And it reports each update feed's status, marking unpublished feeds as not configured

  # The Linux fixture fills a loopback accept queue and redirects only the CLI
  # child's DNS there. These are real connect timeouts, not refused connections.
  # The short startup budget and the independent cap on a long update request
  # must both survive an upstream sync. Neither scenario installs a runtime.
  @id:update-startup-connect-timeout @requires-os:linux
  Scenario: update-02 - Startup remains bounded when metadata connections drop packets
    Given a registered read-only runtime
    When the user runs version with blackholed metadata connections
    Then the startup check records a metadata timeout

  @id:update-report-connect-timeout @requires-os:linux
  Scenario: update-03 - Checking updates fails fast even with a long request budget
    Given a registered read-only runtime
    When the user runs update with blackholed metadata connections
    Then the update report records a metadata timeout
