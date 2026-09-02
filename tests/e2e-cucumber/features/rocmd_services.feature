Feature: rocmd managed-service stop

  # rocmd stops a managed service by the PIDs persisted in its record. PIDs are
  # recycled by the kernel, so each PID is paired with its start-time identity
  # and only signalled when the identity still matches. Start-time identity is
  # read from /proc, so both scenarios are Linux-only.

  @id:rocmd-stop-refuses-recycled-pid @requires-os:linux
  Scenario: 1 - Stopping a service whose PID was recycled leaves the new process alone
    Given a managed service record whose PID now belongs to an unrelated process
    When the user stops the service through rocmd
    Then the unrelated process is still running
    And rocmd reports the PID as skipped, not signaled
    And the service record is marked stopped, since the recorded process is gone

  @id:rocmd-stop-terminates-matching-pid @requires-os:linux
  Scenario: 2 - Stopping a service whose PID identity matches terminates it
    Given a managed service record pointing at a live process it owns
    When the user stops the service through rocmd
    Then the owned process is no longer running
    And rocmd reports the PID as signaled
    And the service record is marked stopped
