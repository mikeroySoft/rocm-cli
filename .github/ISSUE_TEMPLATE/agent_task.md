<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

---
name: Agent task
about: Fully specified work item for the AI factory
title: ''
labels: needs-triage
assignees: ''
type: Task

---

<!--
Conventions:
- One observable change per ticket; split multi-outcome work.
- Declare dependencies in the body as "blocked by #N" — the dispatcher skips
  blocked tickets until the blocker closes.
- Add the `chore` label for mechanical work (renames, doc sync, version bumps);
  it routes to a lighter worker.
-->

**Scope**
What changes, stated as the observable outcome.

**Touches**
Files/symbols expected to change (pointers, not a contract).

**Exit gate**
Acceptance criteria: the observable condition that means done, and the exact
command that verifies it.

**Out of scope**
What this ticket explicitly does not change.
