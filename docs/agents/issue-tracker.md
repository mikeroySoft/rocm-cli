# Issue tracker: GitHub

Issues and specs for this repo live in the `mikeroySoft/rocm-cli` fork's GitHub Issues, not the upstream `ROCm/rocm-cli` tracker. Use the `gh` CLI with `--repo mikeroySoft/rocm-cli` for all operations.

## Conventions

- **Create an issue**: `gh issue create --repo mikeroySoft/rocm-cli --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view --repo mikeroySoft/rocm-cli <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --repo mikeroySoft/rocm-cli --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment --repo mikeroySoft/rocm-cli <number> --body "..."`.
- **Apply / remove labels**: `gh issue edit --repo mikeroySoft/rocm-cli <number> --add-label "..."` / `--remove-label "..."`.
- **Close**: `gh issue close --repo mikeroySoft/rocm-cli <number> --comment "..."`.

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

When set to `yes`, PRs run through the same labels and states as issues, using `gh pr --repo mikeroySoft/rocm-cli` equivalents:

- **Read a PR**: `gh pr view --repo mikeroySoft/rocm-cli <number> --comments` and `gh pr diff --repo mikeroySoft/rocm-cli <number>`.
- **List external PRs for triage**: `gh pr list --repo mikeroySoft/rocm-cli --state open --json number,title,body,labels,author,authorAssociation,comments`, then keep only `authorAssociation` of `CONTRIBUTOR`, `FIRST_TIME_CONTRIBUTOR`, or `NONE`.
- **Comment / label / close**: use `gh pr comment`, `gh pr edit`, and `gh pr close` with `--repo mikeroySoft/rocm-cli`.

GitHub shares one number space across issues and PRs, so a bare `#42` may be either: resolve with `gh pr view --repo mikeroySoft/rocm-cli 42` and fall back to `gh issue view --repo mikeroySoft/rocm-cli 42`.

## When a skill says "publish to the issue tracker"

Create an issue in `mikeroySoft/rocm-cli`.

## When a skill says "fetch the relevant ticket"

Run `gh issue view --repo mikeroySoft/rocm-cli <number> --comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: an issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue. Where sub-issues aren't enabled, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels: `wayfinder:<type>` (`research`, `prototype`, `grilling`, or `task`). Once claimed, assign the ticket to the driving developer.
- **Blocking**: use GitHub native issue dependencies. Add an edge with `gh api --method POST repos/mikeroySoft/rocm-cli/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric database id. Where dependencies aren't available, use a `Blocked by: #<n>, #<n>` line.
- **Frontier query**: list the map's open children, then drop any with an open blocker or assignee; first in map order wins.
- **Claim**: `gh issue edit --repo mikeroySoft/rocm-cli <n> --add-assignee @me`, the session's first write.
- **Resolve**: comment with the answer, close the child, then append a context pointer to the map's Decisions-so-far.
