# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Repository and access

Resolve the repository from `git remote -v`; the current target is `ArtFilippov/tai`.
Run commands inside this clone, or pass `--repo ArtFilippov/tai` explicitly.
The `gh` CLI must be installed and authenticated (`gh auth login`).

## Conventions

- Create an issue: `gh issue create --title "..." --body "..."`.
  For multi-line bodies, use `--body-file <path>`.
- Read an issue and its comments: `gh issue view <number> --comments`.
  For structured data including labels, use
  `gh issue view <number> --json number,title,body,labels,comments`.
- List issues: `gh issue list --state open --json number,title,body,labels,comments`,
  with appropriate `--label` and `--state` filters.
- Comment: `gh issue comment <number> --body "..."`.
- Apply or remove labels: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`.
  Triage roles are mapped in `docs/agents/triage-labels.md`.
- Close: `gh issue close <number> --comment "..."`.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## Tracker operations

- "Publish to the issue tracker": create a GitHub issue.
- "Fetch the relevant ticket": run `gh issue view <number> --comments`.

## Wayfinding operations

Used by `/wayfinder`. The map is a single issue with child issues as tickets.

- Map: an issue labelled `wayfinder:map`, holding Notes / Decisions-so-far / Fog.
- Child ticket: link it to the map as a GitHub sub-issue using `gh api`.
  If sub-issues are unavailable, add it to a task list in the map and put
  `Part of #<map>` at the top of the child body.
  Label it `wayfinder:<type>` (`research`, `prototype`, `grilling`, or `task`).
- Blocking: use native GitHub issue dependencies:
  `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`.
  Obtain the blocker's database ID with
  `gh api repos/<owner>/<repo>/issues/<number> --jq .id`.
  If dependencies are unavailable, use `Blocked by: #<n>, #<n>` in the child body.
  A ticket is unblocked when every blocker is closed.
- Frontier: list the map's open children; exclude assigned tickets and tickets
  with open blockers. For native dependencies, check
  `issue_dependencies_summary.blocked_by`; otherwise resolve the `Blocked by` line.
  Choose the first eligible ticket in map order.
- Claim: `gh issue edit <number> --add-assignee @me`, the session's first write.
- Resolve: comment with the answer, close the child issue, then append
  a summary and link to Decisions-so-far in the map.
