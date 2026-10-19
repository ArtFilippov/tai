# Issue tracker: GitHub

Issues and specs live in GitHub Issues. Use `gh` from this repository.

## Conventions

- Create: `gh issue create --title "..." --body "..."`.
- Read: `gh issue view <number> --comments`; fetch labels and filter comments with `jq` as needed.
- List: `gh issue list --state open --json number,title,body,labels,comments` with appropriate filters.
- Comment: `gh issue comment <number> --body "..."`.
- Label: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`.
- Close: `gh issue close <number> --comment "..."`.

Infer the repository from `git remote -v`; `gh` does so in this clone.
When a skill says to publish to the issue tracker, create an issue.
When a skill says to fetch a ticket, use `gh issue view <number> --comments`.

## Pull requests as a triage surface

**PRs as a request surface: no.**

If switched to `yes`, triage external PRs using `gh pr` equivalents. Read
with `gh pr view <number> --comments` and `gh pr diff <number>`. List open
PRs and keep authors with `authorAssociation` of `CONTRIBUTOR`,
`FIRST_TIME_CONTRIBUTOR`, or `NONE`. Resolve ambiguous `#<number>` with
`gh pr view` before falling back to `gh issue view`.

## Wayfinding operations

A map is one issue labelled `wayfinder:map`. Link child tickets as GitHub
sub-issues; if unavailable, use a task list in the map and `Part of #<map>`
in each child. Label children `wayfinder:<type>` (`research`, `prototype`,
`grilling`, or `task`).

Use native issue dependencies for blockers: POST to
`repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by` with
`issue_id=<blocker-db-id>` (the blocker's numeric database `id`, obtained
from `gh api repos/<owner>/<repo>/issues/<n> --jq .id`). If unavailable,
add `Blocked by: #<n>` to the child body. A child is unblocked when all
blockers are closed. Find the first open, unassigned, unblocked child in
map order; claim it with `gh issue edit <n> --add-assignee @me`. Resolve
by commenting with the answer, closing the child, and adding a context
pointer (gist and link) to the map's Decisions-so-far.
