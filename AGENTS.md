## Agent skills

### Issue tracker

Issues and specs live in GitHub Issues. Before working with them, read `docs/agents/issue-tracker.md`.

### Triage labels

This repo uses the five default triage roles as GitHub labels. Before assigning a triage status, read `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: root `CONTEXT.md` and `docs/adr/`. Before exploring the project, read `docs/agents/domain.md`.

## Eryx

Before changing Python code or analyzer configuration, read `eryx/AGENTS.md`.
The strict profile, scope, and tool limitations are documented in `eryx/PROFILE.md`;
runner compatibility fixes are documented in `eryx/COMPATIBILITY.md`.

Eryx is the project's only static analysis workflow, pinned as the `eryx/` submodule.
From the repository root, prepare a fresh checkout with
`git submodule update --init --recursive` and `uv sync --locked --group dev`,
then run the full static check after Python changes:

```bash
uv run --locked --group dev python eryx/check.py
```

Use `.eryx/reports/latest.json` for the complete diagnostics and per-tool status;
raw logs live alongside each archived report. Exit 0 means all checks passed,
1 means code/documentation/architecture violations, and 2 means a configuration
or tool execution failure. Report the actual exit code and unresolved diagnostics.

Run relevant runtime tests with `uv run --locked --group dev pytest`. When editing
`eryx/`, also run
`uv run --locked --group dev python -B -m unittest discover -s eryx/tests`.
For an environment prepared for Qwen, include `--extra qwen` in uv commands to
retain its optional dependencies.

Change readability limits in `.eryx/readability.toml` and shared exclusions in
`.eryx/scope.toml`, then run
`uv run --locked --group dev python eryx/apply.py --json`.
Review module boundaries, layers, dependencies, and exports in `tach.toml`; after
changing layers, run
`uv run --locked --group dev python eryx/init_architecture.py --accept --json`.
Architecture is already initialized in this repository. Do not run `tach sync`
as part of routine checks or relax rules/exclusions to hide diagnostics.
Keep the gitlink, `.gitmodules`, `.eryx/` settings and native configs in Git.
Only `.eryx/reports/`, `.eryx/logs/`, `.eryx/cache/` and native caches are ignored.
Install/update integration with `bash eryx/install.sh --project . --extra qwen`
when retaining this Qwen environment. All Eryx commands accept `--project`.

Tool fixes require a branch and PR in `ArtFilippov/eryx`, autonomous regression
tests and consumer checks. After that PR merges, open a separate tai PR pinning
the merged SHA and linking the tool PR. See `eryx/README.md` for the full workflow.
