## Agent skills

### Issue tracker

Issues and specs live in GitHub Issues. Before working with them, read `docs/agents/issue-tracker.md`.

### Triage labels

This repo uses the five default triage roles as GitHub labels. Before assigning a triage status, read `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: root `CONTEXT.md` and `docs/adr/`. Before exploring the project, read `docs/agents/domain.md`.

## Python quality

Before changing Python code or analyzer configuration, read `quality/AGENTS.md`.
The strict profile, scope, and tool limitations are documented in `quality/README.md`;
repository-local runner fixes are documented in `quality/LOCAL-CHANGES.md`.

Python Quality is the project's only static analysis workflow.
From the repository root, prepare a fresh checkout with
`uv sync --locked --group dev`, then run the full static check after Python changes:

```bash
uv run --locked --group dev python quality/check.py
```

Use `.quality/reports/latest.json` for the complete diagnostics and per-tool status;
raw logs live alongside each archived report. Exit 0 means all checks passed,
1 means code/documentation/architecture violations, and 2 means a configuration
or tool execution failure. Report the actual exit code and unresolved diagnostics.

Run relevant runtime tests with `uv run --locked --group dev pytest`. When editing
`quality/`, also run
`uv run --locked --group dev python -m unittest discover -s quality/tests`.
For an environment prepared for Qwen, include `--extra qwen` in uv commands to
retain its optional dependencies.

Change readability limits in `quality/readability.toml` and shared exclusions in
`quality/scope.toml`, then run
`uv run --locked --group dev python quality/apply.py --json`.
Review module boundaries, layers, dependencies, and exports in `tach.toml`; after
changing layers, run
`uv run --locked --group dev python quality/init_architecture.py --accept --json`.
Architecture is already initialized in this repository. Do not run `tach sync`
as part of routine checks or relax rules/exclusions to hide diagnostics.
Keep `quality/` and native configs in Git; `.quality/` contains local reports/logs.
