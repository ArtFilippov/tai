# Instructions for an LLM agent using this bundle

Use the project's root as the working directory. Do not edit source code to hide
a diagnostic, loosen limits without authorization, run `tach sync` on every check,
or treat a tool failure as a successful check. Native configs remain ordinary files;
there is no hash-based anti-edit system.

Read `quality/LOCAL-CHANGES.md` before updating or reinstalling the bundle. Preserve
the local runner fixes and verify them with
`uv run --locked --group dev python -m unittest discover -s quality/tests`.

## Commands

Install once: `bash /path/to/python-quality-1.0.0.run --json`.
Read `quality/README.md` before architectural initialization.

Create a draft: `uv run --locked --group dev python quality/init_architecture.py --json`.
Inspect/edit `tach.toml`; then explicitly accept:
`uv run --locked --group dev python quality/init_architecture.py --accept --json`.
A draft is not a reviewed architecture; current imports are not automatically good imports.

Change readability: edit `quality/readability.toml`, then run
`uv run --locked --group dev python quality/apply.py --json`.

Run checks: `uv run --locked --group dev python quality/check.py`.
Full machine output: append `--json`.
In a large repository prefer the short human output, then read selected entries
from `.quality/reports/latest.json` rather than repeatedly ingesting all raw logs.
Example without relying on jq:

```bash
uv run --locked --group dev python -c 'import json; r=json.load(open(".quality/reports/latest.json")); print(json.dumps([d for d in r["diagnostics"] if d["path"] == "src/shop/orders.py"], indent=2))'
```

## Result contract

0 = all checks completed and no diagnostics.
1 = code/documentation/architecture violations.
2 = configuration, missing initialization/version, analyzer crash, timeout, parser failure.
An exit code 2 must be resolved before claiming the repository passed.
`report.status`, `exit_code`, `summary`, `tools`, `diagnostics`, and `coverage`
are machine-readable. Positions are one-based; null means unavailable.
Check `tools[*].status` and `is_infrastructure_error`; retain raw logs when diagnosing failures.
Every native severity blocks success. No checks are canceled on the first failure.
JSON includes all diagnostics even when the human output is truncated.

Fix the underlying code, rerun the full check, and report the final exit code.
Check configuration is not an auto-fix tool. Docstrings are optional, but an existing
one must follow the selected Google contract. Internal comments are not globally banned.
Tests have the explicitly documented relaxed profile. Production architecture rules
remain strict. Excluded/generated directories must not become hiding places for normal code.

Do not claim that architecture community discovery, a universal Protocol-only guarantee,
physical function-line limits, formatting, security scanning or runtime tests ran: those
are not part of this bundle. This is a set of native static checks, not a correctness proof.

Installer runtime logs: `.quality/install.log`, `.quality/install-probes/`.
If installation failed, managed files are restored where possible; inspect
`.quality/install-error.json`. The uv cache/environment may need `uv sync` afterwards.
