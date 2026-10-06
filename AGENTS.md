## Agent skills

### Issue tracker

Issues and specs live in GitHub Issues. Before working with them, read `docs/agents/issue-tracker.md`.

### Triage labels

This repo uses the five default triage roles as GitHub labels. Before assigning a triage status, read `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: root `CONTEXT.md` and `docs/adr/`. Before exploring the project, read `docs/agents/domain.md`.

## Quard

Before changing Python code or analyzer configuration, read `quard/AGENTS.md`.
Consult `quard/PROFILE.md` for analyzer policy and `quard/COMPATIBILITY.md` for
version-specific compatibility. Quard is the project's only static analysis
workflow, pinned as the `quard/` submodule. From the repository root, prepare a
fresh checkout with `git submodule update --init --recursive` and
`uv sync --locked --group dev`, then run the full static check after Python changes:

```bash
uv run --locked --group dev python quard/check.py
```

Use `.quard/reports/latest.json` for complete diagnostics and per-tool status;
raw logs live alongside each archived report. Exit 0 means all checks passed,
1 means violations, and 2 means configuration or tool failure. Report the actual
exit code and unresolved diagnostics. Run relevant runtime tests with
`uv run --locked --group dev pytest`. When editing `quard/`, also run
`uv run --locked --group dev python -B -m unittest discover -s quard/tests`.
For a Qwen environment include `--extra qwen` in uv commands; for a shared
Qwen/vLLM environment include both `--extra qwen --extra vllm`.

Edit readability, scope and optional checks in `.quard/config.toml`, then run
`uv run --locked --group dev python quard/apply.py --json`. This project uses
reviewed, consumer-owned `tach.toml` boundaries. Review modules, layers,
dependencies and exports before accepting changed layers with
`uv run --locked --group dev python quard/init_architecture.py --accept --json`.
Keep the gitlink, `.gitmodules`, `.quard/` settings and native configs in Git;
only `.quard/reports/`, `.quard/logs/`, `.quard/cache/` and native caches are ignored.
Run `bash quard/install.sh --project . --extra qwen` to update this environment;
include `--extra vllm` when provisioning a shared environment. All Quard
commands accept `--project`.

## Changes

Keep changes focused and minimal.
Leave puzzles for later, following Puzzle Driven Development.
Flag smells and refactoring, suggesting issues rather than fixing silently.
Fix style violations in code, trusting checkers rather than suppressing them.

## Code

Favor aesthetics over functionality.
Prepend classes with plain-text docblocks explaining purpose, not usage.
Name variables as single nouns.
Inline single-use expressions, dropping unnecessary local variables.
Name methods as single nouns or verbs, separating commands from queries.

## Objects

Honor Elegant Objects and SOLID.
Follow DDD (domain-driven design) and YAGNI (you arent gonna need it).
Apply SRP (single-responsibility) and DRY (dont repeat yourself).
Favor failing immediately over failing silently.
Allow only assignments in constructor.
Compose objects instead of inheriting implementation.
Reject -er names and utility classes.

## Methods

Drop static methods, public static literals, setters, and getters.
Declare methods in interfaces first, exposing nothing public outside them.
Reject null returns and arguments.
Skip type introspection, casting, and reflection.

## Tests

Follow "Angry Tests".
Place one assertion per test, as its last statement.
Phrase each failure message negatively.
Keep tests short, verifying one behavior with one statement.

## Fixtures

Provide custom arguments to tested objects, not defaults.
Skip irrelevant code, side effects, accessors, constructors, and errors.
Favor fakes and stubs over mocks.
Inline brief fixtures, generating bulky ones at runtime.
Create fixture objects against duplication, using temporary directories.
Keep filesystem, sockets, and memory unmocked.


## English

After every user command written in English, check its grammar before doing anything else.
- If there are grammatical errors, start your response with: `Better say: <corrected version>`
- Preserve the user's original meaning and phrasing as much as possible.
- If the grammar is already correct, start your response with: `Great grammar`
- If there is reasonable doubt about the intended meaning and correcting the grammar would require guessing the user's intent, ask a clarifying question instead of assuming the meaning.
- After the grammar feedback or clarification, proceed with the user's request normally.
Keep your own responses extremely concise, simple, and easy to understand.
