---
name: bootstrap
description: Set up a new Python uv project with Matt Pocock's skills, research-skills instructions, and Quard.
disable-model-invocation: true
---

# Bootstrap

Run from the target project's root. Reproduce this project's agent and style-checker setup without copying its application code. Preserve existing project files and settings; inspect a conflicting file before changing it.

1. **Check the target.** Confirm it is a Git repository with a `src/`-layout Python `uv` project (Python >= 3.12). Check that `npx`, `gh`, `uv`, a C compiler, and Rust/Cargo are available. Install missing prerequisites when possible; use the official Rust installer if Cargo is missing and put `$HOME/.cargo/bin` on `PATH`. Confirm `gh` can access `ArtFilippov/research-skills` and `ArtFilippov/quard`. Finish this step when the project and tools can support both installations.

2. **Install the agent skills.** Run `npx skills@latest add mattpocock/skills --skill '*' --agent opencode -y`. Check the installed skills under `.agents/skills/` and the generated `skills-lock.json`. Finish when the installation reports success; handle existing skills without discarding unrelated work.

3. **Fetch the instructions.** Get `AGENTS.md` and `CODING_STANDARDS.md` from the root of `ArtFilippov/research-skills` using `gh api`. Download them **directly** to the project root, for example:

   ```bash
   set -o pipefail
   gh api repos/ArtFilippov/research-skills/contents/AGENTS.md --jq .content | base64 --decode > AGENTS.md
   gh api repos/ArtFilippov/research-skills/contents/CODING_STANDARDS.md --jq .content | base64 --decode > CODING_STANDARDS.md
   ```

   Inspect any existing files first. Compare each downloaded file's `git hash-object` with the corresponding GitHub contents API `.sha`; finish when both match. The repository spells the second name `CODING_STANDARDS.md`.

4. **Install Quard.** Add `https://github.com/ArtFilippov/quard.git` as the `quard` Git submodule if absent, then run `bash quard/install.sh --project . --json` from the project root. Let the installer generate and pin the project's configuration, dependencies, and lockfile. If Git's normal credential helper cannot reach this repository but `gh` can, use a temporary per-command credential helper (`!gh auth git-credential`) for both `git submodule add` and the installer's nested `uv`/Git commands; leave the user's Git configuration intact. Finish when the installer reports `status: installed` and the smoke tests pass.

5. **Select the two exceptions.** In the generated `.quard/config.toml`, set `[checks] lcom4 = false` and `duplicate_detection = false`. Leave all other generated settings at their defaults. Run `uv run --locked --group dev python quard/apply.py --json`; finish when it reports `status: applied`.

6. **Verify and report.** Run `uv run --locked --group dev python quard/check.py --json`. Confirm that `lcom4`, `jscpd`, `jscpd-renamed`, and `jscpd-similar` have `status: skipped`. Report any other diagnostics against the project's existing code without changing more checks to hide them. Quard's Pylint duplicate-code rule remains part of the default profile even when the three clone jobs are skipped. Report the created files, submodule revision, and check result. Restart OpenCode so newly installed skills and `AGENTS.md` take effect.
