# Next garak attack — handoff

## Where the conversation stands

The user initially asked to implement the next attack from `garak_only_attacks.md` and identify which remaining attacks can separate a base request from a strategy. They then clarified the immediate task: **rank the remaining attacks by ease of that separation; choose the plan and next attack afterward**. Do not assume an attack has been selected. **Exclude AgentBreaker**: the project is interested in plain LLMs, without agent/tool scaffolding. No implementation or repository edits were made during the ranking discussion.

The previous answer recommended **ThreatenJSON** if the base must be an arbitrary textual request, with **FalseAssertion** as a close alternative if a domain-specific base (a known false claim) is acceptable. The user has not chosen between them.

## Findings to carry forward

- The authoritative comparison and implementation-specific classification are `garak_only_attacks.md` and `garak_only_attacks_with_base_strategy.md`. The latter describes *existing garak separation*, not necessarily how easy a new adaptation would be.
- Already implemented here: FITD, Simple Adaptive Attacks, Bad Characters/deletion, DRA, GOAT, and AutoDAN. See `src/attacks/` and `README.md`.
- The ordered shortlist from easiest to hardest to separate, **excluding AgentBreaker and those six implementations**, was:
  1. ThreatenJSON — arbitrary JSON-answerable base request plus JSON-only pressure; requires adapting a fixed garak prompt.
  2. FalseAssertion — known false assertion plus framing templates; the base is not an arbitrary request.
  3. Product keys / Grandma — product name plus roleplay; Win10/Win11 remain key-format-specific. `grandma.GrandmaIntent` separately accepts a textual stub but is not the same Win10/Win11 check.
  4–6. EICAR/GTUBE/GTphish, Repeat divergence, Glitch tokens — easy to parameterize by signature, repeated word, or token, but arbitrary substitution changes the test's meaning.
  7–9. Leak replay, ProPILE, Slur completion — require structured source data **and an expected value** for their specialized checks.
  10–11. WHOIS injection, WordNet — need WHOIS context/verdict or a topic word; WordNet additionally requires a lexicon and response-dependent search.
  12. BEAST — arbitrary request plus suffix is conceptually clear, but pinned garak's suffix search does not use its probe's goal and requires local logits.
  13–16. Snowball, WhoIsRiley, Deadnaming, atkgen.Tox — fixed factual/regression targets or a toxicity objective embedded in the attack model.
  17–18. HF model file formats, ANSI tokenizer vocabulary — artifact audits without an LLM request.
- This ordering balances mechanical ease with preservation of the original test. In particular, a simple string substitution is not automatically a faithful implementation of the named garak check.
- For ThreatenJSON, the installed pinned source is `.venv/lib/python3.12/site-packages/garak/probes/goodside.py`; its `ThreatenJSON.prompts` contains one fixed request for a two-key JSON dictionary and the coercive text. For FalseAssertion, see the installed `garak/probes/misleading.py`: three framing prefixes applied to a corpus of false claims. The source checkout linked from the comparison is not present at the repo root; the installed garak package is available under `.venv/`.
- Project vocabulary in `CONTEXT.md`: a User executes a strategy from a **base request** (`Message`); the strategy presents a complete textual **attack result**, evaluated against that base. A run journal is distinct from the result. ADRs `docs/adr/0001-textual-attack-contract.md` and `0002-dialogue-owned-by-llm-chat.md` constrain implementation.

## Next action

Ask the user to choose a next attack and decide whether its base may be domain-specific or must accept an arbitrary `Message`. Then agree on the strategy's result and evaluation semantics before implementation. For a chosen attack, inspect the pinned garak source and existing attack wiring (`src/attacks/`, `src/app/attack_configuration.py`, `src/app/configuration.py`, `src/tai.py`) and make a focused implementation plan.

## Suggested skills and repository checks

- When asked to implement the selected work from an agreed spec, load **`implement-spec`** (or **`implement`** for a set of tickets).
- When the separation itself becomes a module-interface design question, load **`codebase-design`**.
- Before exploring, follow `docs/agents/domain.md`; before Python or analyzer changes, follow `quard/AGENTS.md`, `quard/PROFILE.md`, and `quard/COMPATIBILITY.md`. If using GitHub issues/specs, read `docs/agents/issue-tracker.md` first.
- After Python changes, run the relevant runtime tests and the full Quard command required by root `AGENTS.md`. Check working-tree status before editing; it was clean when this handoff was created.
