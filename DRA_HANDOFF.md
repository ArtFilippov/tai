# Handoff: garak DRA as the next attack strategy

## Decision and status

Implement **DRA (Disguise and Reconstruction Attack)** next. This is a proposal,
not an implemented strategy. The comparison in
[`garak_only_attacks_with_base_strategy.md`](garak_only_attacks_with_base_strategy.md)
marks DRA as having a separate harmful request and disguise/reconstruction
transformation. Of the remaining candidates, it fits tai's existing
single-candidate execution path better than GOAT (multi-turn Attacker),
AgentBreaker (tool-using Target), or AutoDAN (local logits and genetic search).

Already implemented: FITD, Simple Adaptive Attacks, and Bad Characters/deletion.
Their code and usage are in `src/attacks/` and `docs/`. The DRA comparison is
based on pinned garak revision `3f50ea5ff9cd7050099940647c15c39b07a93392`;
see [`garak/garak/probes/dra.py` at that revision](https://github.com/NVIDIA/garak/blob/3f50ea5ff9cd7050099940647c15c39b07a93392/garak/probes/dra.py).

## Proposed first slice

1. Accept one explicit `Message` as the **base request**. Generate DRA prompts
   from that text using the pinned garak letter map, moderation cache, and
   prompt templates; do not substitute garak's bundled harmful-behavior corpus.
   Start with the cache-based `DRA` variant. `DRAAdvanced` optionally loads
   Detoxify and is outside this first slice.
2. Expose bounded, reproducible candidate generation (for example, a selected
   number of native templates and a documented seed). Preserve the actual
   generated prompt and its provenance in the run journal. Investigate the
   native generator's use of module-global `random` and mutation of its letter
   and moderation maps before running independent Users concurrently; avoid
   cross-run state leakage or silently changing the DRA algorithm.
3. Use `attacks.shared.candidates.CandidateRuntime`: start a fresh Target
   conversation for each candidate, publish the **full Target response** as the
   result, and call `evaluate(base, result)` with the unchanged original base.
   Stop on the first `true`; otherwise retain the last evaluated result. This
   early-stop rule is tai's policy, not garak's default batch-probe behavior.
4. Wire the strategy through `attacks.shared.contracts`,
   `app.attack_configuration`, `app.configuration`, the public Python API,
   CLI profiles, and ALERT series. Follow the `badchars_deletion` strategy's
   package/profile/bootstrap/experiment pattern and existing lifecycle and
   journal behavior. Keep the Target and evaluator conversations independent;
   DRA needs no Attacker chat.

## Acceptance checks

- Compare generated prompts against the pinned garak transformation for fixed
  inputs, templates, and randomness; cover punctuation, Unicode, newlines, and
  repeated independent runs. Document any limitation of garak's lowercase and
  cache-based behavior on arbitrary base text.
- Verify original base identity at evaluation, prompt provenance, candidate
  ordering/caps, independent conversations, first-success stop, exhaustion,
  stop during an in-flight call, and generation/evaluation errors.
- Add a CLI/ALERT integration check using a controlled Target and evaluator.
  A positive verdict on a benign request proves execution, not a jailbreak;
  assess evaluator quality separately before making effectiveness claims.
- Read `eryx/AGENTS.md` before Python edits. After implementation, run relevant
  `uv run --locked --group dev pytest` tests and the full
  `uv run --locked --group dev python eryx/check.py` from a prepared checkout;
  report exit codes and unresolved diagnostics. Use `--extra qwen` when that
  environment includes Qwen dependencies.

Architecture reminders: [`CONTEXT.md`](CONTEXT.md) distinguishes the base request,
attack result, and run journal; [`ADR 0001`](docs/adr/0001-textual-attack-contract.md)
requires a full textual result, and [`ADR 0002`](docs/adr/0002-dialogue-owned-by-llm-chat.md)
assigns conversation history to `LLMChat`.
