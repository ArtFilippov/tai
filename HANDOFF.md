# Session handoff: garak-only attacks with a separate base and strategy

## Scope and status

The user asked which attacks unique to the studied garak implementation separate
the base request or goal from the attack strategy. The source is
[`garak_only_attacks_with_base_strategy.md`](garak_only_attacks_with_base_strategy.md),
whose **✅ Yes** classification yields the seven entries below. “Garak-only”
means the **specific mechanism or ready-made composition** was not found in the
studied PyRIT snapshot, not that PyRIT cannot express a similar attack. The
classification describes garak; the status column describes implementation
in **tai**. Partial (◐) and no (❌) entries are outside this list.

| Attack in garak | Separation and important qualification | tai status |
| --- | --- | --- |
| AutoDAN | `goal_str` is passed separately to the GA/HGA strategy. **AutoDANCached does not qualify**: it uses prebuilt prompts. Requires local logits; the comparison flags a mismatch between fitness and final-prompt construction. | **Not implemented** |
| AgentBreaker | Agent goal/tool is separate from the exploit-building strategy. This is domain-specific to a tool-using Target and uses direct prompting. | **Not implemented** |
| DRA | Harmful behavior/payload is separate from disguise-and-reconstruction transformation *inside the probe*; ordinary `DRA` uses bundled behaviors by default, while `DRAAdvanced` accepts a custom sentence. | **Not implemented** |
| Bad Characters / deletion | Original payload is separate from the printable-character-plus-backspace perturbation *inside the probe*. Tai supplies its explicit base instead of garak's bundled corpus. | **Implemented**: [`src/attacks/badchars/`](src/attacks/badchars/), [`docs/badchars-deletion.md`](docs/badchars-deletion.md) |
| GOAT | `attack_goal` is separate from the adaptive Observation–Thought–Strategy–Response protocol. Requires an Attacker and multi-turn Target conversation. | **Not implemented** |
| FITD | Target request is separate from the escalation/refusal-handling strategy. | **Implemented**: [`src/attacks/fitd/`](src/attacks/fitd/), [`docs/fitd-operations.md`](docs/fitd-operations.md) |
| Simple Adaptive Attacks | Seed goal is inserted separately into a template with cached suffixes; **there is no online adaptive search** in this probe. | **Implemented**: [`src/attacks/adaptive_attacks/`](src/attacks/adaptive_attacks/), [`docs/adaptive-attacks.md`](docs/adaptive-attacks.md) |

No next strategy has been selected. An earlier DRA-first suggestion was
exploratory; the user explicitly requested **no fixed DRA next step**. The
user deleted `DRA_HANDOFF.md`; do not depend on it or restore it. The full
comparison above contains the native garak source links and per-attack caveats.

## Session and continuation

- `164d49d` committed the earlier DRA-specific handoff (without running tests,
  as requested). `cd99ea6` renamed the existing shared-environment handoff to
  [`DISK_USAGE_HANDOFF.md`](DISK_USAGE_HANDOFF.md). This root `HANDOFF.md` is the
  current session handoff; its creation was not committed in this session.
- Before choosing or implementing another strategy, check `CONTEXT.md`,
  `docs/agents/domain.md`, relevant `docs/adr/`, and the pinned garak source
  linked from the comparison. The existing `src/attacks/shared/` runtime and
  implemented strategies show the base/result/evaluator contract.
- Before changing Python or analyzer configuration, read `eryx/AGENTS.md` and
  follow the checks in root `AGENTS.md`. For this shared Qwen/vLLM environment,
  use both uv extras as described in `DISK_USAGE_HANDOFF.md`. A positive model
  verdict for a benign base demonstrates execution, not jailbreak efficacy.
- At the last status check, `.agents/` and `skills-lock.json` were unrelated
  untracked work; leave them alone. The deletion of `DRA_HANDOFF.md` was the
  user's change. No implementation or tests were performed for this handoff.

## Suggested skills

- `implement` — when the user selects a strategy to implement.
- `diagnosing-bugs` — if a pinned garak integration fails during implementation.
