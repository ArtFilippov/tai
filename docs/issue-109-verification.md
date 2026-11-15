# Issue #109 verification

Verified on 2026-11-15 against [#109](https://github.com/ArtFilippov/tai/issues/109)
and its parent [#106](https://github.com/ArtFilippov/tai/issues/106).

## Catalog and execution

- The unchanged garak pin is `3f50ea5ff9cd7050099940647c15c39b07a93392`.
- All 14 named, 127 ablation and 666 wild entries match native loading/rendering.
  Ablation parity uses the pinned native implementation in an isolated subprocess
  with initialization and probe orchestration disconnected, rather than another
  implementation of the new composition algorithm. Target names containing braces
  are included. Wild entries retain duplicates and zero-based source indices.
- Every nonempty section subset preserves canonical named → ablation → wild order.
  Omitted selection contains all 807 source entries.
- Offline standalone and ALERT coverage reaches finished, recovered disk journals:
  individual sections, combined selection, cross-section early success, duplicate
  winner identity, final negative assessment, and execution/assessment/source errors.
  Bounded collection fixtures keep orchestration checks separate from full-catalog
  fidelity checks. ALERT retains one attempt per base and its later-attempt policy.
- Real tiny CPU Qwen weights and a local tokenizer reproduce native context overrun
  and verify explicit rejection. An end-to-end standalone check retains completed
  activation when the second exchange exceeds context. Input and the full configured
  generation allowance must fit; neither is shortened.

## Final checks

| Command | Exit | Result |
| --- | --- | --- |
| `git submodule update --init --recursive` | 0 | Pinned submodules prepared |
| `uv sync --locked --group dev` | 0 | Locked environment prepared |
| `uv run --locked --group dev pytest` | 0 | 328 passed, 1 deselected, 2 warnings |
| `uv run --locked --group dev python quard/check.py` | 0 | 0 diagnostics; no infrastructure failures |
| `git diff --check` | 0 | No whitespace errors |

The deselected test is the opt-in live-Qwen test. The warnings concern the existing
`TestModel` and `TestLogits` fixture classes with constructors. No live GPU benchmark
was required. Quard's existing configuration skips LCOM4 and the three clone jobs;
enabled jobs passed. The complete static report is `.quard/reports/latest.json`.

## Standards review

No remaining documented-standard violations. Review fixes snapshot catalog selection
and preserve resource/index context in wild loading/rendering errors.

Possible follow-up issue: centralize the small duplicated section-validation policy
in command validation and catalog assembly; this is a nonblocking design smell.

## Spec review

No remaining actionable specification gaps after adding explicit native Qwen context
rejection and real-model regression coverage.

Review totals: standards — 0 hard findings, 1 deferred heuristic; spec — 0 findings.
Unresolved checker diagnostics: none.
