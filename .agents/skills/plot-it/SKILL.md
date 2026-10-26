---
name: plot-it
description: Data visualization. Use when creating, revising, or inspecting plots.
---

# Plot It

Scientific and analytical plots.

## Default artifacts placement

**Context:** `<repo-root>/visualizations/<dataset-specific-name>/data_context.md`

**Figures:** `<repo-root>/visualizations/<dataset-specific-name>/[<plot-groups-name>/]<plot-name>.<ext>`

`[<plot-groups-name>/]` is optional. All plot groups for the dataset share its root `data_context.md`.

## Steps

1. **Context.** A data context is a file describing a dataset’s sources, structure, semantics, caveats, and assumptions.

   If the user explicitly requests a new data context, skip report lookup. Otherwise, use context supplied by the user first, or search the location supplied by the user or documented in applicable `AGENTS.md` or `README` files. If no location is specified, or it is marked as the default, use [Default artifacts placement](#default-artifacts-placement).

   Reuse a report whose sources and scope match the task, supplementing it where needed.

   When no matching report is available or a new one is explicitly requested, delegate creation to a fresh subagent following [context.md](context.md). Give it the data sources, task goal, available descriptions, and placement information, including the default placement and whether a search location has been defined. Read the saved report before continuing.

2. **Routing.**
   - New figure, or changes to data, the figure question, or visual encoding: Encoding → Render → Inspection.
   - Presentation-only changes: Render → Inspection.
   - Inspection-only requests with an existing figure: Inspection.
   - On request, add purpose-driven callouts to an existing figure.

3. **Encoding.** Follow [encoding.md](encoding.md).

4. **Render.** Follow local conventions for plotting code and artifact placement. Implement and execute the plotting code.

   **Done:** The current plotting code has produced a saved figure in the selected location, and local code conventions have been followed.

5. **Inspection.**
   Delegate inspection to a fresh subagent. Give it only the figure and instructions to follow [inspection.md](inspection.md).

   Reconcile the findings with the context and plotting code, accounting for every reported data pitfall:
   - Remove findings shown to be unfounded by the context or plotting code.
   - Remove findings already accepted as assumptions. Use assumptions explicitly documented in the context or accepted by the user in the current discussion; briefly list these removed findings as assumptions in a separate **Accepted assumptions** section of the report.
   - Report the remaining applicable findings and data caveats.

   **Done:** Findings have been reconciled and reported. Inspection-only requests finish here. For creation or revision, proceed to Repair → Re-render when the remaining findings require changes to the figure; otherwise finish with the inspected figure and generating code saved.

6. **Repair → Re-render.** Repair the issues requiring figure changes in the plotting code, render again, and repeat Inspection.

   **Done:** The inspected figure and generating code are saved. Display issues requiring correction are resolved; remaining data caveats are surfaced to the operator.
