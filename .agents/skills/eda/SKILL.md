---
name: eda
description: Exploratory data analysis (EDA). Use when the user asks to investigate supplied data, assess its quality, or discover relationships and produce an analytical notebook. Gathers evidence-backed datasets, explores through plot-it, and preserves every research plot in a reproducible Jupyter notebook in investigation order by default.
---

# EDA

Find, gather, and systematize user-provided data, investigate the question, then assemble a notebook documenting the investigation and its findings. Adapt extraction, analysis, and visual exploration to the available sources rather than prescribing data types, chart types, or a fixed profiling checklist. A request for a single plot belongs to `plot-it`.

## Artifact placement

Use one stable `<analysis-name>` for the investigation across these repository-root trees. Reuse it when continuing the same investigation; use a new name for a distinct question.

```text
datasets/<analysis-name>/
  dataset_evidence.md
  collect_*.py
  <auxiliary datasets>

notebooks/<analysis-name>/
  report.ipynb
  prepare_data.py             # when expensive additional preparation is needed
  <prepared report datasets>  # outputs of that preparation, beside the notebook
  <exploration scripts>.py
  _work/analysis_notes.md
```

For research figures and their context, follow [plot-it's default artifact placement](../plot-it/SKILL.md#default-artifacts-placement), using `<analysis-name>` as the dataset-specific name. Keep `data_context.md` only in `visualizations/<analysis-name>/`; it describes analytical meaning and interpretation limits, linking to dataset evidence for provenance. Pass this explicit context location to `plot-it`.

The preparation boundary is computational cost:

- Initial collection and systematization of the supplied sources: datasets and collection scripts in `datasets/<analysis-name>/`.
- Expensive report-specific preparation, such as assembling a new dataset from several sources, large merges, or filtering a very large source: `.py` code and its prepared outputs beside the notebook. Use `prepare_data.py`, splitting into adjacent modules when useful.
- Key analytical derivations, ordinary filtering, aggregations, and plot construction: code cells in the final notebook.

## Workflow

### 1. Frame the question and locate the sources

Identify the user's question, intended scope, and supplied sources. Follow their paths and references, inspect the relevant repository descriptions, and locate the material needed for the analysis. Resolve discoverable facts from those sources; ask about semantics or access only when they remain consequentially ambiguous.

If a question is supplied, proceed with it. Otherwise, inspect the structure and quality of the available data, propose promising questions, and agree on a focus before deeper exploration.

**Done:** The question and scope are explicit, relevant sources are located, and missing inputs that affect the answer are identified.

### 2. Build traceable analysis inputs

Gather and systematize the sources into auxiliary datasets appropriate to the question, preserving the identifiers and semantics needed to connect observations. Place collection code and outputs according to the preparation boundary above.

For each auxiliary dataset, record in `dataset_evidence.md`:

- Source locations and identities, including snapshot or version information where available.
- Collection scripts, runnable commands, parameters, and required environment.
- Assembly decisions, including joins, filtering, sampling, and exclusions.
- Resulting scope and size, collection checks, and known gaps.

Check whether the assembled data support the intended comparisons: consider observation granularity, coverage, units, keys, duplicates, missingness, and join behavior where relevant. Record unresolved limitations rather than silently assigning meanings.

Run expensive report-specific preparation separately, whenever exploration establishes a need for it. Record its inputs, transformation criteria, output identities, and reproduction command in the working notes. Keep those outputs beside the notebook and reuse them for plot design and the final report.

**Done:** Analysis inputs exist, their provenance and assembly decisions are recorded, and collection or expensive preparation can be rerun through the saved scripts.

### 3. Explore through plot-it

Call the Skill tool with `plot-it` and follow its context, encoding, rendering, inspection, and repair workflow for research plots. Keep figure files in its default placement and generating exploration code beside the future notebook. Follow its inspection delegation rules rather than reproducing that procedure here.

Use plots freely to understand the data, diagnose quality, compare alternative views, test hypotheses, and follow unexpected patterns. Let observations lead to new questions and follow-up plots within the investigation's scope. Choose chart types, comparisons, subsets, scales, and encodings as the exploration develops, without a fixed plot quota or a requirement that each plot reveal a distinct finding. For large or costly inputs, begin with summaries or reproducible samples; record their scope and check representativeness before generalizing.

Challenge important findings using relevant subsets, sensitivity to missingness and outliers, and plausible confounding factors. Distinguish descriptive associations, hypotheses, and claims supported by the checks actually performed. Return to input preparation when a new comparison needs another dataset.

Maintain `_work/analysis_notes.md` with hypotheses, methods, data subsets, results, and unresolved questions. As each research plot is used, assign it a stable identifier and record its position in the investigation, the question or observation that prompted it, figure and generating-code locations, data and parameters used, result, inspection status, and remaining caveats. Update the entry as its presentation is refined, preserving its original position.

**Done:** The main question has an evidence-backed answer or a demonstrated data limitation; important relationships have been checked; further relevant probes no longer materially change the conclusions. Scale the depth of investigation to the available data and computational cost. Every research plot is recorded in the working notes and has completed the `plot-it` workflow.

### 4. Organize the complete investigation

Include every research plot used during the investigation: initial overviews, quality diagnostics, alternative views, subset comparisons, sensitivity checks, and plots with inconclusive or negative results, including rejected hypotheses. Distinct analytical views using different data, filters, transformations, scales, or encodings belong in the notebook even when they lead to the same conclusion. For each research plot, retain its final presentation; intermediate revisions of colors, labels, or layout alone need not appear as separate plots.

By default, order the plots by when they were first used in the investigation, following the sequence recorded in the working notes. Preserve the progression from initial observations through emerging questions, hypothesis checks, and follow-up comparisons. Final presentation repairs do not change a plot's position. Use a different plot order only when the user requests it.

Open with the question and a short answer, explain transitions between plots briefly where useful, and close with conclusions and material limitations. Use these explanations to show which observation prompted the next check.

In the final notebook, favor plots over tables for distributions, comparisons, trends, and relationships. Use compact tables only when the reader needs exact values or specific records; avoid tables that duplicate what a plot already communicates.

**Done:** The outline accounts for every recorded research plot in investigation order or the user's requested order, and every substantive conclusion has supporting evidence.

### 5. Assemble the notebook last

Create `report.ipynb` from the complete investigation outline, carrying each plot's identifier into its code cell or adjacent Markdown for traceability to the working notes. Code cells load the existing inputs, perform the key analytical derivations and ordinary filtering, and construct the plots inline. Build each plot with the same data, logic, and presentation finalized through `plot-it`; showing a saved image alone does not satisfy this requirement.

- Markdown cells contain only additional information that did not fit on the plots; do not repeat their content.
- Keep Markdown compact: a few short bullet points, with a short **bold label** when needed, rather than large headings or long paragraphs.

Make paths resolvable from the notebook directory and expose relevant analysis parameters. For preparation moved into adjacent scripts, explain the sources, merge or selection criteria, prepared outputs, and reproduction command in the notebook. Execute that preparation separately: notebook execution must not trigger collection, large merges, or other expensive preparation, including through imports or missing-file fallbacks. Missing prepared inputs should identify the prerequisite and how to produce them.

The notebook need not export figure files. Reusing an inspected plot with unchanged data, logic, and presentation requires no second visual inspection. If any of these change during assembly, route only the changed plots through `plot-it` again.

**Done:** The notebook contains every recorded research plot in the planned order and the investigation's conclusions, builds its plots in code cells, and reads prepared inputs without rebuilding them.

### 6. Execute and deliver

Follow the repository's environment and execution instructions. Validate the notebook format, then execute it top to bottom in a fresh kernel using the intended environment and prepared inputs. Save the executed notebook with its outputs. Reconcile the plot identifiers against the working notes: every research plot must have generating code and a rendered output in the saved notebook, in investigation order unless the user requested otherwise. Check that the narrative agrees with the computed results and execution needs no hidden interactive state. This is report validation, not a repeat of completed visual inspections.

If execution is blocked, report the exact blocker and distinguish the draft from a verified report; do not claim completion or invent outputs. Otherwise, return the notebook path, a concise answer to the original question, the principal findings, and material limitations.

**Done:** The saved notebook executes successfully from a fresh kernel, passes the plot completeness and order check, and provides an evidence-backed answer with the relevant limitations visible.
