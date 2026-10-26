# Log Directory Structure (Functional Testing)

## Overview

The `log_dir` directory contains functional testing results for generative models. The structure consists of individual JSON files corresponding to specific datasets, each storing aggregated metrics and execution time. Global metadata, including MLOps task information, is consolidated in the root `info.json` file.

---

## Directory Structure

```text
log_dir/
├── info.json                     # Global experiment and task metadata
├── ruMMLU.json                   # Results for ruMMLU dataset
├── human_eval.json               # Results for HumanEval dataset
├── mmlu_en.json                  # Results for MMLU (English) dataset
└── ...                           # Other dataset result files
```

## Root Metadata: `info.json`

Located at the top level of `log_dir`. Contains global information about the experiment environment.

### Structure

```json
{
    "logs_type": "generative_models_functional_test",
    "mlops": "local"
}
```

### Fields
- `logs_type`: Indicates the type of logs generated. Expected value: `generative_models_functional_test`.
- `mlops`: Specifies the MLOps platform used for execution. Possible values: `local`, `mlm`, `clearml`, `mlflow`.

## Dataset Result File Structure

Each JSON file in the root directory corresponds to a single dataset evaluation. The filename typically matches the dataset name (e.g., `ruMMLU.json`).

### Structure

```json
{
    "task_name": "ruMMLU",
    "results": {
        "acc": 0.26755189255189255
    },
    "time": 1608.6893680095673
}
```

### Fields
-   `task_name` (string): Identifier of the dataset or benchmark task. Must match the semantic meaning of the filename.
-   `results` (object): Key-value pairs of calculated metrics.
    -   Keys represent metric names (e.g., `acc`, `f1`, `bleu`).
    -   Values are numeric results (integers or floats).
-   `time` (float): Total execution time for the evaluation on this dataset, in seconds.

### Notes
-   One file equals one dataset evaluation.
-   Metric keys inside `results` may vary between different dataset files depending on the benchmark specification.
-   Files are stored flat in the root directory; no subdirectories are used for dataset results.