# Log Directory Structure

## Overview

The `log_dir` directory contains logs, artifacts, metrics, and parameters for testing generative models. 

Pipeline stages include:
- check methods
- attacks 

---

## Directory Structure

```text
log_dir/
├── info.json                     # Global experiment metadata
|
├── description_for_report/       # Report generation metadata
│   ├── metrics.json              # Metric definitions and descriptions
│   ├── parameters.json           # Parameter definitions and descriptions
│   └── problem_types.json        # Problem type mappings
|
├── check_methods/                # Check methods logs
│   ├── mlops_task_info.json      # Task-specific metadata (if available)
│   ├── artifacts/                # Artifacts for check methods
│   │   └── req.json               # Information about implemented wrapper methods
│   ├── metrics.json              # Runner metrics
│   └── runner_params.json        # Runner parameters
│
├── attack1/                      # Attack directory (single run)
│   ├── mlops_task_info.json      # Task-specific metadata (if available)
│   ├── artifacts/
│   │   ├── example1.png
│   │   └── ...
│   ├── metrics.json              # Hierarchical metrics structure
│   └── runner_params.json        # Attack parameters
│
├── attack2/                      # Attack directory (single run)
│   ├── mlops_task_info.json      # Task-specific metadata (if available)
│   ├── artifacts/
│   │   └── ...
│   ├── metrics.json
│   └── runner_params.json
│
└── ...
```

## Root Metadata: `info.json`

Located at the top level of `log_dir`. Contains global information about the experiment environment.

### Structure

```json
{
    "logs_type": "generative_models_test",
    "mlops": "local"
}
```

### Fields
- `logs_type`: Indicates the type of logs generated. Expected value: `generative_models_test`.
- `mlops`: Specifies the MLOps platform used for execution. Possible values: `local`, `mlm`, `clearml`, `mlflow`.

## Report Description Metadata: `description_for_report/`

This directory contains JSON files that define the structure, semantics, and human-readable descriptions for metrics, parameters, and problem types used in the experiment reports.

### `metrics.json`

Contains definitions for all possible metrics, grouped by problem type.

#### Structure Example
```json
{
    "text_to_text": {
        "ASR_": {
            "name": "ASR_",
            "description": "ASR (Attack Success Rate) — доля успешных атак среди всех предпринятых попыток.",
            "higher_better": false,
            "range": [0.0, 1.0]
        },
        ...
    },
    ...
}
```

#### Fields
- **Top-level keys**: Problem types (e.g., `text_to_text`).
- **Second-level keys**: Metric identifiers (e.g., `ASR_`).
- **Metric Object**:
  - `name`: Display name of the metric.
  - `description`: Detailed explanation of what the metric measures.
  - `higher_better`: Boolean indicating if a higher value is better (`true`), worse (`false`), or neutral (`null`).
  - `range`: Array `[min, max]` defining the valid range of values. `null` indicates no upper bound.

### `parameters.json`

Contains definitions for attack parameters, grouped by attack category and specific attack name.

#### Structure Example
```json
{
    "jailbreak": {
        "direct_attack": {
            "max_token": {
                "name": "Максимальное число токенов",
                "description": "Величина максимального числа токенов ..."
            },
            ...
        },
        "multiturn": {
            ...
        }
    },
    ...
}
```

#### Fields
- **Top-level keys**: Attack categories (e.g., `jailbreak`).
- **Second-level keys**: Specific attack names (e.g., `direct_attack`, `multiturn`).
- **Third-level keys**: Parameter names (e.g., `max_token`).
- **Parameter Object**:
  - `name`: Human-readable name of the parameter.
  - `description`: Detailed explanation of the parameter's role.

### `problem_types.json`

Maps internal problem type identifiers to their human-readable Russian names.

#### Structure Example
```json
{
    "Text to text": "Генерация текста",
}
```

## `req.json` Structure

Located in the `check_methods/artifacts/` directory. This file lists the methods available for execution and identifies those that are currently unavailable due to missing interfaces.

### Structure

```json
{
    "available": [
        "hop_skip_jump",
        "sign_opt",
        ...
    ],
    "unavailable": {
        "pgd": [
            "get_grad",
            "get_loss",
            ...
        ],
        "deepfool": [
            "get_grad",
            "get_loss",
            ...
        ],
        ...
    }
}
```

### Fields
- `available`: A list of strings representing the names of methods that are fully implemented and ready for execution.
- `unavailable`: An object where:
  - Keys are the names of methods that cannot be run.
  - Values are lists of strings describing the specific interfaces or dependencies missing for each method.

## Run Metadata: `mlops_task_info.json`

Located inside each stage directory (e.g., `check_methods/`) or attack directory (e.g., `attack1/`). Contains data about the specific task if provided to the tool. This file may be absent if no task data was transmitted.

## `metrics.json` Structure (Generative Models)

Stores evaluation metrics in a hierarchical structure based on question categories and subtopics.

### Structure Logic
1. **Top Level**: Contains keys for each base question category (e.g., `guns`, `violence`) and a special key `results` for aggregated metrics across all categories.
2. **Category Level**: Each category key contains sub-keys for specific subtopics.
3. **Subtopic Level**: Contains the actual calculated metrics as key-value pairs.

### Example

```json
{
    "results": {
        "accuracy": 0.85,
        "safety_score": 0.92
    },
    "guns": {
        "regulations": {
            "accuracy": 0.88,
            "safety_score": 0.95
        },
        "history": {
            "accuracy": 0.82,
            "safety_score": 0.90
        }
    },
    "violence": {
        "physical": {
            "accuracy": 0.80,
            "safety_score": 0.88
        },
        "psychological": {
            "accuracy": 0.84,
            "safety_score": 0.91
        }
    }
}
```

Notes:
- `results`: Aggregated metrics for all questions/categories.
- Category keys (e.g., `guns`, `violence`): Metrics specific to that base category.
- Subtopic keys (e.g., `regulations`, `physical`): Metrics specific to that subtopic.
- Metric values can be integers, floats or strings (textual descriptions, e.g., outputs from "LLM as a judge" evaluations or qualitative summaries).

## `runner_params.json` Structure

Stores parameters used for a specific run.

```json
{
    "verbose": true,
    "max_tokens": 100,
    "top_p": 0.9,
    ...
}
```

Notes:
- Keys represent parameter names.
- Values are parameter values used in the run.

## Attack Experiments

- Each `attack*` directory contains a single run.
- Unlike predictive models, there are no sub-directories with integer indices for parameter sweeps within the attack folder itself.
- Parameters for the attack are stored in `runner_params.json`.
