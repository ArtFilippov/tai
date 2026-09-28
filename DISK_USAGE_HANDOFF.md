# Handoff: shared Qwen and vLLM environment

As of 2026-09-28, the Transformers/Qwen and vLLM backends use the same root `.venv`,
root `uv.lock`, and pinned Qwen model snapshot. The old
`environments/vllm/.venv` was removed after both backends passed live preflight.
The repository measured 7.40 GB afterward, down from 12.47 GB (5.08 GB saved).

## Reproduce and run

```bash
uv sync --locked --group dev --extra qwen --extra vllm
uv run --locked --group dev --extra qwen --extra vllm pytest
uv run --locked --group dev --extra qwen --extra vllm python eryx/check.py
```

Always pass **both** extras to `uv sync` and `uv run` in this shared environment;
otherwise uv may remove one backend's optional dependencies. The vLLM example
in `examples/fitd-alert-vllm.toml` points `server.python` at the root `.venv`.
Adjust that absolute path if the checkout or `UV_PROJECT_ENVIRONMENT` changes.
Both example profiles use the same Hugging Face cache and model revision.
See `docs/vllm.md` for model preparation and live commands.

## Changes and verification

- `pyproject.toml` pins Transformers 4.45.2 and Hub 0.26.2 for Qwen, and adds
  the vLLM extra. The separate vLLM project files were removed; Eryx's saved
  extra selection in `.eryx/installation.json` now includes both backends.
- `examples/fitd-alert-vllm.toml` uses `gpu_memory_utilization = 0.7`. Its old
  value of 0.5 failed to allocate a KV cache on the occupied GPU during this
  check; 0.7 passed. This setting depends on current GPU occupancy.
- Live offline preflight passed for all three roles (Target, Attacker, Evaluator)
  with each backend. `pytest` passed 788 tests; `eryx/check.py` exited 0 with
  zero diagnostics. The Eryx integration installer also exited 0.

## Previous handoff

The obsolete `ERYX_HANDOFF.md` was removed. Current Eryx setup is documented
in `AGENTS.md`, `README.md`, and `docs/eryx-migration.md`.
