"""Local checkpoint identity and validated Qwen generation parameters."""

import math
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


class QwenOptions(TypedDict, total=False):
    dtype: str
    system: str
    generation: Mapping[str, object] | None


class QwenError(RuntimeError):
    """Local preparation, device, or model contract failure (no fallback)."""


class QwenContextError(QwenError):
    """The complete input plus requested output exceeds the context window."""


type Generation = dict[str, int | float | bool]


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value: object, name: str, maximum: float = math.inf) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be positive and finite (top_p <= 1)")
    if not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError(f"{name} must be positive and finite (top_p <= 1)")
    return value


def generation_config(parameters: Mapping[str, object] | None) -> Generation:
    values = dict(parameters or {})
    sampling = {"temperature", "top_p", "top_k"}
    if values.keys() - (sampling | {"max_new_tokens", "do_sample", "repetition_penalty"}):
        raise ValueError("unsupported Qwen generation parameters; see README schema")
    sample = values.get("do_sample", False)
    if not isinstance(sample, bool):
        raise ValueError("do_sample must be a boolean")
    if not sample and sampling & values.keys():
        raise ValueError("temperature/top_p/top_k require do_sample=true")
    result: Generation = {
        "max_new_tokens": _integer(values.get("max_new_tokens", 512), "max_new_tokens", 1),
        "do_sample": sample,
        "repetition_penalty": _positive(values.get("repetition_penalty", 1.0), "repetition_penalty"),
        "num_beams": 1, "num_return_sequences": 1,
        "use_cache": True, "return_dict_in_generate": False,
    }
    if sample:
        result.update({
            "temperature": _positive(values.get("temperature", 1.0), "temperature"),
            "top_p": _positive(values.get("top_p", 1.0), "top_p", 1),
            "top_k": _integer(values.get("top_k", 50), "top_k", 0),
        })
    return result


def _identifier(value: object, pattern: str, reason: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError(reason)
    return value


def validate_dtype(dtype: str) -> str:
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("dtype must be float16, bfloat16, or float32")
    return dtype


def validate_system(system: object) -> str:
    if not isinstance(system, str):
        raise ValueError("system must be a string")
    return system


@dataclass(frozen=True)
class LocalCheckpoint:
    model: str
    revision: str
    cache: Path
    device: str

    @property
    def snapshot(self) -> Path:
        return self.cache / ("models--" + self.model.replace("/", "--")) / "snapshots" / self.revision

    @property
    def preparation(self) -> str:
        return " ".join((
            "uv run --locked --extra qwen hf download", shlex.quote(self.model),
            "--revision", self.revision, "--cache-dir", shlex.quote(str(self.cache)),
        ))

    def key(self, dtype: str) -> tuple[str, str, str, str]:
        return self.model, self.revision, self.device, dtype


def local_checkpoint(model: str, revision: str, cache_dir: str | Path, device: str) -> LocalCheckpoint:
    model = _identifier(model, r"[\w-]+/[\w.-]+", "model must be a Hugging Face owner/model identifier")
    revision = _identifier(
        revision, r"[0-9a-f]{40}", "revision must be a pinned 40-character commit hash",
    )
    device = _identifier(device, r"cuda:(0|[1-9][0-9]*)", "device must explicitly select cuda:N")
    return LocalCheckpoint(model, revision, _cache(cache_dir), device)


def _cache(value: object) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("cache_dir must name a prepared local Hugging Face cache")
    return Path(value)
