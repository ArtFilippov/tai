"""Offline Transformers conversations; optional libraries are imported on send only."""

import math
import re
import shlex
from collections.abc import Mapping
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Event, Lock
from typing import Any
from weakref import WeakValueDictionary

from tai._diagnostics import DiagnosticObject
from tai.chats import ChatCancelled, ChatTurn, LLMChat


class QwenError(RuntimeError):
    """Local preparation, device, or model contract failure (no fallback)."""


class QwenContextError(QwenError):
    """The complete input plus requested output exceeds the context window."""


def _check_cancelled(cancellation: Event) -> None:
    if cancellation.is_set():
        raise ChatCancelled()


def _generation(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(parameters or {})
    allowed = {
        "max_new_tokens",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
    }
    if values.keys() - allowed:
        raise ValueError("unsupported Qwen generation parameters; see README schema")
    values = {
        "max_new_tokens": 512,
        "do_sample": False,
        "repetition_penalty": 1.0,
        **values,
    }
    if type(values["max_new_tokens"]) is not int or values["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if type(values["do_sample"]) is not bool:
        raise ValueError("do_sample must be a boolean")
    sampling = {"temperature", "top_p", "top_k"}
    if not values["do_sample"] and sampling & values.keys():
        raise ValueError("temperature/top_p/top_k require do_sample=true")
    if values["do_sample"]:
        values = {"temperature": 1.0, "top_p": 1.0, "top_k": 50, **values}
    for name in ("temperature", "top_p", "repetition_penalty"):
        if name not in values:
            continue
        value = values[name]
        if (
            type(value) not in (float, int)
            or not math.isfinite(value)
            or value <= 0
            or (name == "top_p" and value > 1)
        ):
            raise ValueError(f"{name} must be positive and finite (top_p <= 1)")
    if "top_k" in values and (type(values["top_k"]) is not int or values["top_k"] < 0):
        raise ValueError("top_k must be a nonnegative integer")
    return {
        **values,
        "num_beams": 1,
        "num_return_sequences": 1,
        "use_cache": True,
        "return_dict_in_generate": False,
    }


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("torch", "transformers", "garak"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


class _Executor:
    """Weights and an execution gate only; no history, KV-cache, or stop signal."""

    def __init__(self, device: str, dtype: str) -> None:
        self.device = device
        self.dtype = dtype
        self.gate = Lock()
        # Publish a complete load atomically, never a partially prepared model.
        self.loaded: tuple[Any, Any, Any, Any] | None = None

    def _load(self, snapshot: Path, preparation: str, cancellation: Event) -> None:
        _check_cancelled(cancellation)
        if self.loaded is not None:
            return
        try:
            torch = import_module("torch")
            transformers = import_module("transformers")
        except ImportError as exc:
            raise QwenError(
                "Qwen dependencies unavailable; run: uv sync --locked --extra qwen"
            ) from exc
        _check_cancelled(cancellation)
        if (
            not torch.cuda.is_available()
            or int(self.device.split(":")[1]) >= torch.cuda.device_count()
        ):
            raise QwenError(
                f"CUDA device {self.device} is unavailable; no CPU fallback"
            )
        with torch.cuda.device(self.device):
            if self.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise QwenError(f"{self.device} does not support bfloat16")
        required = ("config.json", "tokenizer_config.json", "tokenizer.json")
        if not all((snapshot / name).is_file() for name in required) or not any(
            (snapshot / name).is_file()
            for name in ("model.safetensors", "model.safetensors.index.json")
        ):
            raise QwenError(
                f"Pinned local Qwen files are missing; prepare with: {preparation}"
            )
        _check_cancelled(cancellation)
        tokenizer = model = None
        try:
            # A resolved snapshot path, not a Hub ID, also prevents library-side
            # discovery of remote adapters or an alternate generation implementation.
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(snapshot),
                local_files_only=True,
                trust_remote_code=False,
            )
            _check_cancelled(cancellation)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                str(snapshot),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                torch_dtype=getattr(torch, self.dtype),
                attn_implementation="sdpa",
            )
            _check_cancelled(cancellation)
            model.to(self.device)
            model.eval()
            torch.cuda.synchronize(self.device)
            _check_cancelled(cancellation)
            if model.dtype != getattr(torch, self.dtype):
                raise QwenError(f"Model loaded with unexpected dtype: {model.dtype}")
            self.loaded = torch, transformers, tokenizer, model
        except OSError as exc:
            raise QwenError(
                f"Local Qwen load failed: {exc}; prepare with: {preparation}"
            ) from exc
        finally:
            tokenizer = model = None

    def generate(
        self,
        turns: tuple[ChatTurn, ...],
        cancellation: Event,
        snapshot: Path,
        preparation: str,
        generation: dict[str, Any],
    ) -> str:
        _check_cancelled(cancellation)
        while not self.gate.acquire(timeout=0.02):
            _check_cancelled(cancellation)
        inputs = output = None
        try:
            _check_cancelled(cancellation)
            self._load(snapshot, preparation, cancellation)
            _check_cancelled(cancellation)
            assert self.loaded is not None
            torch, transformers, tokenizer, model = self.loaded

            class CancelGeneration(transformers.StoppingCriteria):  # type: ignore[name-defined, misc]
                def __call__(self, *args: Any, **kwargs: Any) -> bool:
                    return cancellation.is_set()

            with torch.inference_mode(), torch.cuda.device(self.device):
                try:
                    inputs = tokenizer.apply_chat_template(
                        [
                            {"role": turn.role, "content": turn.message.to_string()}
                            for turn in turns
                        ],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        truncation=False,
                    )
                    length = inputs["input_ids"].shape[-1]
                    limit = min(
                        model.config.max_position_embeddings, tokenizer.model_max_length
                    )
                    if length + generation["max_new_tokens"] > limit:
                        raise QwenContextError(
                            f"Qwen context overflow: input={length}, "
                            f"max_new_tokens={generation['max_new_tokens']}, "
                            f"limit={limit}; "
                            "no messages were truncated"
                        )
                    _check_cancelled(cancellation)
                    inputs = inputs.to(self.device)
                    # Fresh configuration and a fresh per-call KV-cache. Checkpoint
                    # generation defaults cannot silently change role parameters.
                    config = transformers.GenerationConfig(
                        **generation,
                        bos_token_id=model.generation_config.bos_token_id,
                        eos_token_id=model.generation_config.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    _check_cancelled(cancellation)
                    output = model.generate(
                        **inputs,
                        generation_config=config,
                        stopping_criteria=transformers.StoppingCriteriaList(
                            [CancelGeneration()]
                        ),
                    )
                    _check_cancelled(cancellation)
                    text = tokenizer.decode(
                        output[0, length:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    _check_cancelled(cancellation)
                    return str(text)
                finally:
                    inputs = output = None
                    # STOPPED must follow execution cleanup, including queued CUDA
                    # work. Allocator reservations and shared weights may survive.
                    torch.cuda.synchronize(self.device)
        finally:
            self.gate.release()


_executors: WeakValueDictionary[tuple[str, str, str, str], _Executor] = (
    WeakValueDictionary()
)
_executors_lock = Lock()


class QwenChat(LLMChat):
    """One local conversation on an explicit CUDA device, loaded lazily on send.

    ``cache_dir`` is a prepared Hugging Face Hub cache (``~`` is expanded).
    ``revision`` must be a full commit hash; network downloads are never attempted.
    Matching model/revision/device/dtype share weights while any chat retains them.
    """

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        cache_dir: str | Path,
        device: str,
        dtype: str = "float16",
        system: str = "",
        generation: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model, str) or not re.fullmatch(r"[\w-]+/[\w.-]+", model):
            raise ValueError("model must be a Hugging Face owner/model identifier")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be a pinned 40-character commit hash")
        if not isinstance(device, str) or not re.fullmatch(
            r"cuda:(0|[1-9][0-9]*)", device
        ):
            raise ValueError("device must explicitly select cuda:N")
        if dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("dtype must be float16, bfloat16, or float32")
        if not isinstance(cache_dir, (str, Path)) or not str(cache_dir).strip():
            raise ValueError("cache_dir must name a prepared local Hugging Face cache")
        if not isinstance(system, str):
            raise ValueError("system must be a string")
        self._generation = _generation(generation)
        self._revision = revision
        self._cache_dir = Path(cache_dir).expanduser().resolve()
        self._snapshot = (
            self._cache_dir
            / ("models--" + model.replace("/", "--"))
            / "snapshots"
            / revision
        )
        self._preparation = (
            "uv run --locked --extra qwen hf download "
            f"{shlex.quote(model)} --revision {revision} "
            f"--cache-dir {shlex.quote(str(self._cache_dir))}"
        )
        super().__init__(lambda turns: "", model=model, system=system)
        key = model, revision, device, dtype
        with _executors_lock:
            executor = _executors.get(key)
            if executor is None:
                executor = _Executor(device, dtype)
                _executors[key] = executor
            self._executor = executor

    def settings(self) -> DiagnosticObject:
        loaded = self._executor.loaded
        return {
            **super()._settings(),
            "backend": "qwen",
            "revision": self._revision,
            "cache_dir": str(self._cache_dir),
            "device": self._executor.device,
            "dtype": self._executor.dtype,
            "actual_dtype": str(loaded[3].dtype).removeprefix("torch.")
            if loaded
            else None,
            "loaded": loaded is not None,
            "local_files_only": True,
            "attn_implementation": "sdpa",
            "generation": self._generation.copy(),
            "versions": _versions(),
        }

    def _generate(self, turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
        return self._executor.generate(
            turns,
            cancellation,
            self._snapshot,
            self._preparation,
            self._generation,
        )
