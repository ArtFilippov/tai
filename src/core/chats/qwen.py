"""Offline Transformers conversations; optional libraries are imported on send only."""

from dataclasses import replace
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Event
from typing import Unpack, override

from core.primitives.diagnostics import DiagnosticRedactor
from core.primitives.messages import Message

from .chat_turn_contract import ChatTurn
from .envelopes.conversation import LLMChat
from .qwen_config import (
    QwenContextError, QwenError,
    QwenOptions, generation_config, local_checkpoint, validate_dtype, validate_system,
)
from .qwen_executor import shared_executor
from .scoring import ContinuationScorer

__all__ = ["QwenChat", "QwenError", "QwenContextError"]


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("torch", "transformers", "garak"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


class QwenChat(LLMChat, ContinuationScorer):
    """One local conversation on an explicit CUDA device, loaded lazily on send.

    ``cache_dir`` is a prepared Hugging Face Hub cache (``~`` is expanded).
    ``revision`` must be a full commit hash; network downloads are never attempted.
    Matching model/revision/device/dtype share weights while any chat retains them.

    Args:
        model (tuple[str, str]): Hugging Face model identity and full pinned revision.
        cache_dir (str | Path): Prepared local Hub cache.
        device (str): Explicit CUDA device.
        initial (Message | None): Prepared system turn, when present.
        redactor (DiagnosticRedactor): Policy for detached diagnostics.
        **options (Unpack[QwenOptions]): Dtype, system and generation.

    Raises:
        TypeError: An unknown constructor option was supplied.
    """

    def __init__(
        self, *, model: tuple[str, str], cache_dir: str | Path, device: str,
        initial: Message | None, redactor: DiagnosticRedactor, **options: Unpack[QwenOptions],
    ) -> None:
        if unknown := options.keys() - {"dtype", "system", "generation"}:
            raise TypeError(f"Unexpected QwenChat options: {', '.join(sorted(unknown))}")
        checkpoint = local_checkpoint(model[0], model[1], cache_dir, device)
        self._checkpoint = replace(checkpoint, cache=checkpoint.cache.expanduser().resolve())
        dtype = validate_dtype(options.get("dtype", "float16"))
        self._generation = generation_config(options.get("generation"))
        self._system = validate_system(options.get("system", ""))
        super().__init__(
            lambda turns: "", initial=initial, model=model[0], system=self._system,
            redactor=redactor,
        )
        self._executor = shared_executor(self._checkpoint, dtype)

    @override
    def settings(self) -> dict[str, object] | list[object]:
        return self.redactor.structured({
            **super().settings_data(), "backend": "qwen",
            "revision": self._checkpoint.revision, "cache_dir": str(self._checkpoint.cache),
            "device": self._executor.device, "dtype": self._executor.dtype,
            "actual_dtype": self._executor.actual_dtype(),
            "loaded": self._executor.is_loaded(), "local_files_only": True,
            "attn_implementation": "sdpa", "generation": self._generation.copy(),
            "versions": _versions(),
        })

    @override
    def __call__(self, turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
        return self._executor.generate(
            turns, cancellation, self._generation,
            checkpoint=self._checkpoint, importer=import_module,
        )

    @override
    def score_continuation(self, prompt: str, expected: str) -> float:
        """Return local logits loss, without reading or changing active history.

        Args:
            prompt (str): Full user prompt.
            expected (str): Expected assistant continuation.

        Returns:
            float: Mean token cross-entropy.

        Raises:
            ValueError: The prompt or continuation is empty.
        """
        if not prompt or not expected:
            raise ValueError("AutoDAN scoring requires a prompt and expected continuation")
        system = self._system
        return self._executor.score_continuation(
            prompt, expected, system, checkpoint=self._checkpoint, importer=import_module,
        )
