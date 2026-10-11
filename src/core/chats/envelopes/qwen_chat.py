"""Offline Transformers conversations; optional libraries are imported on send only."""

from importlib import import_module
from pathlib import Path
from threading import Event
from typing import override

from core.primitives.diagnostics import DiagnosticRedactor
from core.primitives.messages import Message

from ..chat_turn_contract import ChatTurn
from .conversation import LLMChat
from ..qwen_context_error import QwenContextError
from ..qwen_error import QwenError
from ..qwen_profile_contract import QwenProfilePort
from ..qwen_versions_utils import versions
from .qwen_executor import Executor
from .qwen_executor_utils import shared_executor
from .qwen_profile import QwenProfile

__all__ = ["QwenChat", "QwenError", "QwenContextError"]


class QwenChat(LLMChat):
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
        **options (object): Dtype, system and generation settings.
    """

    def __init__(
        self, *, model: tuple[str, str], cache_dir: str | Path, device: str,
        initial: Message | None, redactor: DiagnosticRedactor, **options: object,
    ) -> None:
        self.profile: QwenProfilePort = QwenProfile(model, cache_dir, device, options)
        super().__init__(
            lambda turns: "", initial=initial, model=model[0],
            system=self.profile.system,
            redactor=redactor,
        )
        self.executors: list[Executor] = []

    def executor(self) -> Executor:
        if not self.executors:
            checkpoint = self.profile.checkpoint()
            self.executors.append(shared_executor(
                checkpoint, self.profile.dtype(), checkpoint.key(self.profile.dtype())[2],
            ))
        return self.executors[0]

    @override
    def settings(self) -> dict[str, object] | list[object]:
        executor = self.executor()
        return self.redactor.structured({
            **super().settings_data(), "backend": "qwen",
            **self.profile.details(),
            "actual_dtype": executor.actual_dtype(),
            "loaded": executor.is_loaded(), "local_files_only": True,
            "attn_implementation": "sdpa",
            "versions": versions(),
        })

    @override
    def __call__(self, turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
        return self.executor().generate(
            turns, cancellation, self.profile.generation(),
            checkpoint=self.profile.checkpoint(), importer=import_module,
        )

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
        return self.executor().score_continuation(
            prompt, expected, self.profile.system,
            checkpoint=self.profile.checkpoint(), importer=import_module,
        )
