"""Lazy optional-library loading and a history-free local inference session."""

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Event
from typing import override

from ..chat_turn_contract import ChatTurn
from ..qwen_load_buffers_contract import LoadBuffersPort
from ..qwen_runtime_contract import LocalRuntimeProtocol
from ..qwen_torch import Torch
from ..qwen_transformers_contract import TransformersPort
from .qwen_scoring_utils import score_continuation

from .qwen_cancel_generation import CancelGeneration
from .qwen_gate_utils import check_cancelled
from .qwen_generation_buffers import GenerationBuffers

@dataclass(frozen=True)
class LocalRuntime(LocalRuntimeProtocol):
    libraries: tuple[Torch, TransformersPort]
    resources: LoadBuffersPort

    @override
    def score_continuation(self, prompt: str, expected: str, system: str, *, device: str) -> float:
        """Score the assistant continuation under the shared model weights.

        Args:
            prompt (str): Full user prompt.
            expected (str): Assistant continuation.
            system (str): Target system instruction.
            device (str): Pinned CUDA device.

        Returns:
            float: Mean token loss.
        """
        return score_continuation(
            (self.libraries[0], self.resources.tokenizer(), self.resources.model()),
            prompt, expected, system, device=device,
        )

    @override
    def actual_dtype(self) -> str:
        return str(self.resources.model().dtype).removeprefix("torch.")

    def decode(
        self, buffers: GenerationBuffers, cancellation: Event,
        generation: Mapping[str, int | float | bool],
    ) -> str:
        factory = self.libraries[1].generation_config
        defaults = self.resources.model().generation_config
        config = factory(
            **generation, bos_token_id=defaults.bos_token_id,
            eos_token_id=defaults.eos_token_id,
            pad_token_id=self.resources.tokenizer().pad_token_id,
        )
        criteria = self.libraries[1].stopping_criteria
        check_cancelled(cancellation)
        buffers.accept(self.resources.model().generate(
            **buffers.batch(), generation_config=config,
            stopping_criteria=criteria([CancelGeneration(cancellation)]),
        ))
        check_cancelled(cancellation)
        text = buffers.decode(self.resources.tokenizer())
        check_cancelled(cancellation)
        return text

    @override
    def generate(
        self, turns: tuple[ChatTurn, ...], cancellation: Event,
        generation: Mapping[str, int | float | bool], *, device: str,
    ) -> str:
        cuda = self.libraries[0].cuda
        with self.libraries[0].inference_mode(), cuda.device(device):
            try:
                return self.execute(turns, cancellation, generation, device=device)
            finally:
                # STOPPED follows all queued CUDA work and call-local tensor cleanup.
                cuda.synchronize(device)

    def execute(
        self, turns: tuple[ChatTurn, ...], cancellation: Event,
        generation: Mapping[str, int | float | bool], *, device: str,
    ) -> str:
        buffers = GenerationBuffers()
        try:
            return self.run(buffers, turns, cancellation, generation, device=device)
        finally:
            # Tracebacks may retain helper frames. Keep tensors in one holder so
            # cancellation/failure clears them before the outer CUDA synchronize.
            buffers.clear()

    def run(
        self, buffers: GenerationBuffers, turns: tuple[ChatTurn, ...],
        cancellation: Event, generation: Mapping[str, int | float | bool], *, device: str,
    ) -> str:
        buffers.prepare(
            self.resources.tokenizer(), self.resources.model(),
            [{"role": turn.role, "content": turn.text()} for turn in turns],
            generation["max_new_tokens"],
        )
        check_cancelled(cancellation)
        buffers.move(device)
        return self.decode(buffers, cancellation, generation)
