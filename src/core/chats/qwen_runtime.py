"""Lazy optional-library loading and a history-free local inference session."""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import cast, override

from .conversation import ChatCancelled, ChatTurn
from .qwen_config import Generation, LocalCheckpoint, QwenContextError, QwenError
from .qwen_contracts import (
    InputBatch, LocalRuntimeProtocol, Model, Tensor, Tokenizer, Torch, Transformers,
)

_BFLOAT16 = "bfloat16"


def check_cancelled(cancellation: Event) -> None:
    if cancellation.is_set():
        raise ChatCancelled()


@dataclass(frozen=True)
class _CancelGeneration:
    cancellation: Event

    def __call__(self, *args: object, **kwargs: object) -> bool:
        return self.cancellation.is_set()


@dataclass
class _GenerationBuffers:
    inputs: InputBatch | None = None
    output: Tensor | None = None

    def clear(self) -> None:
        self.inputs = None
        self.output = None

    def move(self, device: str) -> None:
        assert self.inputs is not None
        self.inputs = self.inputs.to(device)


@dataclass
class _LoadBuffers:
    tokenizer: Tokenizer | None = None
    model: Model | None = None

    def clear(self) -> None:
        self.tokenizer = None
        self.model = None

    def prepare(self, torch: Torch, device: str, dtype: object, cancellation: Event) -> None:
        check_cancelled(cancellation)
        assert self.model is not None
        self.model = self.model.to(device)
        self.model = self.model.eval()
        cuda = torch.cuda
        cuda.synchronize(device)
        check_cancelled(cancellation)
        if self.model.dtype != dtype:
            raise QwenError(f"Model loaded with unexpected dtype: {self.model.dtype}")


@dataclass(frozen=True)
class LocalRuntime(LocalRuntimeProtocol):
    torch: Torch
    transformers: Transformers
    tokenizer: Tokenizer
    model: Model

    @override
    def actual_dtype(self) -> str:
        return str(self.model.dtype).removeprefix("torch.")

    def _inputs(
        self, buffers: _GenerationBuffers, turns: tuple[ChatTurn, ...], generation: Generation,
    ) -> None:
        buffers.inputs = self.tokenizer.apply_chat_template(
            [{"role": turn.role, "content": turn.text()} for turn in turns],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, truncation=False,
        )
        length = buffers.inputs["input_ids"].shape[-1]
        limit = min(self.model.config.max_position_embeddings, self.tokenizer.model_max_length)
        if length + generation["max_new_tokens"] > limit:
            raise QwenContextError(
                f"Qwen context overflow: input={length}, "
                + f"max_new_tokens={generation['max_new_tokens']}, "
                + f"limit={limit}; no messages were truncated"
            )

    def _decode(
        self, buffers: _GenerationBuffers, cancellation: Event, generation: Generation,
    ) -> str:
        assert buffers.inputs is not None
        factory = self.transformers.generation_config
        defaults = self.model.generation_config
        config = factory(
            **generation, bos_token_id=defaults.bos_token_id,
            eos_token_id=defaults.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
        )
        criteria = self.transformers.stopping_criteria
        check_cancelled(cancellation)
        buffers.output = self.model.generate(
            **buffers.inputs, generation_config=config,
            stopping_criteria=criteria([_CancelGeneration(cancellation)]),
        )
        check_cancelled(cancellation)
        text = self.tokenizer.decode(
            buffers.output[0, buffers.inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        check_cancelled(cancellation)
        return text

    @override
    def generate(
        self, turns: tuple[ChatTurn, ...], cancellation: Event,
        generation: Generation, *, device: str,
    ) -> str:
        cuda = self.torch.cuda
        with self.torch.inference_mode(), cuda.device(device):
            try:
                return self._execute(turns, cancellation, generation, device=device)
            finally:
                # STOPPED follows all queued CUDA work and call-local tensor cleanup.
                cuda.synchronize(device)

    def _execute(
        self, turns: tuple[ChatTurn, ...], cancellation: Event,
        generation: Generation, *, device: str,
    ) -> str:
        buffers = _GenerationBuffers()
        try:
            return self._run(buffers, turns, cancellation, generation, device=device)
        finally:
            # Tracebacks may retain helper frames. Keep tensors in one holder so
            # cancellation/failure clears them before the outer CUDA synchronize.
            buffers.clear()

    def _run(
        self, buffers: _GenerationBuffers, turns: tuple[ChatTurn, ...],
        cancellation: Event, generation: Generation, *, device: str,
    ) -> str:
        self._inputs(buffers, turns, generation)
        check_cancelled(cancellation)
        buffers.move(device)
        return self._decode(buffers, cancellation, generation)


def _libraries(importer: Callable[[str], object]) -> tuple[Torch, Transformers]:
    # These casts describe the real optional-library surface. No optional package
    # is imported for type checking, settings observation, or chat construction.
    return cast(Torch, importer("torch")), Transformers.from_module(importer("transformers"))


def _device(torch: Torch, device: str, dtype: str) -> object:
    cuda = torch.cuda
    if not cuda.is_available() or int(device.split(":")[1]) >= cuda.device_count():
        raise QwenError(f"CUDA device {device} is unavailable; no CPU fallback")
    with cuda.device(device):
        if dtype == _BFLOAT16 and not cuda.is_bf16_supported():
            raise QwenError(f"{device} does not support bfloat16")
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]


def _load(
    libraries: tuple[Torch, Transformers], checkpoint: LocalCheckpoint,
    dtype: str, cancellation: Event,
) -> LocalRuntime:
    buffers = _LoadBuffers()
    try:
        return _load_into(buffers, libraries, checkpoint, dtype, cancellation=cancellation)
    finally:
        buffers.clear()


def _load_into(
    buffers: _LoadBuffers, libraries: tuple[Torch, Transformers],
    checkpoint: LocalCheckpoint, dtype: str, *, cancellation: Event,
) -> LocalRuntime:
    torch, transformers = libraries
    target_dtype = _device(torch, checkpoint.device, dtype)
    _check_files(checkpoint)
    check_cancelled(cancellation)
    buffers.tokenizer = transformers.load_tokenizer(str(checkpoint.snapshot))
    check_cancelled(cancellation)
    buffers.model = transformers.load_model(str(checkpoint.snapshot), target_dtype)
    buffers.prepare(torch, checkpoint.device, target_dtype, cancellation)
    return LocalRuntime(torch, transformers, buffers.tokenizer, buffers.model)


def _check_files(checkpoint: LocalCheckpoint) -> None:
    required = ("config.json", "tokenizer_config.json", "tokenizer.json")
    weights = ("model.safetensors", "model.safetensors.index.json")
    if not all((checkpoint.snapshot / name).is_file() for name in required) or not any(
        (checkpoint.snapshot / name).is_file() for name in weights
    ):
        raise QwenError(
            f"Pinned local Qwen files are missing; prepare with: {checkpoint.preparation}"
        )


def load_runtime(
    checkpoint: LocalCheckpoint, dtype: str, cancellation: Event,
    importer: Callable[[str], object],
) -> LocalRuntime:
    check_cancelled(cancellation)
    try:
        libraries = _libraries(importer)
    except ImportError as exc:
        raise QwenError(
            "Qwen dependencies unavailable; run: uv sync --locked --extra qwen"
        ) from exc
    check_cancelled(cancellation)
    try:
        return _load(libraries, checkpoint, dtype, cancellation)
    except OSError as exc:
        raise QwenError(
            f"Local Qwen load failed: {exc}; prepare with: {checkpoint.preparation}"
        ) from exc
