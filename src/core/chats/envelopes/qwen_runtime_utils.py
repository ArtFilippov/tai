"""Load one prepared local Transformers runtime without retaining partial weights."""

from collections.abc import Callable
from threading import Event

from ..qwen_checkpoint_contract import LocalCheckpointPort
from ..qwen_failure import QwenFailure
from ..qwen_diagnostic_utils import failed, missing, unavailable, unsupported
from ..qwen_torch import Torch
from ..qwen_transformers import Transformers
from ..qwen_transformers_contract import TransformersPort
from .qwen_gate_utils import check_cancelled
from .qwen_load_buffers import LoadBuffers
from .qwen_runtime import LocalRuntime

BFLOAT16 = "bfloat16"


def libraries(importer: Callable[[str], object]) -> tuple[Torch, TransformersPort]:
    # Optional packages are imported only on the first admitted model request.
    transformers = Transformers(importer)
    return transformers.prepare(), transformers


def device(torch: Torch, name: str, dtype: str) -> object:
    cuda = torch.cuda
    if not cuda.is_available() or int(name.split(":")[1]) >= cuda.device_count():
        raise QwenFailure(unavailable(name))
    with cuda.device(name):
        if dtype == BFLOAT16 and not cuda.is_bf16_supported():
            raise QwenFailure(unsupported(name))
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]


def load(
    packages: tuple[Torch, TransformersPort], checkpoint: LocalCheckpointPort,
    dtype: str, cancellation: Event,
) -> LocalRuntime:
    with LoadBuffers() as buffers:
        return load_into(buffers, packages, checkpoint, dtype, cancellation=cancellation)


def load_into(
    buffers: LoadBuffers, packages: tuple[Torch, TransformersPort],
    checkpoint: LocalCheckpointPort, dtype: str, *, cancellation: Event,
) -> LocalRuntime:
    torch, transformers = packages
    target = device(torch, checkpoint.key(dtype)[2], dtype)
    check_files(checkpoint)
    check_cancelled(cancellation)
    buffers.load_tokenizer(transformers, str(checkpoint.snapshot))
    check_cancelled(cancellation)
    buffers.load_model(transformers, str(checkpoint.snapshot), target)
    buffers.prepare(torch, checkpoint.key(dtype)[2], target, cancellation)
    runtime = LocalRuntime(packages, buffers)
    buffers.retain()
    return runtime


def check_files(checkpoint: LocalCheckpointPort) -> None:
    required = ("config.json", "tokenizer_config.json", "tokenizer.json")
    weights = ("model.safetensors", "model.safetensors.index.json")
    if not all((checkpoint.snapshot / name).is_file() for name in required) or not any(
        (checkpoint.snapshot / name).is_file() for name in weights
    ):
        raise QwenFailure(missing(checkpoint))


def load_runtime(
    checkpoint: LocalCheckpointPort, dtype: str, cancellation: Event,
    importer: Callable[[str], object],
) -> LocalRuntime:
    check_cancelled(cancellation)
    try:
        packages = libraries(importer)
    except ImportError as exc:
        raise QwenFailure(
            "Qwen dependencies unavailable; run: uv sync --locked --extra qwen"
        ) from exc
    check_cancelled(cancellation)
    try:
        return load(packages, checkpoint, dtype, cancellation)
    except OSError as exc:
        raise QwenFailure(failed(checkpoint)) from exc
