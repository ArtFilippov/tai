"""Non-context failures while preparing or running local model weights."""

from typing import override

from .qwen_error import QwenError


class QwenFailure(QwenError):
    """A missing model file, unavailable GPU, dtype, or dependency failure."""

    @override
    def category(self) -> str:
        return "runtime"
