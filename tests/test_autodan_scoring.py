"""Local logits scoring preserves garak token alignment and does not generate."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from core.chats import qwen_scoring
from core.chats.qwen_config import QwenContextError


class Tensor:
    def __init__(self, shape: tuple[int, ...], operations: list[tuple[object, ...]]) -> None:
        super().__init__()
        self.shape, self.operations = shape, operations

    def to(self, device: str) -> Tensor:
        self.operations.append(("move", device))
        return self

    def __getitem__(self, key: object) -> Tensor:
        self.operations.append(("slice", key))
        return self

    def contiguous(self) -> Tensor:
        return self

    def size(self, dimension: int) -> int:
        assert dimension == -1
        return self.shape[-1]

    def view(self, *dimensions: int) -> Tensor:
        self.operations.append(("view", dimensions))
        return self


def libraries(operations: list[tuple[object, ...]], *, limit: int = 30) -> tuple[Any, Any, Any]:
    def encode(value: list[str], **options: object) -> dict[str, Tensor]:
        operations.append(("expected", value, options))
        return {"input_ids": Tensor((1, 2), operations)}

    def template(messages: list[dict[str, str]], **options: object) -> Batch:
        operations.append(("template", messages, options))
        tokens = Tensor((1, 7), operations)
        return Batch(tokens, operations)

    tokenizer = SimpleNamespace(__call__=encode, apply_chat_template=template, model_max_length=limit)
    # __call__ is looked up on the tokenizer object by the optional adapter.
    def forward(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(logits=Tensor((1, 7, 9), operations))

    def device_context(device: str) -> Any:
        return nullcontext()

    def synchronize(device: str) -> None:
        operations.append(("sync", device))

    model = SimpleNamespace(__call__=forward,
                            config=SimpleNamespace(max_position_embeddings=30))
    cuda = SimpleNamespace(device=device_context, synchronize=synchronize)
    torch = SimpleNamespace(inference_mode=nullcontext, cuda=cuda)
    return torch, tokenizer, model


class Batch:
    def __init__(self, tokens: Tensor, operations: list[tuple[object, ...]]) -> None:
        super().__init__()
        self.tokens, self.operations = tokens, operations

    def to(self, device: str) -> Batch:
        self.operations.append(("batch_move", device))
        return self

    def __getitem__(self, key: str) -> Tensor:
        return self.tokens


def test_full_template_and_expected_token_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    operations: list[tuple[object, ...]] = []

    def cross_entropy(logits: Tensor, labels: Tensor, **options: object) -> SimpleNamespace:
        operations.append(("loss", logits, labels, options))
        return SimpleNamespace(mean=lambda: SimpleNamespace(item=lambda: 0.25))

    def module(name: str) -> SimpleNamespace:
        assert name == "torch.nn.functional"
        return SimpleNamespace(cross_entropy=cross_entropy)

    monkeypatch.setattr(qwen_scoring, "import_module", module)
    score = qwen_scoring.score_continuation(libraries(operations), "prefix Base", "Sure", "Target system", device="cuda:0")
    assert score == 0.25
    assert ("expected", ["Sure"], {"add_special_tokens": False, "return_tensors": "pt"}) in operations
    template = next(item for item in operations if item[0] == "template")
    assert template[1] == [
        {"role": "system", "content": "Target system"},
        {"role": "user", "content": "prefix Base"},
        {"role": "assistant", "content": "Sure"},
    ]
    assert ("slice", (Ellipsis, slice(4, -1), slice(None))) in operations
    assert ("sync", "cuda:0") == operations[-1]


def test_context_overflow_does_not_run_forward_and_still_synchronizes() -> None:
    operations: list[tuple[object, ...]] = []
    with pytest.raises(QwenContextError, match="overflow"):
        qwen_scoring.score_continuation(libraries(operations, limit=5), "prefix Base", "Sure", "", device="cuda:0")
    assert ("sync", "cuda:0") == operations[-1]
    assert not any(item[0] == "slice" for item in operations)
