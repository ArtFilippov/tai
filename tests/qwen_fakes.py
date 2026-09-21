"""Controlled, typed implementations of the optional local-inference contracts."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Self, TypedDict
from uuid import uuid4

import pytest

from core.chats import qwen
from core.chats.qwen_contracts import GenerationConfig, StoppingCriterion, Tensor


@dataclass
class Tokens:
    tokens: list[int]

    @property
    def shape(self) -> tuple[int, int]:
        return 1, len(self.tokens)

    def __getitem__(self, index: tuple[int, slice]) -> Tokens:
        row, span = index
        assert not row
        return Tokens(self.tokens[span])


@dataclass
class Inputs:
    input_ids: Tokens
    attention_mask: Tokens
    device: str | None = None

    def keys(self) -> list[str]:
        return ["input_ids", "attention_mask"]

    def __getitem__(self, key: str) -> Tokens:
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}[key]

    def to(self, device: str) -> Self:
        self.device = device
        return self


class GenerateArguments(TypedDict):
    generation_config: GenerationConfig
    stopping_criteria: Sequence[StoppingCriterion]


@dataclass
class Records:
    loads: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    requests: list[list[dict[str, str]]] = field(default_factory=list[list[dict[str, str]]])
    generations: list[tuple[str, GenerateArguments]] = field(
        default_factory=list[tuple[str, GenerateArguments]],
    )
    decoded: list[list[int]] = field(default_factory=list[list[int]])
    imports: list[str] = field(default_factory=list[str])


@dataclass
class Actions:
    load: Callable[[], None] = lambda: None
    generate: Callable[[Sequence[StoppingCriterion]], None] = lambda criteria: None
    sync: Callable[[], None] = lambda: None
    inference: bool = False


@dataclass
class FakeModel:
    library: LocalLibraries
    dtype: object
    device: str = ""
    evaluating: bool = False
    generation_config: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(bos_token_id=99, eos_token_id=[99]),
    )

    @property
    def config(self) -> SimpleNamespace:
        return SimpleNamespace(max_position_embeddings=self.library.limit)

    def to(self, device: str) -> Self:
        self.device = device
        return self

    def eval(self) -> Self:
        self.evaluating = True
        return self

    def generate(
        self, *, generation_config: GenerationConfig,
        stopping_criteria: Sequence[StoppingCriterion], **inputs: Tensor,
    ) -> Tokens:
        assert self.evaluating and self.library.actions.inference
        assert "past_key_values" not in inputs
        self.library.records.generations.append((self.device, {
            "generation_config": generation_config, "stopping_criteria": stopping_criteria,
        }))
        self.library.actions.generate(stopping_criteria)
        tokens = inputs["input_ids"]
        assert isinstance(tokens, Tokens)
        return Tokens(tokens.tokens + [42, 99])


@dataclass
class LocalLibraries:
    records: Records = field(default_factory=Records)
    actions: Actions = field(default_factory=Actions)
    cuda_available: bool = True
    input_length: int = 7
    limit: int = 32768
    responses: deque[str] = field(default_factory=deque[str])

    def imports_for(self, name: str) -> object:
        self.records.imports.append(name)
        return {"torch": self.torch(), "transformers": self.transformers()}[name]

    def torch(self) -> SimpleNamespace:
        return SimpleNamespace(
            float16="torch.float16", bfloat16="torch.bfloat16", float32="torch.float32",
            inference_mode=self.inference_mode,
            cuda=SimpleNamespace(
                is_available=lambda: self.cuda_available, device_count=lambda: 2,
                is_bf16_supported=lambda: False, device=self.device,
                synchronize=self.synchronize,
            ),
        )

    def transformers(self) -> SimpleNamespace:
        return SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=self.tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=self.model),
            GenerationConfig=SimpleNamespace, StoppingCriteria=object, StoppingCriteriaList=list,
        )

    def synchronize(self, _device: str) -> None:
        self.actions.sync()

    @contextmanager
    def inference_mode(self) -> Generator[None]:
        assert not self.actions.inference
        self.actions.inference = True
        try:
            yield
        finally:
            self.actions.inference = False

    @contextmanager
    def device(self, name: str) -> Generator[None]:
        assert name in {"cuda:0", "cuda:1"}
        yield

    def tokenizer(self, path: str, **kwargs: object) -> SimpleNamespace:
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        self.records.loads.append(("tokenizer", path))
        return SimpleNamespace(
            apply_chat_template=self.template, decode=self.decode,
            model_max_length=self.limit, pad_token_id=99,
        )

    def model(self, path: str, **kwargs: object) -> FakeModel:
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        assert kwargs["use_safetensors"] is True
        self.records.loads.append(("model", path))
        self.actions.load()
        return FakeModel(self, kwargs["torch_dtype"])

    def template(self, messages: list[dict[str, str]], **kwargs: object) -> Inputs:
        assert kwargs == {
            "tokenize": True, "add_generation_prompt": True,
            "return_tensors": "pt", "return_dict": True, "truncation": False,
        }
        self.records.requests.append(messages)
        return Inputs(Tokens([1] * self.input_length), Tokens([1] * self.input_length))

    def decode(self, tokens: Tensor, **kwargs: object) -> str:
        assert isinstance(tokens, Tokens)
        assert tokens.tokens == [42, 99], "Input prompt must not be decoded as assistant text"
        assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        self.records.decoded.append(tokens.tokens)
        return self.responses.popleft() if self.responses else "yes"


class LocalOptions(TypedDict):
    model: str
    revision: str
    cache_dir: Path
    device: str


type LocalFixture = tuple[LocalLibraries, LocalOptions, Path]


@pytest.fixture
def local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalFixture:
    libraries = LocalLibraries()
    monkeypatch.setattr(qwen, "import_module", libraries.imports_for)
    revision = uuid4().hex + "12345678"
    cache = tmp_path / "cache"
    snapshot = cache / "models--Qwen--test" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (snapshot / name).touch()
    kwargs: LocalOptions = {
        "model": "Qwen/test", "revision": revision, "cache_dir": cache, "device": "cuda:0",
    }
    return libraries, kwargs, snapshot
