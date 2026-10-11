"""Exercise the local Transformers chat without optional model downloads."""

import json
from collections.abc import Mapping
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from core.chats.chat_turn_contract import ChatTurn
from core.chats.envelopes.qwen_chat import QwenChat
from core.chats.envelopes.qwen_load_buffers import LoadBuffers
from core.chats.envelopes.qwen_runtime import LocalRuntime
from core.chats.qwen_stopping_criterion import StoppingCriterion
from core.primitives.diagnostics.diagnostic_redactor.redaction import Redaction
from core.primitives.messages.message.text_message import TextMessage


class TestQwenChat(TestCase):
    """Independent dialogues share a history-free local model."""

    def runtime(self) -> SimpleNamespace:
        requests: list[tuple[str, ...]] = []
        scores: list[tuple[str, str, str]] = []

        def generate(turns: tuple[ChatTurn, ...], *args: object, **kwargs: object) -> str:
            requests.append(tuple(turn.text() for turn in turns))
            return f"reply-{len(turns)}"

        def score(prompt: str, expected: str, system: str, **kwargs: object) -> float:
            scores.append((prompt, expected, system))
            return 0.75

        runtime = SimpleNamespace(
            requests=requests, scores=scores, generate=generate,
            score_continuation=score, actual_dtype=lambda: "float16",
        )
        replacement = patch("core.chats.envelopes.qwen_executor.load_runtime", return_value=runtime)
        replacement.start()
        self.addCleanup(replacement.stop)
        return runtime

    def chat(self, generation: Mapping[str, object] | None = None) -> QwenChat:
        return QwenChat(
            model=("example/llm-chat-test", "a" * 40), cache_dir="/tmp/llm-chat-test",
            device="cuda:0", initial=TextMessage("policy"), system="policy",
            redactor=Redaction(()),
            generation=generation if generation is not None else {"max_new_tokens": 8},
        )

    def fixture(self) -> tuple[SimpleNamespace, QwenChat]:
        return self.runtime(), self.chat()

    def rollback(self) -> tuple[SimpleNamespace, QwenChat]:
        runtime, chat = self.fixture()
        checkpoint = chat.checkpoint()
        chat.send(TextMessage("discarded"))
        chat.restore(checkpoint)
        chat.send(TextMessage("alternate"))
        return runtime, chat

    def inference(self) -> tuple[LocalRuntime, MagicMock, MagicMock]:
        def criteria(callbacks: list[StoppingCriterion]) -> list[StoppingCriterion]:
            return callbacks

        torch = MagicMock()
        for mode in (torch.inference_mode, torch.cuda.device):
            mode.return_value = nullcontext()
        tokenizer = MagicMock()
        tokens = MagicMock()
        tokens.shape = (1, 2)
        inputs = MagicMock()
        inputs.keys.return_value = ("input_ids",)
        inputs.__getitem__.return_value = tokens
        inputs.to.return_value = inputs
        tokenizer.apply_chat_template.return_value = inputs
        tokenizer.model_max_length = 64
        tokenizer.pad_token_id = 0
        tokenizer.decode.return_value = "answer"
        model = MagicMock()
        model.config.max_position_embeddings = 64
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        model.generate.return_value = MagicMock()
        transformers = MagicMock()
        transformers.generation_config.return_value = MagicMock()
        transformers.stopping_criteria.side_effect = criteria
        transformers.load_tokenizer.return_value = tokenizer
        transformers.load_model.return_value = model
        buffers = LoadBuffers()
        buffers.load_tokenizer(transformers, "snapshot")
        buffers.load_model(transformers, "snapshot", "float16")
        buffers.retain()
        return LocalRuntime((torch, transformers), buffers), model, tokenizer

    def test_shared_weights_preserve_separate_histories(self) -> None:
        runtime = self.runtime()
        target, attacker = self.chat(), self.chat()
        for chat, prompt in ((target, "first"), (attacker, "other"), (target, "second")):
            chat.send(TextMessage(prompt))
        assert runtime.requests == [
            ("policy", "first"), ("policy", "other"),
            ("policy", "first", "reply-2", "second"),
        ], "Shared Qwen weights mixed or dropped the conversations"

    def test_scoring_does_not_use_or_change_conversation(self) -> None:
        runtime, chat = self.fixture()
        chat.send(TextMessage("first"))
        before = chat.history()
        loss = chat.score_continuation("candidate", "completion")
        assert (loss, runtime.scores, runtime.requests, chat.history()) == (
            0.75, [("candidate", "completion", "policy")],
            [("policy", "first")], before,
        ), "Qwen scoring used generation history or changed the conversation"

    def test_rollback_changes_next_generation_only(self) -> None:
        runtime, chat = self.rollback()
        assert (runtime.requests[-1], chat.history()[-2].text()) == (
            ("policy", "alternate"), "alternate",
        ), "Qwen rollback retained a discarded generation"

    def test_generation_and_scoring_use_distinct_model_paths(self) -> None:
        runtime, model, tokenizer = self.inference()
        labels = MagicMock()
        labels.shape = (1, 1)
        labels.to.return_value = labels
        tokenizer.return_value = {"input_ids": labels}
        entropy = MagicMock()
        entropy.mean.return_value.item.return_value = 0.75

        def cross_entropy(*args: object, **kwargs: object) -> MagicMock:
            return entropy

        with patch("core.chats.envelopes.qwen_executor.load_runtime", return_value=runtime), patch(
            "core.chats.envelopes.qwen_scoring_utils.import_module",
            return_value=SimpleNamespace(cross_entropy=cross_entropy),
        ):
            chat = self.chat()
            chat.send(TextMessage("first"))
            chat.send(TextMessage("second"))
            loss = chat.score_continuation("candidate", "completion")
        assert (
            loss, model.generate.call_count, model.call_count, len(chat.history()),
            [call.args[0] for call in tokenizer.apply_chat_template.call_args_list],
        ) == (
            0.75, 2, 1, 5,
            [
                [{"role": "system", "content": "policy"}, {"role": "user", "content": "first"}],
                [{"role": "system", "content": "policy"}, {"role": "user", "content": "first"},
                 {"role": "assistant", "content": "answer"}, {"role": "user", "content": "second"}],
                [{"role": "system", "content": "policy"}, {"role": "user", "content": "candidate"},
                 {"role": "assistant", "content": "completion"}],
            ],
        ), "Local Qwen confused scoring with generation or lost its dialogue"

    def test_generation_options_are_detached_at_construction(self) -> None:
        parameters: dict[str, object] = {"max_new_tokens": 8}
        chat = self.chat(parameters)
        parameters["max_new_tokens"] = 32
        assert '"max_new_tokens": 8' in json.dumps(chat.settings()), (
            "Mutation of generation options changed an existing Qwen conversation"
        )

    def test_invalid_generation_is_rejected_before_model_loading(self) -> None:
        with self.assertRaises(ValueError):
            self.chat({"max_new_tokens": 0})
