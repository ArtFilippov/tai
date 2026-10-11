"""Diagnostic policies protect detached evidence while preserving JSON structure."""

import json
from collections.abc import Sequence
from tempfile import TemporaryDirectory
from unittest import TestCase

from app.chat_factory import chat_factory
from core.chats.envelopes.qwen_chat import QwenChat
from core.primitives.diagnostics import DiagnosticRedactor
from core.primitives.diagnostics.diagnostic_redactor.redaction import Redaction
from core.primitives.messages.message.text_message import TextMessage
from core.runs.journal import RunJournal


class TestDiagnosticRedactor(TestCase):
    """Verify redaction through the diagnostic contract."""

    def policy(self, credentials: Sequence[str] = ("café", "Bearer café")) -> DiagnosticRedactor:
        return Redaction(credentials)

    def test_combined_credentials(self) -> None:
        policy = self.policy().combine(Redaction(("other",)))
        assert policy.text("Bearer café and other; keep this") == (
            "[REDACTED] and [REDACTED]; keep this"
        ), "Combined diagnostics exposed credentials or lost ordinary text"

    def test_masked_fields_preserve_other_values(self) -> None:
        result = self.policy().structured(
            {"api_key": "café", "temperature": 0.5}, mask_fields=True,
        )
        assert result == {"api_key": "[REDACTED]", "temperature": 0.5}, (
            "Masked settings exposed a credential or lost ordinary values"
        )

    def test_redacted_keys_keep_colliding_fields(self) -> None:
        result = self.policy().structured({"café": "first", "[REDACTED]": "second"})
        assert result == [["[REDACTED]", "first"], ["[REDACTED]", "second"]], (
            "Redaction discarded a colliding diagnostic field"
        )

    def test_credentials_are_detached_from_mutable_input(self) -> None:
        credentials = ["key"]
        policy = self.policy(credentials)
        credentials[0] = "other"
        assert policy.text("key and other") == "[REDACTED] and other", "Mutation leaked secrets"

    def test_chat_factory_redacts_configuration(self) -> None:
        chat = chat_factory({
            "backend": "openai", "base_url": "https://example.test/v1",
            "model": "model", "api_key": "private",
            "generation": {"temperature": 0.5, "note": "private"},
        }, "Target", TextMessage, redaction=self.policy)()
        settings = json.dumps(chat.settings())
        assert all((
            self.policy(("private",)).text("private") in settings,
            "private" not in settings, '"temperature": 0.5' in settings,
        )), "Assembled chat exposed a credential or lost settings"

    def test_journal_read_keeps_redaction(self) -> None:
        journal = RunJournal(redactor=self.policy())
        journal.bind("run")
        journal.record("exchange", message="café", note="ordinary")
        expected = {"message": "[REDACTED]", "note": "ordinary"}
        assert journal.events()[-1].data == expected, "Journal lost redacted evidence"

    def test_qwen_settings_do_not_load_model_or_expose_text(self) -> None:
        with TemporaryDirectory() as directory:
            chat = QwenChat(
                model=("team/model", "a" * 40), cache_dir=directory, device="cuda:0",
                initial=TextMessage("café"), redactor=self.policy(),
            )
            settings = json.dumps(chat.settings(), ensure_ascii=False)
        assert all((
            "café" not in settings, '"device": "cuda:0"' in settings,
            '"loaded": false' in settings,
        )), "Qwen settings exposed text, lost the device, or loaded the model"
