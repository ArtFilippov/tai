"""Validated participants shared by readiness and experiment composition."""

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Literal

from attacks.shared.contracts import AttackType
from core.chats import LLMChat
from core.evaluation.configuration import EvaluatorSetup
from core.primitives.diagnostics import DiagnosticRedactor

type ModelRole = Literal["Target", "Attacker", "Evaluator"]


@dataclass(frozen=True)
class ParticipantProfile:
    target: LLMChat
    attacker: LLMChat | None
    evaluator: EvaluatorSetup
    max_calls_per_conv: int | None
    attack_type: AttackType = field(default="fitd", kw_only=True)
    autodan_mode: str = field(default="ga", kw_only=True)
    check_lexicon: Callable[[], None] = field(default=lambda: None, kw_only=True)

    def __post_init__(self) -> None:
        chats = tuple(self.chats().values())
        if len({id(chat) for chat in chats}) != len(chats):
            raise ValueError("Participants must use independent LLMChat instances")

    def chats(self) -> dict[ModelRole, LLMChat]:
        chats: dict[ModelRole, LLMChat] = {"Target": self.target}
        if self.attacker is not None:
            chats["Attacker"] = self.attacker
        if self.evaluator.chat is not None:
            chats["Evaluator"] = self.evaluator.chat
        return chats

    def redactor(self) -> DiagnosticRedactor:
        return self.target.redactor.combine(*(chat.redactor for chat in self.chats().values()))

    def settings(self) -> dict[str, object]:
        redactor = self.redactor()
        return {
            "evaluator": redactor.snapshot(self.evaluator.settings()),
            **{role: chat.diagnostic_settings(redactor) for role, chat in self.chats().items()},
        }
