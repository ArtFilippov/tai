"""Validated participants shared by readiness and experiment composition."""

from dataclasses import dataclass

from core.chats import LLMChat, ModelRole
from core.evaluation.configuration import EvaluatorSetup
from core.primitives.diagnostics import DiagnosticRedactor


@dataclass(frozen=True)
class ParticipantProfile:
    target: LLMChat
    attacker: LLMChat
    evaluator: EvaluatorSetup
    max_calls_per_conv: int

    def __post_init__(self) -> None:
        chats = tuple(self.chats().values())
        if len({id(chat) for chat in chats}) != len(chats):
            raise ValueError("Participants must use independent LLMChat instances")

    def chats(self) -> dict[ModelRole, LLMChat]:
        chats: dict[ModelRole, LLMChat] = {"Target": self.target, "Attacker": self.attacker}
        if self.evaluator.chat is not None:
            chats["Evaluator"] = self.evaluator.chat
        return chats

    def redactor(self) -> DiagnosticRedactor:
        return DiagnosticRedactor.combine(*(chat.redactor for chat in self.chats().values()))

    def settings(self) -> dict[str, object]:
        redactor = self.redactor()
        return {
            "evaluator": redactor.snapshot(self.evaluator.settings()),
            **{role: chat.diagnostic_settings(redactor) for role, chat in self.chats().items()},
        }
