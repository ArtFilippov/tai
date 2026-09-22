"""Validate and compose a complete experiment before any work or artifacts."""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from attacks.fitd import FITDExperiment
from core.evaluation.assembly import EvaluatorSetup
from core.evaluation.model_evaluator import MODEL_RUBRIC_VERSION
from core.chats import LLMChat
from core.chats.qwen_server import QwenServer
from core.primitives.messages import Message
from core.runs.journal import RunJournal
from core.runs.journal_ports import Journal

from .chat_factory import chat_factory
from .configuration_values import MODEL_EVALUATOR, TEXT_EVALUATOR, round_limit, string, table
from .profiles import ParticipantProfile
from .server_factory import create_server

DEFAULT_SERVER_LOG = Path("runs/vllm.log")


@dataclass(frozen=True)
class RoleProfile(ParticipantProfile):
    assembly: ProfileAssembly | None = None
    journal_factory: Callable[..., Journal] = RunJournal

    def create_experiment(self, base: Message, journal_dir: Path | None = None) -> FITDExperiment:
        return FITDExperiment(
            self.target, self.attacker, base, self.evaluator,
            journal_dir=journal_dir, max_calls_per_conv=self.max_calls_per_conv,
            journal_factory=self.journal_factory,
        )

    def fresh(self) -> RoleProfile:
        if self.assembly is None:
            raise ValueError("A parallel series requires a participant factory")
        return self.assembly.create()

    def close(self) -> None:
        if self.assembly is not None:
            self.assembly.close()


@dataclass(frozen=True)
class ProfileAssembly:
    target: Callable[[], LLMChat]
    attacker: Callable[[], LLMChat]
    evaluator: Callable[[], EvaluatorSetup]
    rounds: int
    server: QwenServer | None

    def create(self) -> RoleProfile:
        return RoleProfile(self.target(), self.attacker(), self.evaluator(), self.rounds, self)

    def settings(self) -> dict[str, object]:
        return self.server.settings() if self.server is not None else {}

    def close(self) -> None:
        if self.server is not None:
            self.server.close()


def _read_config(config: Path) -> dict[str, object]:
    with config.open("rb") as source:
        try:
            return tomllib.load(source)
        except tomllib.TOMLDecodeError:
            # TOML errors can contain source snippets, including credentials.
            raise ValueError("Некорректный TOML; проверьте синтаксис конфигурации") from None


def _model_setup(value: object, server: QwenServer | None) -> Callable[[], EvaluatorSetup]:
    evaluation = table(value, "evaluator", {"type", "rubric_version", "chat"})
    if (rubric_version := string(evaluation.get("rubric_version"), "evaluator.rubric_version")) != MODEL_RUBRIC_VERSION:
        raise ValueError(f"evaluator.rubric_version: требуется {MODEL_RUBRIC_VERSION}")
    factory = chat_factory(evaluation.get("chat"), "evaluator.chat", server)
    return lambda: EvaluatorSetup.model(factory(), rubric_version)


def _evaluator(value: object, server: QwenServer | None, *, alert_series: bool) -> Callable[[], EvaluatorSetup]:
    evaluation = table(value, "evaluator")
    kind = evaluation.get("type")
    if alert_series and kind != MODEL_EVALUATOR:
        raise ValueError("ALERT series requires evaluator.type=model")
    if kind == TEXT_EVALUATOR:
        evaluation = table(evaluation, "evaluator", {"type", "criterion"})
        return partial(EvaluatorSetup.contains_text, string(evaluation.get("criterion"), "evaluator.criterion"))
    if kind == MODEL_EVALUATOR:
        return _model_setup(evaluation, server)
    raise ValueError("evaluator.type: поддерживается contains_text или model")


def load_profile(config: Path, *, alert_series: bool = False, server_log: Path = DEFAULT_SERVER_LOG) -> RoleProfile:
    """Build one validated profile shared by readiness and terminal experiments.

    Args:
        config (Path): Complete TOML experiment configuration.
        alert_series (bool, default=False): Require the pinned ALERT evaluator and round limit.
        server_log (Path, default=DEFAULT_SERVER_LOG): Separate server process output.

    Returns:
        RoleProfile: Independent chats and a reusable evaluator, without inference.
    """
    data = table(_read_config(config), "config", {"target", "attacker", "evaluator", "run", "server"})
    server = create_server(data.get("server"), server_log)
    return ProfileAssembly(
        chat_factory(data.get("target"), "target", server), chat_factory(data.get("attacker"), "attacker", server),
        _evaluator(data.get("evaluator"), server, alert_series=alert_series),
        round_limit(data.get("run", {}), alert_series=alert_series), server,
    ).create()
