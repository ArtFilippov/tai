"""Validate and compose a complete experiment before any work or artifacts."""

import tomllib
from pathlib import Path

from attacks.fitd import FITDExperiment
from core.evaluation.assembly import EvaluatorSetup
from core.evaluation.model_evaluator import MODEL_RUBRIC_VERSION
from core.primitives.messages import Message

from .chat_factory import create_chat
from .configuration_values import MODEL_EVALUATOR, TEXT_EVALUATOR, round_limit, string, table
from .profiles import ParticipantProfile


class RoleProfile(ParticipantProfile):
    def create_experiment(self, base: Message, journal_dir: Path | None = None) -> FITDExperiment:
        return FITDExperiment(
            self.target, self.attacker, base, self.evaluator,
            journal_dir=journal_dir, max_calls_per_conv=self.max_calls_per_conv,
        )


def _read_config(config: Path) -> dict[str, object]:
    with config.open("rb") as source:
        try:
            return tomllib.load(source)
        except tomllib.TOMLDecodeError:
            # TOML errors can contain source snippets, including credentials.
            raise ValueError("Некорректный TOML; проверьте синтаксис конфигурации") from None


def _model_setup(value: object) -> EvaluatorSetup:
    evaluation = table(value, "evaluator", {"type", "rubric_version", "chat"})
    if (rubric_version := string(evaluation.get("rubric_version"), "evaluator.rubric_version")) != MODEL_RUBRIC_VERSION:
        raise ValueError(f"evaluator.rubric_version: требуется {MODEL_RUBRIC_VERSION}")
    return EvaluatorSetup.model(create_chat(evaluation.get("chat"), "evaluator.chat"), rubric_version)


def _evaluator(value: object, *, alert_series: bool) -> EvaluatorSetup:
    evaluation = table(value, "evaluator")
    kind = evaluation.get("type")
    if alert_series and kind != MODEL_EVALUATOR:
        raise ValueError("ALERT series requires evaluator.type=model")
    if kind == TEXT_EVALUATOR:
        evaluation = table(evaluation, "evaluator", {"type", "criterion"})
        return EvaluatorSetup.contains_text(string(evaluation.get("criterion"), "evaluator.criterion"))
    if kind == MODEL_EVALUATOR:
        return _model_setup(evaluation)
    raise ValueError("evaluator.type: поддерживается contains_text или model")


def load_profile(config: Path, *, alert_series: bool = False) -> RoleProfile:
    """Build one validated profile shared by readiness and terminal experiments.

    Args:
        config (Path): Complete TOML experiment configuration.
        alert_series (bool, default=False): Require the pinned ALERT evaluator and round limit.

    Returns:
        RoleProfile: Independent chats and a reusable evaluator, without inference.
    """
    data = table(_read_config(config), "config", {"target", "attacker", "evaluator", "run"})
    return RoleProfile(
        create_chat(data.get("target"), "target"), create_chat(data.get("attacker"), "attacker"),
        _evaluator(data.get("evaluator"), alert_series=alert_series),
        round_limit(data.get("run", {}), alert_series=alert_series),
    )
