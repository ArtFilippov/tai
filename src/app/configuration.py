"""Validate and compose a complete experiment before any work or artifacts."""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import cast

from attacks.fitd import FITDExperiment
from attacks.adaptive_attacks import AdaptiveAttacksExperiment
from attacks.badchars import BadCharactersDeletionExperiment
from attacks.dra import DRAExperiment
from attacks.threaten_json import ThreatenJSONExperiment
from attacks.threaten_json.evaluator import JSONAnswerEvaluator
from attacks.goat import GOATExperiment
from attacks.autodan import AutoDANExperiment
from attacks.autodan.profile import AutoDANOptions, autodan_profile
from attacks.autodan.genetic import require_hga_resources
from attacks.autodan.profile import HGA
from attacks.badchars.profile import deletion_profile
from attacks.dra.profile import dra_profile
from attacks.shared.contracts import ADAPTIVE_ATTACKS, AUTODAN, BADCHARS_DELETION, DRA, FITD, GOAT, THREATEN_JSON, StrategyConfiguration, AttackType
from attacks.shared.experiment import Experiment
from core.evaluation.assembly import EvaluatorSetup
from core.evaluation.model_evaluator import ANSWER_RUBRIC_VERSION, MODEL_RUBRIC_VERSION
from core.chats import LLMChat
from core.chats.qwen_server import QwenServer
from core.primitives.messages import Message
from core.runs.journal import RunJournal
from core.runs.journal_ports import Journal

from .chat_factory import chat_factory
from .attack_configuration import attack_configuration, strategy_configuration
from .configuration_values import MODEL_EVALUATOR, TEXT_EVALUATOR, string, table
from .profiles import ParticipantProfile
from .server_factory import create_server

DEFAULT_SERVER_LOG = Path("runs/vllm.log")
GENERATIONS_SETTING = "generations"


@dataclass(frozen=True)
class RoleProfile(ParticipantProfile):
    assembly: ProfileAssembly | None = None
    journal_factory: Callable[..., Journal] = RunJournal

    def create_experiment(self, base: Message, journal_dir: Path | None = None) -> Experiment:
        if self.attack_type == AUTODAN:
            return self._autodan(base, journal_dir)
        if self.attack_type in (ADAPTIVE_ATTACKS, THREATEN_JSON):
            experiment_type = ThreatenJSONExperiment if self.strategy().attack_type == THREATEN_JSON else AdaptiveAttacksExperiment
            return experiment_type(self.target, base, self.evaluator, journal_dir=journal_dir, journal_factory=self.journal_factory)
        if self.attack_type in (BADCHARS_DELETION, DRA):
            return self._candidate(base, journal_dir)
        assert self.attacker is not None and self.max_calls_per_conv is not None
        experiment_type = GOATExperiment if self.attack_type == GOAT else FITDExperiment
        return experiment_type(
            self.target, self.attacker, base, self.evaluator, journal_dir=journal_dir,
            max_calls_per_conv=self.max_calls_per_conv, journal_factory=self.journal_factory,
        )

    def _candidate(self, base: Message, journal_dir: Path | None) -> Experiment:
        if self.attack_type == BADCHARS_DELETION:
            profile = deletion_profile(self.strategy().parameters)
            return BadCharactersDeletionExperiment(
                self.target, base, self.evaluator, journal_dir=journal_dir, journal_factory=self.journal_factory,
                perturbation_budget=profile.perturbation_budget, max_position_candidates=profile.max_position_candidates,
                max_ascii_variants=profile.max_ascii_variants,
            )
        dra = dra_profile(self.strategy().parameters)
        return DRAExperiment(
            self.target, base, self.evaluator, journal_dir=journal_dir, journal_factory=self.journal_factory,
            max_variants=dra.max_variants, seed=dra.seed, toxic_trunc=dra.toxic_trunc,
            benign_trunc=dra.benign_trunc, benign_full_token_prob=dra.benign_full_token_prob,
        )

    def _autodan(self, base: Message, journal_dir: Path | None) -> AutoDANExperiment:
        assert self.attacker is not None
        autodan = autodan_profile(self.strategy().parameters)
        return AutoDANExperiment(
            self.target, self.attacker, base, self.evaluator, journal_dir=journal_dir,
            journal_factory=self.journal_factory,
            **cast(AutoDANOptions, {key: value for key, value in autodan.configuration().parameters.items() if key != GENERATIONS_SETTING}),
        )

    def strategy(self) -> StrategyConfiguration:
        if self.assembly is not None:
            return self.assembly.strategy
        return strategy_configuration(self.attack_type, self.max_calls_per_conv)

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
    attacker: Callable[[], LLMChat] | None
    evaluator: Callable[[], EvaluatorSetup]
    strategy: StrategyConfiguration
    server: QwenServer | None

    def create(self) -> RoleProfile:
        return RoleProfile(
            self.target(), self.attacker() if self.attacker is not None else None, self.evaluator(),
            self.strategy.max_calls_per_conv, self, attack_type=self.strategy.attack_type,
            autodan_mode=str(self.strategy.parameters.get("mode", "ga")) if self.strategy.attack_type == AUTODAN else "ga",
            check_lexicon=require_hga_resources if self.strategy.attack_type == AUTODAN and self.strategy.parameters.get("mode") == HGA else lambda: None,
        )

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


def _model_setup(value: object, server: QwenServer | None, rubric: str,
                 create: Callable[[str], Message]) -> Callable[[], EvaluatorSetup]:
    evaluation = table(value, "evaluator", {"type", "rubric_version", "chat"})
    if string(evaluation.get("rubric_version"), "evaluator.rubric_version") != rubric:
        raise ValueError(f"evaluator.rubric_version: требуется {rubric}")
    factory = chat_factory(evaluation.get("chat"), "evaluator.chat", create, server)
    def answer() -> EvaluatorSetup:
        chat = factory()
        setup = EvaluatorSetup.answer(chat, create)
        return EvaluatorSetup(JSONAnswerEvaluator(setup.evaluator), chat, setup.configuration)

    if rubric == ANSWER_RUBRIC_VERSION:
        return answer
    return lambda: EvaluatorSetup.model(factory(), create)


def _evaluator(value: object, server: QwenServer | None, create: Callable[[str], Message],
               *, alert_series: bool, attack_type: AttackType) -> Callable[[], EvaluatorSetup]:
    evaluation = table(value, "evaluator")
    kind = evaluation.get("type")
    if (alert_series or attack_type == THREATEN_JSON) and kind != MODEL_EVALUATOR:
        raise ValueError("ThreatenJSON requires evaluator.type=model" if attack_type == THREATEN_JSON
                         else "ALERT series requires evaluator.type=model")
    if kind == TEXT_EVALUATOR:
        evaluation = table(evaluation, "evaluator", {"type", "criterion"})
        return partial(EvaluatorSetup.contains_text, string(evaluation.get("criterion"), "evaluator.criterion"))
    if kind == MODEL_EVALUATOR:
        return _model_setup(
            evaluation, server,
            ANSWER_RUBRIC_VERSION if attack_type == THREATEN_JSON else MODEL_RUBRIC_VERSION,
            create,
        )
    raise ValueError("evaluator.type: поддерживается contains_text или model")


def load_profile(config: Path, create: Callable[[str], Message], *, alert_series: bool = False,
                 server_log: Path = DEFAULT_SERVER_LOG) -> RoleProfile:
    """Build one validated profile shared by readiness and terminal experiments.

    Args:
        config (Path): Complete TOML experiment configuration.
        create (Callable[[str], Message]): Assembled textual result constructor.
        alert_series (bool, default=False): Require the pinned ALERT model evaluator.
        server_log (Path, default=DEFAULT_SERVER_LOG): Separate server process output.

    Returns:
        RoleProfile: Independent chats and a reusable evaluator, without inference.
    """
    data = table(_read_config(config), "config", {"attack", "target", "attacker", "evaluator", "run", "server"})
    strategy = attack_configuration(data)
    server = create_server(data.get("server"), server_log)
    return ProfileAssembly(
        chat_factory(data.get("target"), "target", create, server),
        chat_factory(data.get("attacker"), "attacker", create, server)
        if strategy.attack_type in (FITD, GOAT, AUTODAN) else None,
        _evaluator(
            data.get("evaluator"), server, create,
            alert_series=alert_series, attack_type=strategy.attack_type,
        ),
        strategy, server,
    ).create()
