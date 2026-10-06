"""Validate strategy selection and its own parameters before model assembly."""

from attacks.adaptive_attacks.profile import ADAPTIVE_PROFILE
from attacks.badchars.profile import PARAMETERS, DeletionProfile, deletion_profile
from attacks.dra.profile import PARAMETERS as DRA_PARAMETERS, DRAProfile, dra_profile
from attacks.fitd.profile import FITDProfile
from attacks.goat.profile import GOATProfile
from attacks.autodan.profile import PARAMETERS as AUTODAN_PARAMETERS, autodan_profile
from attacks.shared.contracts import ADAPTIVE_ATTACKS, AUTODAN, BADCHARS_DELETION, DRA, FITD, GOAT, THREATEN_JSON, AttackType, StrategyConfiguration

from .configuration_values import QWEN_BACKEND, round_limit, table

ATTACKER_SECTION = "attacker"
TYPE = "type"


def strategy_configuration(kind: AttackType, rounds: int | None) -> StrategyConfiguration:
    if kind == AUTODAN:
        raise ValueError("autodan requires an explicit expected continuation")
    if kind in (ADAPTIVE_ATTACKS, THREATEN_JSON):
        return (ADAPTIVE_PROFILE if kind == ADAPTIVE_ATTACKS else
                StrategyConfiguration(THREATEN_JSON, {"attempts": ("control", "pressured")}))
    if kind == BADCHARS_DELETION:
        return DeletionProfile().configuration()
    if kind == DRA:
        return DRAProfile().configuration()
    assert rounds is not None
    return (GOATProfile(rounds) if kind == GOAT else FITDProfile(max_calls_per_conv=rounds)).configuration()


def attack_configuration(data: dict[str, object]) -> StrategyConfiguration:
    attack = table(data.get("attack", {"type": FITD}), "attack")
    kind = attack.get("type")
    key = kind if isinstance(kind, str) else ""
    allowed = {"type"} | {BADCHARS_DELETION: set(PARAMETERS), DRA: set(DRA_PARAMETERS), AUTODAN: set(AUTODAN_PARAMETERS)}.get(key, set())
    attack = table(attack, "attack", allowed)
    if kind == FITD:
        return strategy_configuration(FITD, round_limit(data.get("run", {})))
    if kind == GOAT:
        if ATTACKER_SECTION not in data:
            raise ValueError("goat: требуется роль Attacker")
        run = table(data.get("run", {}), "run")
        rounds = round_limit({"max_calls_per_conv": GOATProfile().max_calls_per_conv, **run})
        return GOATProfile(rounds).configuration()
    if kind == AUTODAN:
        return _autodan_configuration(data, attack)
    return _candidate_configuration(data, attack)


def _candidate_configuration(data: dict[str, object], attack: dict[str, object]) -> StrategyConfiguration:
    if (kind := attack.get("type")) not in (ADAPTIVE_ATTACKS, BADCHARS_DELETION, DRA, THREATEN_JSON):
        raise ValueError("attack.type: поддерживается fitd, adaptive_attacks, badchars_deletion, dra, goat, autodan или threaten_json")
    if ATTACKER_SECTION in data:
        raise ValueError(f"{kind}: роль Attacker не используется")
    run = table(data.get("run", {}), "run", {"generations"})
    _ = round_limit(run)
    if kind == THREATEN_JSON:
        return strategy_configuration(THREATEN_JSON, None)
    if kind == BADCHARS_DELETION:
        return deletion_profile(attack).configuration()
    return dra_profile(attack).configuration() if kind == DRA else ADAPTIVE_PROFILE


def _autodan_configuration(data: dict[str, object], attack: dict[str, object]) -> StrategyConfiguration:
    if ATTACKER_SECTION not in data:
        raise ValueError("autodan: требуется роль Attacker")
    if table(data.get("target"), "target").get("backend", "openai") != QWEN_BACKEND:
        raise ValueError("autodan: Target requires backend=qwen with local logits scoring")
    run = table(data.get("run", {}), "run", {"generations"})
    _ = round_limit({"max_calls_per_conv": 1, **run})
    return autodan_profile({key: value for key, value in attack.items() if key != TYPE}).configuration()
