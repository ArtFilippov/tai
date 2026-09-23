"""Validate strategy selection and its own parameters before model assembly."""

from attacks.adaptive_attacks.profile import ADAPTIVE_PROFILE
from attacks.badchars.profile import PARAMETERS, DeletionProfile, deletion_profile
from attacks.fitd.profile import FITDProfile
from attacks.shared.contracts import ADAPTIVE_ATTACKS, BADCHARS_DELETION, FITD, AttackType, StrategyConfiguration

from .configuration_values import round_limit, table

ATTACKER_SECTION = "attacker"


def strategy_configuration(kind: AttackType, rounds: int | None) -> StrategyConfiguration:
    if kind == ADAPTIVE_ATTACKS:
        return ADAPTIVE_PROFILE
    if kind == BADCHARS_DELETION:
        return DeletionProfile().configuration()
    assert rounds is not None
    return FITDProfile(max_calls_per_conv=rounds).configuration()


def attack_configuration(data: dict[str, object]) -> StrategyConfiguration:
    attack = table(data.get("attack", {"type": FITD}), "attack")
    kind = attack.get("type")
    allowed = {"type"} | (set(PARAMETERS) if kind == BADCHARS_DELETION else set[str]())
    attack = table(attack, "attack", allowed)
    if kind == FITD:
        return strategy_configuration(FITD, round_limit(data.get("run", {})))
    return _candidate_configuration(data, attack)


def _candidate_configuration(data: dict[str, object], attack: dict[str, object]) -> StrategyConfiguration:
    if (kind := attack.get("type")) not in (ADAPTIVE_ATTACKS, BADCHARS_DELETION):
        raise ValueError("attack.type: поддерживается fitd, adaptive_attacks или badchars_deletion")
    if ATTACKER_SECTION in data:
        raise ValueError(f"{kind}: роль Attacker не используется")
    run = table(data.get("run", {}), "run", {"generations"})
    _ = round_limit(run)
    return deletion_profile(attack).configuration() if kind == BADCHARS_DELETION else ADAPTIVE_PROFILE
