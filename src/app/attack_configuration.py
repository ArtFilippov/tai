"""Validate strategy selection and its own parameters before model assembly."""

from attacks.adaptive_attacks.profile import ADAPTIVE_PROFILE
from attacks.badchars.profile import PARAMETERS, DeletionProfile, deletion_profile
from attacks.dra.profile import PARAMETERS as DRA_PARAMETERS, DRAProfile, dra_profile
from attacks.fitd.profile import FITDProfile
from attacks.shared.contracts import ADAPTIVE_ATTACKS, BADCHARS_DELETION, DRA, FITD, AttackType, StrategyConfiguration

from .configuration_values import round_limit, table

ATTACKER_SECTION = "attacker"


def strategy_configuration(kind: AttackType, rounds: int | None) -> StrategyConfiguration:
    if kind == ADAPTIVE_ATTACKS:
        return ADAPTIVE_PROFILE
    if kind == BADCHARS_DELETION:
        return DeletionProfile().configuration()
    if kind == DRA:
        return DRAProfile().configuration()
    assert rounds is not None
    return FITDProfile(max_calls_per_conv=rounds).configuration()


def attack_configuration(data: dict[str, object]) -> StrategyConfiguration:
    attack = table(data.get("attack", {"type": FITD}), "attack")
    kind = attack.get("type")
    allowed = {"type"} | (set(PARAMETERS) if kind == BADCHARS_DELETION else set(DRA_PARAMETERS) if kind == DRA else set[str]())
    attack = table(attack, "attack", allowed)
    if kind == FITD:
        return strategy_configuration(FITD, round_limit(data.get("run", {})))
    return _candidate_configuration(data, attack)


def _candidate_configuration(data: dict[str, object], attack: dict[str, object]) -> StrategyConfiguration:
    if (kind := attack.get("type")) not in (ADAPTIVE_ATTACKS, BADCHARS_DELETION, DRA):
        raise ValueError("attack.type: поддерживается fitd, adaptive_attacks, badchars_deletion или dra")
    if ATTACKER_SECTION in data:
        raise ValueError(f"{kind}: роль Attacker не используется")
    run = table(data.get("run", {}), "run", {"generations"})
    _ = round_limit(run)
    if kind == BADCHARS_DELETION:
        return deletion_profile(attack).configuration()
    return dra_profile(attack).configuration() if kind == DRA else ADAPTIVE_PROFILE
