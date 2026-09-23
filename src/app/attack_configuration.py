"""Validate strategy selection and its own parameters before model assembly."""

from attacks.adaptive_attacks.profile import ADAPTIVE_PROFILE
from attacks.fitd.profile import FITDProfile
from attacks.shared.contracts import ADAPTIVE_ATTACKS, FITD, AttackType, StrategyConfiguration

from .configuration_values import round_limit, table

ATTACKER_SECTION = "attacker"


def strategy_configuration(kind: AttackType, rounds: int | None) -> StrategyConfiguration:
    if kind == ADAPTIVE_ATTACKS:
        return ADAPTIVE_PROFILE
    assert rounds is not None
    return FITDProfile(max_calls_per_conv=rounds).configuration()


def attack_configuration(data: dict[str, object]) -> StrategyConfiguration:
    attack = table(data.get("attack", {"type": FITD}), "attack", {"type"})
    kind = attack.get("type")
    if kind == FITD:
        return strategy_configuration(FITD, round_limit(data.get("run", {})))
    if kind != ADAPTIVE_ATTACKS:
        raise ValueError("attack.type: поддерживается fitd или adaptive_attacks")
    if ATTACKER_SECTION in data:
        raise ValueError("adaptive_attacks: роль Attacker не используется")
    run = table(data.get("run", {}), "run", {"generations"})
    _ = round_limit(run)
    return ADAPTIVE_PROFILE
