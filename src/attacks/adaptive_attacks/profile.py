"""Fixed, template-only Adaptive Attacks profile; no online search."""

from types import MappingProxyType

from attacks.shared.contracts import StrategyConfiguration

VARIANT_COUNT = 5
ADAPTIVE_PROFILE = StrategyConfiguration("adaptive_attacks", MappingProxyType({
    "generations": 1, "variants": VARIANT_COUNT, "online_search": False,
    "early_stop": "evaluator_success",
}))
