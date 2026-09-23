"""Assemble native FITD behind the observable probe façade and runtime port."""

from dataclasses import dataclass
from typing import override

from core.primitives.messages import Message

from attacks.shared.contracts import GARAK_REVISION, CompletionReason, StrategyPorts, StrategyRuntime
from .probe import FITDProbe
from .profile import FITD_PROFILE, FITDProfile
from .target import FITDTarget

__all__ = ["FITDProbe", "GarakRuntime", "GARAK_REVISION"]


@dataclass(frozen=True)
class GarakRuntime(StrategyRuntime):
    target_type: type[FITDTarget] = FITDTarget
    probe_type: type[FITDProbe] = FITDProbe
    profile: FITDProfile = FITD_PROFILE

    @property
    @override
    def revision(self) -> str:
        return GARAK_REVISION

    @override
    def __call__(self, base: Message, ports: StrategyPorts) -> CompletionReason:
        target = self.target_type(ports)
        probe = self.probe_type(base, ports, target, profile=self.profile)
        return probe()
