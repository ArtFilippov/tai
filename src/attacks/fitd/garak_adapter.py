"""Assemble native FITD behind the observable probe façade and runtime port."""

from dataclasses import dataclass
from typing import override

from core.primitives.messages import Message

from .contracts import CompletionReason, StrategyPorts, StrategyRuntime
from .probe import FITDProbe
from .profile import GARAK_REVISION, FITDProfile
from .target import FITDTarget

__all__ = ["FITDProbe", "GarakRuntime", "GARAK_REVISION"]


@dataclass(frozen=True)
class GarakRuntime(StrategyRuntime):
    target_type: type[FITDTarget] = FITDTarget
    probe_type: type[FITDProbe] = FITDProbe

    @property
    @override
    def revision(self) -> str:
        return GARAK_REVISION

    @override
    def __call__(self, base: Message, ports: StrategyPorts, profile: FITDProfile) -> CompletionReason:
        target = self.target_type(ports)
        probe = self.probe_type(base, ports, target, profile=profile)
        return probe()
