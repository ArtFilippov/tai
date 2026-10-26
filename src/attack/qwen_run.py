"""An isolated attempt bound to reusable model resources."""

from dataclasses import dataclass
from pathlib import Path
from typing import override

from attack.alert_run import AlertRun
from attack.run_operation import RunOperation
from base.base_request import BaseRequest
from target.model import Model


@dataclass(eq=False)
class QwenRun(AlertRun):
    """Keep model resources shared while each invocation builds new conversations.

    Attributes:
        model (Model): Shared Target and judge model.
        attacker (Model): Shared Attacker model.
        turns (int): Maximum exchanges.
        execute (RunOperation): Per-attempt assembly.
    """

    model: Model
    attacker: Model
    turns: int
    execute: RunOperation

    @override
    def run(self, base: BaseRequest, path: Path) -> None:
        """Assemble independent roles and journal for one base.

        Args:
            base (BaseRequest): Selected request.
            path (Path): Unique journal destination.
        """
        self.execute(
            base, path, self.model, self.attacker, turns=self.turns
        )
