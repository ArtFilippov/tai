"""Native conditional formatting bound to a configured Target resource."""

from types import SimpleNamespace
from typing import override

from prompt_collection.dan_text import DanText
from target.model import Model


class TargetDanText(DanText):
    """Preserve garak's Target-name formatting without importing its probe state."""

    def __init__(self, target: Model) -> None:
        """Bind the configured model without reading or starting it.

        Args:
            target (Model): Target resource whose settings declare its identity.
        """
        super().__init__()
        self.target = target

    @override
    def render(self, text: str) -> str:
        match (
            self.target.settings()
            .get("id")
        ):
            case str() as identity if identity:
                return text.format(generator=SimpleNamespace(name=identity)) if "{generator.name}" in text else text
            case _:
                raise ValueError("DAN requires the configured Target model identity")

    @override
    def description(self) -> dict[str, object]:
        return {
            "source": "garak.probes.dan",
            "revision": "3f50ea5ff9cd7050099940647c15c39b07a93392",
        }
