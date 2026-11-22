"""Sequential tense experiments with independent journals and failure boundaries."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import override

from attack.phrasing.phrasing_factory import PhrasingFactory
from attack.phrasing.phrasing_pair import PhrasingPair
from attack.phrasing.phrasing_plan import PhrasingPlan
from journal.journal_view_factory import JournalViewFactory


class JournaledPhrasingPair(PhrasingPair):
    """Reserve new artifacts and attempt both tenses despite ordinary run failures."""

    def __init__(self, directory: Path, plans: Sequence[PhrasingPlan], factory: PhrasingFactory, views: JournalViewFactory) -> None:
        """Keep deferred attack and read-side construction separate from execution.

        Args:
            directory (Path): New artifact directory.
            plans (Sequence[PhrasingPlan]): Past then future experiment plans.
            factory (PhrasingFactory): Construction of one independent attack.
            views (JournalViewFactory): Read-side journal construction.
        """
        super().__init__()
        self.directory = directory
        self.plans = plans
        self.factory = factory
        self.views = views

    @override
    def run(self) -> Mapping[str, Mapping[str, object]]:
        for plan in self.plans:
            _ = plan.generations()
        self.directory.mkdir(
            parents=True, exist_ok=False
        )
        outcomes: dict[str, Mapping[str, object]] = {}
        for plan in self.plans:
            name = f"phrasing_{plan.kind()}"
            path = self.directory / f"{name}.jsonl"
            try:
                path.touch(exist_ok=False)
                _ = (
                    self.factory(
                        plan, path
                    ).run()
                )
                outcomes[name] = (
                    self.views(path)
                    .read()
                )
            except Exception as exc:
                try:
                    outcome = (
                        self.views(path)
                        .read()
                    )
                except Exception as recovery:
                    outcome = {
                        "journal": str(path),
                        "recovery_error": repr(recovery)
                    }
                outcomes[name] = {
                    "reason": "execution_error",
                    **outcome,
                    "status": "failure",
                    "error": repr(exc)
                }
        return outcomes

    @override
    def description(self) -> dict[str, object]:
        return {
            "type": "phrasing_pair",
            "plans": [plan.description() for plan in self.plans]
        }
