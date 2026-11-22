"""Read-side construction for independently stored run journals."""

from pathlib import Path
from typing import Protocol

from journal.journal_view import JournalView


class JournalViewFactory(Protocol):
    """Open recorded facts for one run without creating or evaluating them."""

    def __call__(self, path: Path) -> JournalView:
        ...
