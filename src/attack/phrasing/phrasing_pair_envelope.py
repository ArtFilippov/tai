"""Forward the independent tense-pair execution boundary."""

from collections.abc import Mapping
from typing import override

from attack.phrasing.phrasing_pair import PhrasingPair


class PhrasingPairEnvelope(PhrasingPair):
    """Preserve the pair contract for composed execution variants."""

    def __init__(self, delegate: PhrasingPair) -> None:
        """Store the independently executing pair.

        Args:
            delegate (PhrasingPair): Pair receiving the run operation.
        """
        super().__init__()
        self.delegate = delegate

    @override
    def run(self) -> Mapping[str, Mapping[str, object]]:
        return self.delegate.run()

    @override
    def description(self) -> dict[str, object]:
        return self.delegate.description()
