"""Apply one independent stochastic policy to a shared inference resource."""

from collections.abc import Sequence
from typing import override

from target.model import Model
from target.sampling import Sampling


class SampledModel(Model):
    """Share model weights while owning a separate seed-zero decoding stream."""

    def __init__(self, delegate: Model, sampling: Sampling) -> None:
        """Bind a resource without loading weights or resetting its settings.

        Args:
            delegate (Model): Shared history-free inference resource.
            sampling (Sampling): Independent per-run decoding policy and stream.
        """
        super().__init__()
        self.delegate = delegate
        self.sampling = sampling

    @override
    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        return self.delegate.sample(
            messages, self.sampling
        )

    @override
    def sample(self, messages: Sequence[dict[str, str]], sampling: Sampling) -> str:
        return self.delegate.sample(
            messages, sampling
        )

    @override
    def decoding(self, sampling: Sampling) -> dict[str, object]:
        return self.delegate.decoding(sampling)

    @override
    def prefill(self, messages: Sequence[dict[str, str]], prefix: str) -> str:
        raise NotImplementedError("Sampled assistant prefill is not supported")

    @override
    def revision(self) -> str:
        return self.delegate.revision()

    @override
    def settings(self) -> dict[str, object]:
        return {
            **self.delegate.settings(),
            "generation": self.delegate.decoding(self.sampling)
        }
