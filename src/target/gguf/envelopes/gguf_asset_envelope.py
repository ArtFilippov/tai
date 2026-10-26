"""Forward asset operations to a selected GGUF quantization."""

from pathlib import Path
from typing import override

from target.gguf.gguf_asset import GGUFAsset


class GGUFAssetEnvelope(GGUFAsset):
    """Preserve the asset contract for pinned quantization variants."""

    def __init__(self, delegate: GGUFAsset) -> None:
        """Keep the selected quantization.

        Args:
            delegate (GGUFAsset): Asset receiving every operation.
        """
        self.delegate = delegate

    @override
    def prepare(self) -> Path:
        return self.delegate.prepare()

    @override
    def local(self) -> Path:
        return self.delegate.local()

    @override
    def identity(self) -> str:
        return self.delegate.identity()

    @override
    def commit(self) -> str:
        return self.delegate.commit()

    @override
    def details(self) -> dict[str, object]:
        return self.delegate.details()
