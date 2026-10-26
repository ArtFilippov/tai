"""The pinned optional GGUF Attacker for ALERT and standalone FITD."""

from target.gguf.gguf_file import GGUFFile
from target.gguf.envelopes.gguf_asset_envelope import GGUFAssetEnvelope


class AlertGGUFFile(GGUFAssetEnvelope):
    """Keep one specific quantization and revision as a reusable asset."""

    def __init__(self) -> None:
        """Bind the tested GGUF file identity."""
        super().__init__(
            GGUFFile(
                "mradermacher/Huihui-Qwen3.5-2B-abliterated-GGUF",
                "7f5aee7d2dc4df18bd67eab85ea81e0bc2e98a2c",
                "Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf"
            )
        )
