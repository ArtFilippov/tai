"""Compose a pinned GGUF Attacker context for a selected address."""

from target.gguf.gguf_asset import GGUFAsset
from target.gguf.gguf_address import GGUFAddress
from target.gguf_model import GGUFModel
from target.gguf.gguf_run import GGUFRun
from target.gguf.gguf_server import GGUFServer
from target.gguf.gguf_server_utils import gguf_service

from target.gguf.envelopes.gguf_source_envelope import GGUFSourceEnvelope


class GGUFProfile(GGUFSourceEnvelope):
    """Bind one file, service and model without starting them before entry."""

    def __init__(self, file: GGUFAsset, location: GGUFAddress, gpu: int, tokens: int) -> None:
        """Connect a pinned file to one selected server and model.

        Args:
            file (GGUFAsset): Pinned quantization.
            location (GGUFAddress): Selected address.
            gpu (int): Selected CUDA device.
            tokens (int): Generation limit.
        """
        super().__init__(
            GGUFRun(
                file,
                GGUFServer(
                    file, location, gpu, lambda host, port, path, gpu, expected: gguf_service( host, port, path, gpu, expected=expected )
                ),
                tokens, lambda file, service, tokens: GGUFModel( file=file, service=service, tokens=tokens )
            )
        )
