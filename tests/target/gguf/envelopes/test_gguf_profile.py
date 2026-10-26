"""Verify the assembled GGUF profile prepares and reuses a local service."""

import socket
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from target.gguf.envelopes.gguf_profile import GGUFProfile
from target.gguf.gguf_file import GGUFFile
from target.gguf.gguf_location import GGUFLocation

from ..gguf_fixture_utils import service


class TestGGUFProfile(unittest.TestCase):
    """Exercise the public preparation and Attacker context together."""

    def test_reopens_a_verified_server_for_separate_runs(self) -> None:
        with TemporaryDirectory() as directory:
            Path(
                directory, "pinned.gguf"
            ).write_bytes(b"prepared")
            file = GGUFFile(
                "lab/model", "commit", "pinned.gguf"
            )
            with socket.socket() as address:
                address.bind(
                    (
                        "127.0.0.1", 0
                    )
                )
                port = address.getsockname()[1]
            profile = GGUFProfile(
                file,
                GGUFLocation(
                    "127.0.0.1", port
                ),
                2, 24
            )
            with patch(
                "target.gguf.gguf_file.snapshot_download", return_value=directory
            ):
                with service(
                    port,
                    file.identity(),
                    "normal"
                ):
                    with profile.open() as model:
                        first = model.settings()["server"]
                    with profile.open() as model:
                        second = model.settings()["server"]
            assert (first, second) == ("reused", "reused"), "The profile could not reuse a verified service twice"
