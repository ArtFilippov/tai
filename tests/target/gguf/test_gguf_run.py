"""Verify the reusable GGUF context exposes truthful Attacker settings."""

import socket
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from target.gguf.gguf_file import GGUFFile
from target.gguf.gguf_location import GGUFLocation
from target.gguf_model import GGUFModel
from target.gguf.gguf_run import GGUFRun
from target.gguf.gguf_server import GGUFServer
from target.gguf.gguf_server_utils import gguf_server

from .gguf_fixture_utils import owned, service


class TestGGUFRun(unittest.TestCase):
    """Check journal settings from real reused and owned HTTP services."""

    def port(self) -> int:
        """Select a free local port for an independent service.

        Returns:
            int: Free TCP port.
        """
        with socket.socket() as address:
            address.bind(
                (
                    "127.0.0.1", 0
                )
            )
            return address.getsockname()[1]

    def test_reused_model_reports_server_controlled_settings(self) -> None:
        with TemporaryDirectory() as directory:
            Path(
                directory, "pinned.gguf"
            ).write_bytes(b"prepared")
            file = GGUFFile(
                "lab/model", "commit", "pinned.gguf"
            )
            address = self.port()
            server = GGUFServer(
                file,
                GGUFLocation(
                    "127.0.0.1", address
                ),
                2, gguf_server
            )
            profile = GGUFRun(
                file, server, 24, GGUFModel
            )
            with patch(
                "target.gguf.gguf_file.snapshot_download", return_value=directory
            ):
                with service(
                    address,
                    file.identity(),
                    "normal"
                ):
                    with profile.open() as model:
                        settings = model.settings()
            assert settings == {
                "id": "lab/model@commit/pinned.gguf",
                "revision": "commit",
                "repository": "lab/model",
                "file": "pinned.gguf",
                "generation": {
                    "max_tokens": 24,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": 1,
                    "min_p": 0,
                    "repeat_penalty": 1
                },
                "server": "reused",
                "thinking": "server-controlled",
                "device": "server-controlled",
                "server_settings": "server-controlled"
            }, "A reused Attacker reported settings that were not verified"

    def test_owned_model_reports_selected_gpu_and_server_settings(self) -> None:
        with TemporaryDirectory() as directory:
            Path(
                directory, "pinned.gguf"
            ).write_bytes(b"prepared")
            file = GGUFFile(
                "lab/model", "commit", "pinned.gguf"
            )
            server = GGUFServer(
                file,
                GGUFLocation(
                    "127.0.0.1",
                    self.port()
                ),
                2, gguf_server
            )
            profile = GGUFRun(
                file, server, 24, GGUFModel
            )
            processes: list[subprocess.Popen[bytes]] = []
            with patch(
                "target.gguf.gguf_file.snapshot_download", return_value=directory
            ):
                with owned(processes):
                    with profile.open() as model:
                        settings = model.settings()
            assert (settings["server"], settings["device"], settings["thinking"], settings["server_settings"]) == (
                "owned", "cuda:2", False,
                {
                    "n_gpu_layers": -1,
                    "main_gpu": 0,
                    "n_ctx": 2048,
                    "enable_thinking": False
                }
            ), "An owned Attacker omitted its selected server settings"
