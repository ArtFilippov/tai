"""Exercise GGUF HTTP identity, process ownership and conversation isolation."""

import socket
import subprocess
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from garak.attempt import Conversation, Message, Turn

from target.gguf_model import GGUFModel
from target.gguf.gguf_location import GGUFLocation
from target.gguf.gguf_server import GGUFServer
from target.gguf.gguf_server_utils import gguf_server, start
from target.model_chat import ModelChat
from target.gguf.gguf_file import GGUFFile

from .gguf.gguf_fixture_utils import failed_start, owned, service


class TestGGUFModel(unittest.TestCase):
    """Check real local HTTP exchanges and owned child-process lifetimes."""

    def model(self, port: int, tokens: int) -> GGUFModel:
        """Configure a model for an existing local test service.

        Args:
            port (int): Service port.
            tokens (int): Generation limit.

        Returns:
            GGUFModel: Model backed by the test service.
        """
        file = GGUFFile(
            "custom/repository", "commit", "pinned.gguf"
        )
        return GGUFModel(
            file,
            GGUFServer(
                file,
                GGUFLocation(
                    "127.0.0.1", port
                ),
                0, gguf_server
            ),
            tokens
        )

    def port(self) -> int:
        """Choose a vacant loopback port for a local service.

        Returns:
            int: Available TCP port.
        """
        with socket.socket() as connection:
            connection.bind(
                (
                    "127.0.0.1", 0
                )
            )
            return connection.getsockname()[1]

    def test_reuses_matching_server_without_stopping_it(self) -> None:
        port = self.port()
        with service(
            port, "pinned", "normal"
        ) as process:
            with gguf_server(
                "127.0.0.1", port,
                Path("unused.gguf"),
                0, expected="pinned"
            ) as owned:
                pass
            assert not owned and process.poll() is None, "An existing server was stopped or claimed"

    def test_rejects_different_model(self) -> None:
        port = self.port()
        with service(
            port, "different", "normal"
        ):
            with self.assertRaisesRegex(
                RuntimeError, "identity mismatch"
            ):
                with gguf_server(
                    "127.0.0.1", port,
                    Path("unused.gguf"),
                    0, expected="pinned"
                ):
                    pass

    def test_rejects_unusable_occupied_address(self) -> None:
        port = self.port()
        with service(
            port, "pinned", "unusable"
        ):
            with self.assertRaisesRegex(
                ConnectionError, "occupied"
            ):
                with gguf_server(
                    "127.0.0.1", port,
                    Path("unused.gguf"),
                    0, expected="pinned"
                ):
                    pass

    def test_rejects_malformed_identity_response(self) -> None:
        port = self.port()
        with service(
            port, "pinned", "malformed"
        ):
            with self.assertRaisesRegex(
                ValueError, "Malformed"
            ):
                with gguf_server(
                    "127.0.0.1", port,
                    Path("unused.gguf"),
                    0, expected="pinned"
                ):
                    pass

    def test_cleans_up_owned_server_on_failure(self) -> None:
        port = self.port()
        processes: list[subprocess.Popen[bytes]] = []
        with owned(processes):
            with self.assertRaisesRegex(
                RuntimeError, "experiment failed"
            ):
                with gguf_server(
                    "127.0.0.1", port,
                    Path("prepared.gguf"),
                    2, expected="pinned"
                ) as launched:
                    if launched:
                        raise RuntimeError("experiment failed")
        assert processes[0].poll() is not None, "The owned process survived the failed run"

    def test_cleans_up_owned_server_on_interruption(self) -> None:
        port = self.port()
        processes: list[subprocess.Popen[bytes]] = []
        with owned(processes):
            with self.assertRaises(KeyboardInterrupt):
                with gguf_server(
                    "127.0.0.1", port,
                    Path("prepared.gguf"),
                    2, expected="pinned"
                ):
                    raise KeyboardInterrupt
        assert processes[0].poll() is not None, "The owned process survived interruption"

    def test_cleans_up_owned_server_on_normal_exit(self) -> None:
        port = self.port()
        processes: list[subprocess.Popen[bytes]] = []
        with owned(processes):
            with gguf_server(
                "127.0.0.1", port,
                Path("prepared.gguf"),
                2, expected="pinned"
            ):
                pass
        assert processes[0].poll() is not None, "The owned process survived normal exit"

    def test_reports_server_startup_failure(self) -> None:
        port = self.port()
        with patch("target.gguf.gguf_server_utils.gpu_support"), patch(
            "target.gguf.gguf_server_utils.start", side_effect=failed_start
        ):
            with self.assertRaisesRegex(
                RuntimeError, "exited during startup"
            ):
                with gguf_server(
                    "127.0.0.1", port,
                    Path("prepared.gguf"),
                    2, expected="pinned"
                ):
                    pass

    def test_offline_missing_file_fails_clearly(self) -> None:
        with patch(
            "target.gguf.gguf_file.snapshot_download",
            side_effect=OSError("not cached")
        ):
            with self.assertRaisesRegex(
                FileNotFoundError, "tai-qwen prepare --attacker gguf"
            ):
                GGUFFile(
                    "custom/repository", "custom-commit", "chosen.Q4_K_M.gguf"
                ).local()

    def test_preparation_selects_only_pinned_quantization(self) -> None:
        with TemporaryDirectory() as directory:
            name = "chosen.Q4_K_M.gguf"
            path = Path(directory) / name
            path.write_bytes(b"prepared")
            with patch(
                "target.gguf.gguf_file.snapshot_download", return_value=directory
            ) as download:
                _ = GGUFFile(
                    "custom/repository", "custom-commit", name
                ).prepare()
                download.assert_called_once_with(
                    repo_id="custom/repository", revision="custom-commit",
                    allow_patterns=name, local_files_only=False
                )

    def test_launch_selects_cuda_and_disables_thinking(self) -> None:
        with patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,7"}
        ), patch("target.gguf.gguf_server_utils.subprocess.Popen") as process:
            _ = start(
                "127.0.0.1", 8765,
                Path("custom.gguf"),
                3, expected="pinned"
            )
            assert process.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "7" and (
                "--n_gpu_layers", "-1", "--main_gpu", "0", "--chat_template_kwargs", '{"enable_thinking": false}'
            ) == tuple(process.call_args.args[0][-6:]), "The launched server did not use CUDA and non-thinking mode"

    def test_keeps_role_conversations_separate(self) -> None:
        port = self.port()
        with service(
            port, "pinned", "normal"
        ):
            model = self.model(
                port, 24
            )
            target = ModelChat(model)
            attacker = ModelChat(model)
            _ = target.reply(
                Conversation(
                    [
                        Turn(
                            "user",
                            Message(text="Target secret")
                        )
                    ]
                )
            )
            response = attacker.reply(
                Conversation(
                    [
                        Turn(
                            "user",
                            Message(text="Attacker task")
                        )
                    ]
                )
            )
            assert response == "Attacker task", "The Target conversation leaked into the Attacker call"

    def test_reports_http_completion_failure(self) -> None:
        port = self.port()
        with service(
            port, "pinned", "error"
        ):
            with self.assertRaisesRegex(
                RuntimeError, "GGUF completion failed"
            ):
                (
                    self.model(
                        port, 24
                    ).generate(
                        [
                            {
                                "role": "user",
                                "content": "Hello"
                            }
                        ]
                    )
                )

    def test_reports_malformed_completion(self) -> None:
        port = self.port()
        with service(
            port, "pinned", "bad_reply"
        ):
            with self.assertRaisesRegex(
                ValueError, "Malformed GGUF completion"
            ):
                (
                    self.model(
                        port, 24
                    ).generate(
                        [
                            {
                                "role": "user",
                                "content": "Hello"
                            }
                        ]
                    )
                )
