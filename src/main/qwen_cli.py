"""Prepare pinned weights and run a local CUDA FITD experiment."""

from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
from pathlib import Path

from huggingface_hub import snapshot_download

from attack.fitd_attack import FITDAttack
from attack.fitd_probe import FITDProbe
from journal.configured_journal import ConfiguredJournal
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from target.continuing_chat import ContinuingChat
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.gguf_assets_utils import gguf_path
from target.gguf_model import GGUFModel
from target.gguf_server_utils import gguf_server
from target.model import Model
from target.model_chat import ModelChat
from target.qwen_device_utils import cuda_device


PREPARE = "prepare"
GGUF = "gguf"
REPOSITORY = "mradermacher/Huihui-Qwen3.5-2B-abliterated-GGUF"
REVISION = "7f5aee7d2dc4df18bd67eab85ea81e0bc2e98a2c"
FILENAME = "Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf"
IDENTITY = f"{REPOSITORY}@{REVISION}/{FILENAME}"


def snapshot(offline: bool) -> str:
    """Locate pinned weights, or download them only during preparation.

    Args:
        offline (bool): Whether to require already cached files.

    Returns:
        str: Path to the pinned model and tokenizer.

    Raises:
        FileNotFoundError: If an offline run has no prepared snapshot.
    """
    try:
        return snapshot_download(
            "Qwen/Qwen2.5-0.5B-Instruct",
            revision="7ae557604adf67be50417f59c2c2f167def9a775",
            local_files_only=offline
        )
    except OSError as exc:
        raise FileNotFoundError("Prepare the pinned Qwen snapshot before running offline") from exc


def main() -> None:
    """Prepare the pinned snapshot or run using local files on a chosen GPU."""
    parser = ArgumentParser(description="Local Qwen FITD experiment")
    commands = parser.add_subparsers(
        dest="command", required=True
    )
    prepare = commands.add_parser(
        PREPARE, help="Download pinned model and tokenizer"
    )
    _ = prepare.add_argument(
        "--attacker", choices=("qwen", "gguf"), default="qwen"
    )
    run = commands.add_parser(
        "run", help="Run offline FITD"
    )
    _ = run.add_argument(
        "journal", type=Path
    )
    _ = run.add_argument("base")
    _ = run.add_argument(
        "--gpu", type=int, required=True
    )
    _ = run.add_argument(
        "--turns", type=int, default=10
    )
    _ = run.add_argument(
        "--tokens", type=int, default=192
    )
    _ = run.add_argument(
        "--attacker", choices=("qwen", "gguf"), default="qwen"
    )
    _ = run.add_argument(
        "--host", default="127.0.0.1"
    )
    _ = run.add_argument(
        "--port", type=int, default=8080
    )
    launch(
        parser.parse_args()
    )


def gguf_identity() -> str:
    """Return the configured GGUF model identity.

    Returns:
        str: Pinned repository, revision and file.
    """
    return IDENTITY


def gguf_revision() -> str:
    """Return the pinned GGUF revision.

    Returns:
        str: Pinned commit.
    """
    return REVISION


def gguf_address(host: str, port: int) -> str:
    """Form the selected service origin.

    Args:
        host (str): Selected host.
        port (int): Selected port.

    Returns:
        str: HTTP origin.
    """
    return f"http://{host}:{port}"


def gguf_model(arguments: Namespace) -> Model:
    """Build one history-free GGUF Attacker.

    Args:
        arguments (Namespace): Selected address and token limit.

    Returns:
        Model: Selected Attacker.
    """
    return GGUFModel(
        gguf_address(
            arguments.host, arguments.port
        ),
        arguments.tokens,
        gguf_identity(),
        gguf_revision()
    )


def launch(arguments: Namespace) -> None:
    """Assemble one run from parsed command arguments.

    Args:
        arguments (Namespace): Parsed command arguments.

    Raises:
        ValueError: If the round limit is invalid.
        FileExistsError: If the journal already exists.
    """
    if arguments.command == PREPARE:
        print(gguf_path(False, REPOSITORY, REVISION, FILENAME) if arguments.attacker == GGUF else snapshot(False))
        return
    if arguments.turns < 1:
        raise ValueError("FITD round limit must be positive")
    if arguments.journal.exists():
        raise FileExistsError(arguments.journal)
    model = LazyQwenModel(
        cuda_device(arguments.gpu),
        arguments.tokens
    )
    _ = snapshot(True)
    path = gguf_path(True, REPOSITORY, REVISION, FILENAME) if arguments.attacker == GGUF else None
    context = gguf_server(arguments.host, arguments.port, path, arguments.gpu, expected=IDENTITY) if path is not None else nullcontext(False)
    with context as owned:
        attacker = model if path is None else gguf_model(arguments)
        arguments.journal.touch(exist_ok=False)
        execute(
            arguments, model, attacker, owned, gguf=path is not None
        )
    print(
        ParsedJournal(arguments.journal).read()
    )


def settings(model: Model, attacker: Model, arguments: Namespace, owned: bool, *, gguf: bool) -> dict[str, object]:
    """Describe the effective generation policy of each participant.

    Args:
        model (Model): Qwen Target and judge.
        attacker (Model): Selected Attacker.
        arguments (Namespace): Generation limit and selected CUDA GPU.
        owned (bool): Whether this run started a server.
        gguf (bool): Whether the Attacker uses GGUF.

    Returns:
        dict[str, object]: Model identities and generation settings.
    """
    qwen = {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": model.revision(),
        "generation": {
            "max_new_tokens": arguments.tokens,
            "do_sample": False,
            "pad_token_id": "eos_token_id"
        },
        "device": f"cuda:{arguments.gpu}",
        "dtype": "float16"
    }
    if not gguf:
        return {
            "target": qwen,
            "judge": qwen,
            "attacker": qwen
        }
    return {
        "target": qwen,
        "judge": qwen,
        "attacker": {
            "id": IDENTITY,
            "revision": attacker.revision(),
            "repository": REPOSITORY,
            "file": FILENAME,
            "generation": {
                "max_tokens": arguments.tokens,
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "min_p": 0,
                "repeat_penalty": 1
            },
            "server": "owned" if owned else "reused",
            "thinking": False if owned else "server-controlled",
            "device": f"cuda:{arguments.gpu}" if owned else "server-controlled",
            "server_settings": {
                "n_gpu_layers": -1,
                "main_gpu": 0,
                "n_ctx": 2048,
                "enable_thinking": False
            } if owned else "server-controlled"
        }
    }


def execute(arguments: Namespace, model: Model, attacker: Model, owned: bool, *, gguf: bool) -> None:
    """Give each role a separate conversation and record the FITD outcome.

    Args:
        arguments (Namespace): Base request and run limits.
        model (Model): Shared Qwen model for Target and judge.
        attacker (Model): Selected Attacker model.
        owned (bool): Whether this run started a server.
        gguf (bool): Whether the Attacker uses GGUF.
    """
    journal = ConfiguredJournal(
        DiskJournal(arguments.journal),
        settings(
            model, attacker, arguments, owned, gguf=gguf
        )
    )
    _ = FITDAttack(
        target=ModelChat(model),
        judge=RecordedJudge(
            QwenJudge(
                arguments.base,
                ModelChat(model)
            ),
            journal=journal, base=arguments.base
        ),
        journal=journal,
        probe=FITDProbe(
            base=arguments.base,
            attacker=ContinuingChat(
                ModelChat(attacker)
            ),
            journal=journal, turns=arguments.turns
        )
    ).run(arguments.base)
