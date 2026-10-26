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
from target.envelopes.gguf_profile import GGUFProfile
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.gguf_file import GGUFFile
from target.gguf_location import GGUFLocation
from target.model import Model
from target.model_chat import ModelChat
from target.qwen_device_utils import cuda_device


PREPARE = "prepare"
GGUF = "gguf"


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


def launch(arguments: Namespace) -> None:
    """Assemble one run from parsed command arguments.

    Args:
        arguments (Namespace): Parsed command arguments.

    Raises:
        ValueError: If the round limit is invalid.
        FileExistsError: If the journal already exists.
    """
    file = GGUFFile(
        "mradermacher/Huihui-Qwen3.5-2B-abliterated-GGUF",
        "7f5aee7d2dc4df18bd67eab85ea81e0bc2e98a2c",
        "Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf"
    )
    if arguments.command == PREPARE:
        print(file.prepare() if arguments.attacker == GGUF else snapshot(False))
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
    context = GGUFProfile(
        file, GGUFLocation(arguments.host, arguments.port), arguments.gpu, arguments.tokens
    ).open() if arguments.attacker == GGUF else nullcontext(model)
    with context as attacker:
        arguments.journal.touch(exist_ok=False)
        execute(
            arguments, model, attacker
        )
    print(
        ParsedJournal(arguments.journal).read()
    )


def execute(arguments: Namespace, model: Model, attacker: Model) -> None:
    """Give each role a separate conversation and record the FITD outcome.

    Args:
        arguments (Namespace): Base request and run limits.
        model (Model): Shared Qwen model for Target and judge.
        attacker (Model): Selected Attacker model.
    """
    journal = ConfiguredJournal(
        DiskJournal(arguments.journal),
        model, attacker
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
