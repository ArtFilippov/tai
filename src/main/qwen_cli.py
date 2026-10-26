"""Standalone Qwen FITD preparation and one-journal execution."""

from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
from pathlib import Path

from attack.fitd_attack import FITDAttack
from attack.fitd_probe import FITDProbe
from base.plain_base import PlainBase
from journal.disk_journal import DiskJournal
from journal.journal import Journal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from target.continuing_chat import ContinuingChat
from target.envelopes.lazy_qwen_model import LazyQwenModel
from target.qwen_snapshot_utils import snapshot
from target.gguf.envelopes.alert_gguf_file import AlertGGUFFile
from target.gguf.envelopes.gguf_profile import GGUFProfile
from target.gguf.gguf_location import GGUFLocation
from target.model import Model
from target.model_chat import ModelChat
from target.qwen_device_utils import cuda_device

GGUF = "gguf"
PREPARE = "prepare"


def qwen_main() -> None:
    """Prepare pinned weights or run a local CUDA FITD experiment."""
    parser = ArgumentParser(description="Local Qwen FITD experiment")
    commands = parser.add_subparsers(
        dest="command", required=True
    )
    preparation = commands.add_parser(
        PREPARE, help="Download pinned model and tokenizer"
    )
    _ = preparation.add_argument(
        "--attacker",
        choices=(
            "qwen", GGUF
        ),
        default="qwen"
    )
    command = commands.add_parser(
        "run", help="Run offline FITD"
    )
    _ = command.add_argument(
        "journal", type=Path
    )
    _ = command.add_argument("base")
    _ = command.add_argument(
        "--gpu", type=int, required=True
    )
    _ = command.add_argument(
        "--turns", type=int, default=10
    )
    _ = command.add_argument(
        "--tokens", type=int, default=192
    )
    _ = command.add_argument(
        "--attacker",
        choices=(
            "qwen", GGUF
        ),
        default="qwen"
    )
    _ = command.add_argument(
        "--host", default="127.0.0.1"
    )
    _ = command.add_argument(
        "--port", type=int, default=8080
    )
    qwen_launch(
        parser.parse_args()
    )


def qwen_launch(arguments: Namespace) -> None:
    """Execute one standalone attempt, keeping all roles independent.

    Args:
        arguments (Namespace): Parsed command arguments.

    Raises:
        ValueError: If the round limit is invalid.
        FileExistsError: If the journal already exists.
    """
    if arguments.command == PREPARE:
        print(AlertGGUFFile().prepare() if arguments.attacker == GGUF else snapshot(False))
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
    context = (
        GGUFProfile(AlertGGUFFile(), GGUFLocation(arguments.host, arguments.port), arguments.gpu, arguments.tokens).open()
        if arguments.attacker == GGUF else nullcontext(model)
    )
    with context as attacker:
        arguments.journal.touch(exist_ok=False)
        run(
            arguments.base, model, attacker,
            VersionedJournal(
                DiskJournal(arguments.journal)
            ),
            turns=arguments.turns
        )
    print(
        ParsedJournal(arguments.journal).read()
    )


def run(base: str, model: Model, attacker: Model, journal: Journal, *, turns: int) -> None:
    """Build independent conversations for one standalone attempt.

    Args:
        base (str): Original text.
        model (Model): Shared Target and assessment resource.
        attacker (Model): Selected Attacker resource.
        journal (Journal): Reserved attempt destination.
        turns (int): Maximum Target exchanges.
    """
    _ = FITDAttack(
        PlainBase(base),
        ModelChat(model),
        RecordedJudge(
            QwenJudge(
                base,
                ModelChat(model)
            ),
            journal, base
        ),
        journal,
        probe=FITDProbe(
            base,
            ContinuingChat(
                ModelChat(attacker)
            ),
            journal, turns
        )
    ).run()


if __name__ == "__main__":
    qwen_main()
