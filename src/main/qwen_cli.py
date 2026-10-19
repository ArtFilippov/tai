"""Prepare pinned weights and run a local CUDA FITD experiment."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from attack.fitd_attack import FITDAttack
from attack.fitd_probe import FITDProbe
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from target.continuing_chat import ContinuingChat
from target.qwen_chat import QwenChat
from target.qwen_model import QwenModel

PREPARE = "prepare"


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
    _ = commands.add_parser(
        PREPARE, help="Download pinned model and tokenizer"
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
    if arguments.command == PREPARE:
        print(
            snapshot(False)
        )
        return
    if arguments.turns < 1:
        raise ValueError("FITD round limit must be positive")
    if arguments.journal.exists():
        raise FileExistsError(arguments.journal)
    if arguments.gpu < 0 or arguments.gpu >= torch.cuda.device_count():
        raise ValueError(f"CUDA GPU {arguments.gpu} is unavailable")
    path = snapshot(True)
    device = torch.device(
        "cuda", arguments.gpu
    )
    weights = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.float16
    )
    _ = weights.to(device)
    _ = weights.eval()
    model = QwenModel(
        device, arguments.tokens,
        AutoTokenizer.from_pretrained(
            path, local_files_only=True
        ),
        weights
    )
    arguments.journal.touch(exist_ok=False)
    journal = DiskJournal(arguments.journal)
    _ = FITDAttack(
        target=QwenChat(model),
        judge=RecordedJudge(
            QwenJudge(
                arguments.base,
                QwenChat(model)
            ),
            journal=journal, base=arguments.base
        ),
        journal=journal,
        probe=FITDProbe(
            base=arguments.base,
            attacker=ContinuingChat(
                QwenChat(model)
            ),
            journal=journal, turns=arguments.turns
        )
    ).run(arguments.base)
    print(
        ParsedJournal(arguments.journal).read()
    )
