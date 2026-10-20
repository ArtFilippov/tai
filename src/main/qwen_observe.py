"""Observe candidate changes in a pinned Qwen FITD run without disk events."""

from argparse import ArgumentParser, Namespace

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from attack.fitd_attack import FITDAttack
from attack.fitd_probe import FITDProbe
from journal.memory_journal import MemoryJournal
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from target.continuing_chat import ContinuingChat
from target.qwen_chat import QwenChat
from target.qwen_model import QwenModel


def main() -> None:
    """Run an offline experiment on a chosen GPU and print its observation."""
    parser = ArgumentParser(description="Observe Qwen FITD candidate changes")
    _ = parser.add_argument("base")
    _ = parser.add_argument(
        "--gpu", type=int, required=True
    )
    _ = parser.add_argument(
        "--turns", type=int, default=2
    )
    _ = parser.add_argument(
        "--tokens", type=int, default=192
    )
    launch(
        parser.parse_args()
    )


def launch(arguments: Namespace) -> None:
    """Assemble isolated roles and observe one run with local weights.

    Args:
        arguments (Namespace): Base request and inference limits.

    Raises:
        ValueError: If the GPU or target exchange limit is invalid.
    """
    if arguments.turns < 1:
        raise ValueError("FITD round limit must be positive")
    if arguments.gpu < 0 or arguments.gpu >= torch.cuda.device_count():
        raise ValueError(f"CUDA GPU {arguments.gpu} is unavailable")
    path = snapshot_download(
        "Qwen/Qwen2.5-0.5B-Instruct",
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        local_files_only=True
    )
    weights = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.float16
    )
    device = torch.device(
        "cuda", arguments.gpu
    )
    _ = weights.to(device)
    _ = weights.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True
    )
    model = QwenModel(
        device, arguments.tokens, tokenizer, weights
    )
    journal = MemoryJournal()
    target = QwenChat(model)
    judge = RecordedJudge(
        QwenJudge(
            arguments.base,
            QwenChat(model)
        ),
        journal, arguments.base
    )
    probe = FITDProbe(
        arguments.base, ContinuingChat(
            QwenChat(model)
        ), journal, arguments.turns
    )
    _ = FITDAttack(
        target, judge, journal, probe
    ).run(arguments.base)
    print(
        journal.report()
    )
