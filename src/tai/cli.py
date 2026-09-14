"""The tai command: configuration and input errors never start an experiment."""

import argparse
import sys
from pathlib import Path

from tai import Message
from tai._config import load_user
from tai.terminal import run_foreground


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tai", description="Эксперименты с LLM")
    commands = parser.add_subparsers(dest="command", required=True)
    fitd = commands.add_parser(
        "fitd",
        help="Запустить FITD через прямой OpenAI-совместимый API",
        description=(
            "Один базовый запрос, generations=1. max_calls_per_conv (по умолчанию 10) "
            "— библиотечный лимит раундов FITD, а не общий бюджет обращений. "
            "HTTP-таймаут каждой роли по умолчанию 120 секунд."
        ),
    )
    fitd.add_argument("--config", type=Path, required=True, help="TOML-конфигурация")
    fitd.add_argument(
        "--artifacts-dir", type=Path, required=True, help="Каталог артефактов запусков"
    )
    source = fitd.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Базовый текст непосредственно")
    source.add_argument("--text-file", type=Path, help="Файл базового текста в UTF-8")
    args = parser.parse_args(argv)
    try:
        text = (
            args.text_file.read_bytes().decode("utf-8")
            if args.text_file is not None
            else args.text
        )
        user = load_user(args.config, Message(text), args.artifacts_dir)
    except UnicodeError:
        print("Ошибка: входной файл должен содержать корректный UTF-8", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"Ошибка конфигурации или чтения: {exc}", file=sys.stderr)
        return 2
    return run_foreground(user, artifacts_dir=args.artifacts_dir)
