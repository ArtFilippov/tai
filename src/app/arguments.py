"""Command-line syntax, independent of experiment composition."""

import argparse
from collections.abc import Callable
from pathlib import Path

from alert.records import DEFAULT_SEED


class CommandLine:
    def __init__(self) -> None:
        super().__init__()
        self._parser = argparse.ArgumentParser(prog="tai", description="Эксперименты с LLM")
        self._commands = self._parser.add_subparsers(dest="command", required=True)

    def add_command(
        self, name: str, configure: Callable[[argparse.ArgumentParser], None], *,
        help_text: str, description: str | None = None,
    ) -> None:
        configure(self._commands.add_parser(name, help=help_text, description=description))

    def parse(self, argv: list[str] | None) -> argparse.Namespace:
        return self._parser.parse_args(argv)


def _fitd(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("--config", type=Path, required=True, help="TOML-конфигурация")
    _ = parser.add_argument("--artifacts-dir", type=Path, required=True, help="Каталог артефактов запусков")
    source = parser.add_mutually_exclusive_group(required=True)
    _ = source.add_argument("--text", help="Базовый текст непосредственно")
    _ = source.add_argument("--text-file", type=Path, help="Файл базового текста в UTF-8")


def _preflight(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("--config", type=Path, required=True, help="TOML-конфигурация")
    _ = parser.add_argument("--report", type=Path, help="Сохранить JSON-диагностику")


def _prepare(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("--source-file", type=Path, required=True, help="Локальный alert.jsonl")
    _ = parser.add_argument("--download", action="store_true", help="Скачать закреплённый источник в --source-file")
    _ = parser.add_argument("--output", type=Path, required=True, help="Выходной JSON-манифест")
    _ = parser.add_argument("--taxonomy", type=Path, default=Path("ALERT_taxonomy.md"))
    _ = parser.add_argument("--seed", type=int, default=DEFAULT_SEED)


def _series(parser: argparse.ArgumentParser) -> None:
    for name in ("manifest", "profile", "artifacts-dir"):
        _ = parser.add_argument(f"--{name}", type=Path, required=True)
    _ = parser.add_argument("--taxonomy", type=Path, default=Path("ALERT_taxonomy.md"))


def _report(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("--index", type=Path, required=True)
    _ = parser.add_argument("--output", type=Path)


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    commands = CommandLine()
    commands.add_command("fitd", _fitd, help_text="Запустить FITD через локальный Qwen или OpenAI-совместимый API",
                         description="Один базовый запрос, generations=1. max_calls_per_conv (по умолчанию 10) "
                         + "— библиотечный лимит раундов FITD, а не общий бюджет обращений. "
                         + "HTTP-таймаут каждой роли по умолчанию 120 секунд.")
    commands.add_command("preflight", _preflight, help_text="Проверить текстовый ответ каждой роли без запуска FITD")
    commands.add_command("prepare-alert", _prepare, help_text="Подготовить локальный манифест 32 запросов ALERT")
    commands.add_command("alert-series", _series, help_text="Последовательно выполнить 32 запуска FITD по ALERT")
    commands.add_command("alert-report", _report, help_text="Пересчитать JSON по сохранённому индексу без инференса")
    return commands.parse(argv)
