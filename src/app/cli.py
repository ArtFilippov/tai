"""The tai command: configuration and input errors never start an experiment."""

import argparse
import json
import sys
from pathlib import Path

from alert.manifest import DEFAULT_SEED
from core.primitives.messages import Message

from .configuration import load_profile
from .preflight import run_preflight
from .terminal import run_foreground


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tai", description="Эксперименты с LLM")
    commands = parser.add_subparsers(dest="command", required=True)
    fitd = commands.add_parser(
        "fitd",
        help="Запустить FITD через локальный Qwen или OpenAI-совместимый API",
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
    preflight = commands.add_parser(
        "preflight", help="Проверить текстовый ответ каждой роли без запуска FITD"
    )
    preflight.add_argument(
        "--config", type=Path, required=True, help="TOML-конфигурация"
    )
    preflight.add_argument("--report", type=Path, help="Сохранить JSON-диагностику")
    prepare = commands.add_parser(
        "prepare-alert", help="Подготовить локальный манифест 32 запросов ALERT"
    )
    prepare.add_argument(
        "--source-file", type=Path, required=True, help="Локальный alert.jsonl"
    )
    prepare.add_argument(
        "--download",
        action="store_true",
        help="Скачать закреплённый источник в --source-file",
    )
    prepare.add_argument(
        "--output", type=Path, required=True, help="Выходной JSON-манифест"
    )
    prepare.add_argument("--taxonomy", type=Path, default=Path("ALERT_taxonomy.md"))
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    series = commands.add_parser(
        "alert-series", help="Последовательно выполнить 32 запуска FITD по ALERT"
    )
    series.add_argument("--manifest", type=Path, required=True)
    series.add_argument("--profile", type=Path, required=True)
    series.add_argument("--artifacts-dir", type=Path, required=True)
    series.add_argument("--taxonomy", type=Path, default=Path("ALERT_taxonomy.md"))
    report = commands.add_parser(
        "alert-report", help="Пересчитать JSON по сохранённому индексу без инференса"
    )
    report.add_argument("--index", type=Path, required=True)
    report.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "alert-series":
            from .series import run_alert_series

            result = run_alert_series(
                args.manifest,
                args.profile,
                args.artifacts_dir,
                taxonomy_path=args.taxonomy,
            )
            print(f"Серия: {result.directory}", file=sys.stderr)
            if result.exit_code == 1:
                print(
                    f"Ошибки запуска или сохранения: {result.directory / 'index.json'}",
                    file=sys.stderr,
                )
            print(json.dumps(result.report, ensure_ascii=False, indent=2))
            return result.exit_code
        if args.command == "alert-report":
            from alert.report import reconstruct_report

            print(
                json.dumps(
                    reconstruct_report(args.index, args.output),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "prepare-alert":
            from alert.preparation import prepare_alert

            manifest = prepare_alert(
                args.source_file,
                args.output,
                taxonomy_path=args.taxonomy,
                seed=args.seed,
                download=args.download,
            )
            print(
                f"ALERT: {len(manifest.records)} записей, seed={manifest.seed}; "
                f"{args.output}"
            )
            return 0
        if args.command == "preflight":
            return run_preflight(load_profile(args.config), args.report)
        text = (
            args.text_file.read_bytes().decode("utf-8")
            if args.text_file is not None
            else args.text
        )
        profile = load_profile(args.config)
        experiment = profile.create_experiment(Message(text), args.artifacts_dir)
    except UnicodeError:
        print("Ошибка: входной файл должен содержать корректный UTF-8", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"Ошибка конфигурации, чтения или записи: {exc}", file=sys.stderr)
        return 2
    return run_foreground(
        experiment, artifacts_dir=args.artifacts_dir, redact=profile.redactor().text
    )
