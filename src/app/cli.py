"""The tai command: composition after parsing; input errors never start work."""

import argparse
import json
import sys
from contextlib import closing
from pathlib import Path

from alert.preparation import prepare_alert
from alert.report import reconstruct_report
from attacks.shared.contracts import FITD, THREATEN_JSON
from core.primitives.messages import Message

from .arguments import parse_arguments
from .configuration import load_profile
from .preflight import run_preflight
from .series import run_alert_series
from .terminal import run_profile

THREATEN_COMMAND = "threaten-json"


def _series(args: argparse.Namespace) -> int:
    result = run_alert_series(args.manifest, args.profile, args.artifacts_dir, taxonomy_path=args.taxonomy,
                              jobs=args.jobs, lines_per_attack=args.lines_per_attack)
    print(f"Серия: {result.directory}", file=sys.stderr)
    if result.exit_code == 1:
        print(f"Ошибки запуска или сохранения: {result.directory / 'index.json'}", file=sys.stderr)
    print(json.dumps(result.report, ensure_ascii=False, indent=2))
    return result.exit_code


def _report(args: argparse.Namespace) -> int:
    print(json.dumps(reconstruct_report(args.index, args.output), ensure_ascii=False, indent=2))
    return 0


def _prepare(args: argparse.Namespace) -> int:
    manifest = prepare_alert(args.source_file, args.output, taxonomy_path=args.taxonomy, download=args.download,
                             sampling=(args.seed, args.per_category))
    print(f"ALERT: {len(manifest.records)} записей, seed={manifest.seed}; {args.output}")
    return 0


def _preflight(args: argparse.Namespace) -> int:
    with closing(load_profile(args.config, server_log=(args.report or Path("runs/preflight.json")).with_suffix(".vllm.log"))) as profile:
        return run_preflight(profile, args.report)


def _base_request(text: str | None, path: Path | None) -> Message:
    if path is not None:
        return Message(path.read_bytes().decode("utf-8"))
    assert text is not None
    return Message(text)


def _attack(args: argparse.Namespace) -> int:
    base = _base_request(args.text, args.text_file)
    with closing(load_profile(args.config, server_log=args.artifacts_dir / "vllm.log")) as profile:
        if args.command == FITD and profile.attack_type != FITD:
            raise ValueError("tai fitd требует attack.type=fitd; используйте tai attack")
        if args.command == THREATEN_COMMAND and profile.attack_type != THREATEN_JSON:
            raise ValueError("tai threaten-json требует attack.type=threaten_json; используйте tai attack")
        return run_profile(profile, base, args.artifacts_dir)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    commands = {
        "attack": _attack, "fitd": _attack, THREATEN_COMMAND: _attack, "preflight": _preflight,
        "prepare-alert": _prepare, "alert-series": _series, "alert-report": _report,
    }
    try:
        return commands[args.command](args)
    except UnicodeError:
        print("Ошибка: входной файл должен содержать корректный UTF-8", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"Ошибка конфигурации, чтения или записи: {exc}", file=sys.stderr)
        return 2
