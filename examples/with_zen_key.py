"""Explicitly load a user's Zen JSON into the example profile's two key variables."""

import argparse
import json
import os
import sys
from pathlib import Path

from app.cli import main as tai_main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="tai arguments")
    args = parser.parse_args()
    if not args.command:
        parser.error("укажите команду tai: preflight или fitd")
    try:
        data = json.loads(args.key_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        print("Ошибка: не удалось прочитать JSON авторизации opencode", file=sys.stderr)
        return 2
    account = data.get("opencode") if isinstance(data, dict) else None
    key = account.get("key") if isinstance(account, dict) else None
    if (
        not isinstance(account, dict)
        or account.get("type") != "api"
        or not isinstance(key, str)
        or not key.strip()
        or not key.isascii()
        or any(ord(char) < 32 or ord(char) == 127 for char in key)
    ):
        print(
            "Ошибка: требуются opencode.type = api и непустой "
            "однострочный ASCII opencode.key",
            file=sys.stderr,
        )
        return 2
    # Only explicitly named variables, in this process and its children. Nothing
    # is exported to the parent shell or read from the current coding agent.
    os.environ["TAI_TARGET_API_KEY"] = key
    os.environ["TAI_ATTACKER_API_KEY"] = key
    return tai_main(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
