#!/usr/bin/env python3
"""Apply the editable readability and exclusion parameters to native configs."""
from __future__ import annotations
import argparse
import sys
from _common import error_exit, emit_event, repo_lock, require_runtime, root_dir
from _policy import configure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print one machine-readable JSON event")
    args = parser.parse_args()
    try:
        require_runtime()
        root = root_dir()
        with repo_lock(root):
            changed = configure(root)
        emit_event("applied", json_output=args.json, modified_files=changed,
                   next_command="uv run --locked --group dev python quality/check.py", exit_code=0)
        return 0
    except Exception as exc:
        return error_exit(exc, args.json)


if __name__ == "__main__":
    sys.exit(main())
