"""Record the final full pytest and Quard checks and their real exit codes."""

import argparse
import os
from pathlib import Path
import shutil

from record import digest, record, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {key: os.environ[key] for key in (
        "HOME", "PATH", "LANG", "UV_CACHE_DIR", "UV_PROJECT_ENVIRONMENT",
    ) if key in os.environ}
    paths = sorted(path for folder in ("src", "tests") for path in Path(folder).rglob("*.py"))
    write_json(args.output / "checked-source-sha256.json", {str(path): digest(path) for path in paths})
    prefix = ["uv", "run", "--locked", "--group", "dev", "--extra", "qwen"]
    record(args.output / "pytest", [*prefix, "pytest"], environment)
    record(args.output / "quard", [*prefix, "python", "quard/check.py"], environment)
    shutil.copyfile(".quard/reports/latest.json", args.output / "quard-report.json")


if __name__ == "__main__":
    main()
