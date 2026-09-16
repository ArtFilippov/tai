"""Issue #17 command recorder; run with the prepared Qwen environment's Python.

No model substitutions/retries. Each invocation records one named command with
unbounded subprocess wait; a detached supervisor can survive harness timeouts.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def now():
    return datetime.now(UTC).isoformat()


def capture(argv, root):
    result = subprocess.run(argv, cwd=root, capture_output=True, text=True)
    return {
        "argv": argv,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "preflight", "series", "rebuild"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    root, evidence = args.root.resolve(), args.evidence.resolve()
    evidence.mkdir(exist_ok=True)
    python = Path(sys.executable)
    env = {
        "HOME": "/root",
        "PATH": f"{python.parent}:/usr/bin:/bin",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    if args.phase == "prepare":
        for source, name in [
            ("examples/fitd-alert-qwen.toml", "profile.toml"),
            ("examples/alert-manifest.json", "manifest.json"),
            ("ALERT_taxonomy.md", "ALERT_taxonomy.md"),
        ]:
            shutil.copyfile(root / source, evidence / name)
        git_files = (
            subprocess.check_output(
                ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root
            )
            .decode()
            .split("\0")
        )
        files = sorted(
            {p for p in git_files if p and not p.startswith("docs/acceptance/")}
        )
        hashes = {p: digest(root / p) for p in files}
        with tarfile.open(evidence / "source.tar.gz", "w:gz") as archive:
            for name in files:
                archive.add(root / name, arcname=name, recursive=False)
        patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
        (evidence / "dirty-tree.patch").write_bytes(patch)
        write(
            evidence / "source.json",
            {
                "captured_at": now(),
                "root": str(root),
                "git_base": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root
                )
                .decode()
                .strip(),
                "git_status": capture(["git", "status", "--short"], root),
                "description": (
                    "Dirty working tree, NOT the HEAD tree. Source archive includes "
                    "tracked and untracked non-acceptance files."
                ),
                "sha256": hashes,
                "archive_sha256": digest(evidence / "source.tar.gz"),
                "source_digest": hashlib.sha256(
                    json.dumps(hashes, sort_keys=True).encode()
                ).hexdigest(),
            },
        )
        import garak
        import torch
        import transformers

        import tai

        snapshot = (
            Path.home()
            / ".cache/tai/huggingface/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
            / "7ae557604adf67be50417f59c2c2f167def9a775"
        )
        write(
            evidence / "model-files.json",
            {
                p.name: {
                    "path": str(p),
                    "resolved": str(p.resolve()),
                    "bytes": p.stat().st_size,
                    "sha256": digest(p),
                }
                for p in sorted(snapshot.iterdir())
                if p.is_file()
            },
        )
        source = Path(
            "/29a_fil/.opencode/tmp/interactive/opencode/tai-issue15-alert.jsonl"
        )
        shutil.copyfile(source, evidence / "alert-source.jsonl")
        write(
            evidence / "environment.json",
            {
                "captured_at": now(),
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "libc": platform.libc_ver(),
                "tai_path": tai.__file__,
                "garak_path": garak.__file__,
                "transformers_path": transformers.__file__,
                "packages": dict(
                    sorted(
                        (d.metadata["Name"], d.version)
                        for d in metadata.distributions()
                    )
                ),
                "garak_direct_url": json.loads(
                    metadata.distribution("garak").read_text("direct_url.json")
                ),
                "garak_source_sha256": {
                    str(p.relative_to(Path(garak.__file__).parent)): digest(p)
                    for p in sorted(Path(garak.__file__).parent.rglob("*.py"))
                },
                "torch_cuda": torch.version.cuda,
                "cuda_device": {
                    "index": 0,
                    "name": torch.cuda.get_device_name(0),
                    "capability": torch.cuda.get_device_capability(0),
                },
                "nvidia_smi": capture(["nvidia-smi"], root),
                "gpu_identity": capture(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,uuid,driver_version,memory.total",
                        "--format=csv",
                    ],
                    root,
                ),
                "uv": capture(["uv", "--version"], root),
                "prepared_dataset": {
                    "path": str(source),
                    "bytes": source.stat().st_size,
                    "sha256": digest(source),
                },
                "runtime_environment": env,
                "network_namespace_probe": capture(
                    [
                        "unshare",
                        "--net",
                        "--",
                        "/usr/bin/readlink",
                        "/proc/self/ns/net",
                    ],
                    root,
                ),
                "parent_network_namespace": os.readlink("/proc/self/ns/net"),
            },
        )
        return
    tai = str(python.parent / "tai")
    if args.phase == "preflight":
        command = [
            tai,
            "preflight",
            "--config",
            str(evidence / "profile.toml"),
            "--report",
            str(evidence / "preflight.json"),
        ]
    elif args.phase == "series":
        command = [
            tai,
            "alert-series",
            "--manifest",
            str(evidence / "manifest.json"),
            "--profile",
            str(evidence / "profile.toml"),
            "--taxonomy",
            str(evidence / "ALERT_taxonomy.md"),
            "--artifacts-dir",
            str(evidence / "series"),
        ]
    else:
        (index,) = (evidence / "series").glob("*/index.json")
        command = [
            tai,
            "alert-report",
            "--index",
            str(index),
            "--output",
            str(evidence / "report-rebuilt.json"),
        ]
    command = ["/usr/bin/unshare", "--net", "--", *command]
    record_path = evidence / f"{args.phase}-command.json"
    if record_path.exists():
        raise RuntimeError(
            "Refusing to overwrite command evidence; use a separate evidence directory"
        )
    record = {"argv": command, "cwd": str(root), "env": env, "started_at": now()}
    write(record_path, record)
    start = time.monotonic()
    with (
        (evidence / f"{args.phase}.stdout").open("wb") as stdout,
        (evidence / f"{args.phase}.stderr").open("wb") as stderr,
    ):
        child = subprocess.Popen(
            command, cwd=root, env=env, stdout=stdout, stderr=stderr
        )
        record["pid"] = child.pid
        write(record_path, record)
        code = child.wait()  # Deliberately no timeout, including for FAILED attempts.
    record.update(
        exit_code=code, finished_at=now(), duration_seconds=time.monotonic() - start
    )
    write(record_path, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
