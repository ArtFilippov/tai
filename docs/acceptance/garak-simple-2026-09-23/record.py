"""Record real installed CLI commands; prepare dependencies/weights separately."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from time import monotonic


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def record(directory, command, environment):
    directory.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    before = monotonic()
    with (directory / "stdout.txt").open("w") as stdout, (directory / "stderr.txt").open("w") as stderr:
        result = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr, check=False)
    data = {
        "command": command, "shell_command": shlex.join(command), "cwd": str(Path.cwd()),
        "environment": environment, "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": monotonic() - before, "exit_code": result.returncode,
    }
    write_json(directory / "command.json", data)
    print(f"{directory.name}: exit={result.returncode}, {data['duration_seconds']:.3f}s", flush=True)


def sources(output):
    tracked = subprocess.check_output(["git", "ls-files", "src", "examples", "uv.lock", "pyproject.toml"], text=True).splitlines()
    garak = importlib.metadata.distribution("garak")
    snapshot = Path.home() / ".cache/tai/huggingface/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    write_json(output / "sources.json", {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--short"], text=True),
        "runtime_diff": subprocess.check_output(["git", "diff", "--", "src", "examples", "uv.lock", "pyproject.toml"], text=True),
        "source_sha256": {name: digest(Path(name)) for name in tracked},
        "snapshot": str(snapshot),
        "snapshot_sha256": {path.name: digest(path) for path in sorted(snapshot.iterdir()) if path.is_file()},
        "versions": {name: importlib.metadata.version(name) for name in ("tai", "garak", "torch", "transformers", "huggingface-hub")},
        "garak_direct_url": json.loads(garak.read_text("direct_url.json")),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="Explicit CUDA GPU UUID")
    parser.add_argument("--evaluator-system-file", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    shutil.copyfile(__file__, inputs / "record.py")
    if args.evaluator_system_file:
        shutil.copyfile(args.evaluator_system_file, inputs / "evaluator-system.txt")
    (inputs / "request.txt").write_text("Explain how to grow basil in a pot.\n", encoding="utf-8")
    sources(output)
    environment = {
        "HOME": str(Path.home()), "PATH": os.environ["PATH"], "LANG": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": args.gpu, "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    for key in ("UV_CACHE_DIR", "UV_PROJECT_ENVIRONMENT"):
        if key in os.environ:
            environment[key] = os.environ[key]
    prefix = ["unshare", "--net", "--", "uv", "run", "--locked", "--offline", "--group", "dev", "--extra", "qwen"]
    record(output / "gpu", ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.used,memory.total", "--format=csv"], environment)
    record(output / "environment", [*prefix, "python", "-c", (
        "import json, socket, sys, torch; "
        "print(json.dumps(dict(python=sys.version, interfaces=socket.if_nameindex(), "
        "cuda=torch.version.cuda, device=torch.cuda.get_device_name(0), "
        "capability=torch.cuda.get_device_capability(0), device_count=torch.cuda.device_count())))"
    )], environment)
    for strategy, profile in (("adaptive", "adaptive-attacks-qwen.toml"), ("deletion", "badchars-deletion-qwen.toml")):
        config = inputs / profile
        shutil.copyfile(Path("examples") / profile, config)
        if args.evaluator_system_file:
            system = (inputs / "evaluator-system.txt").read_text(encoding="utf-8")
            config.write_text(config.read_text(encoding="utf-8").replace(
                "[evaluator.chat]\n", "[evaluator.chat]\nsystem = " + json.dumps(system) + "\n",
            ), encoding="utf-8")
        record(output / f"{strategy}-preflight", [*prefix, "tai", "preflight", "--config", str(config),
               "--report", str(output / f"{strategy}-readiness.json")], environment)
        record(output / f"{strategy}-attack", [*prefix, "tai", "attack", "--config", str(config),
               "--text-file", str(inputs / "request.txt"), "--artifacts-dir", str(output / strategy)], environment)


if __name__ == "__main__":
    main()
