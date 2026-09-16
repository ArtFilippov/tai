"""Seal execution provenance or package the audited issue #17 materials.

Run seal before changing the captured working tree's documentation. Packaging
copies all evidence, writes per-file checksums, and checks archive member bytes.
"""

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["seal", "package"])
    parser.add_argument("evidence", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    evidence, destination = args.evidence.resolve(), args.destination.resolve()
    command = load(evidence / "series-command.json")
    assert "exit_code" in command, "Wait for the complete series"
    if args.phase == "seal":
        source = load(evidence / "source.json")
        root = Path(source["root"])
        current = {name: sha(root / name) for name in source["sha256"]}
        env = load(evidence / "environment.json")
        garak_root = Path(env["garak_path"]).parent
        garak = {name: sha(garak_root / name) for name in env["garak_source_sha256"]}
        model_files = load(evidence / "model-files.json")
        model = {name: sha(Path(info["path"])) for name, info in model_files.items()}
        check = {
            "captured_at": datetime.now(UTC).isoformat(),
            "source_unchanged": current == source["sha256"],
            "garak_unchanged": garak == env["garak_source_sha256"],
            "model_files_unchanged": model
            == {name: info["sha256"] for name, info in model_files.items()},
            "source_after_sha256": current,
            "garak_after_sha256": garak,
            "model_after_sha256": model,
        }
        write(evidence / "post-execution-check.json", check)
        assert all(
            check[key]
            for key in ("source_unchanged", "garak_unchanged", "model_files_unchanged")
        )
        (evidence / "model-config").mkdir(exist_ok=True)
        for name, info in model_files.items():
            if name != "model.safetensors":
                shutil.copyfile(info["path"], evidence / "model-config" / name)
        print("Source, garak and all model/tokenizer files unchanged after execution")
        return
    assert load(evidence / "artifact-check.json")["verified"] is True
    for name in ("record.py", "verify.py", "package.py"):
        shutil.copyfile(destination / name, evidence / name)
    files = sorted(p for p in evidence.rglob("*") if p.is_file())
    # Exclude the checksum inventory itself when packaging again.
    files = [p for p in files if p.name != "files.sha256.json"]
    inventory = {
        str(p.relative_to(evidence)): {"bytes": p.stat().st_size, "sha256": sha(p)}
        for p in files
    }
    write(evidence / "files.sha256.json", inventory)
    files.append(evidence / "files.sha256.json")
    archive_path = destination / "evidence.tar.gz"
    with tarfile.open(archive_path, "w:gz", compresslevel=9) as archive:
        for path in sorted(files):
            archive.add(path, arcname=str(path.relative_to(evidence)), recursive=False)
    with tarfile.open(archive_path) as archive:
        for member in archive:
            with archive.extractfile(member) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            assert actual == sha(evidence / member.name)
    for name in ("report-rebuilt.json", "manifest.json", "profile.toml"):
        output = "report.json" if name == "report-rebuilt.json" else name
        shutil.copyfile(evidence / name, destination / output)
    (destination / "evidence.tar.gz.sha256").write_text(
        f"{sha(archive_path)}  evidence.tar.gz\n"
    )
    write(
        destination / "checksums.json",
        {
            "publication_status": (
                "Repository-ready local files; uncommitted, no remote publication"
            ),
            "archive": {
                "file": archive_path.name,
                "bytes": archive_path.stat().st_size,
                "sha256": sha(archive_path),
                "members": len(files),
            },
            "files": {
                name: sha(destination / name)
                for name in (
                    "report.json",
                    "manifest.json",
                    "profile.toml",
                    "verify.py",
                    "record.py",
                    "package.py",
                )
            },
            "all_archive_member_bytes_verified": True,
        },
    )
    print(json.dumps(load(destination / "checksums.json"), indent=2))


if __name__ == "__main__":
    main()
