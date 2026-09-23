"""Adaptive Attacks profiles and installed CLI against controlled HTTP peers."""

import subprocess
import sys
from pathlib import Path

from app_test_support import HttpPeer


def adaptive_profile(endpoint: HttpPeer, root: Path) -> Path:
    path = root / "adaptive.toml"
    path.write_text(f'''
[attack]
type = "adaptive_attacks"
[target]
base_url = "{endpoint.url}/target"
model = "controlled"
system = "Target system"
opencode_session = true
[evaluator]
type = "model"
rubric_version = "alert-safety-v1"
[evaluator.chat]
base_url = "{endpoint.url}/evaluator"
model = "controlled"
system = "Evaluator system"
opencode_session = true
''', encoding="utf-8")
    return path


def cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(Path(sys.executable).with_name("tai")), *arguments],
                          capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
