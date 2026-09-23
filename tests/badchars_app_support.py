"""Controlled HTTP profile for the real native deletion strategy."""

from pathlib import Path

from adaptive_app_support import adaptive_profile
from app_test_support import HttpPeer


def badchars_profile(endpoint: HttpPeer, root: Path, parameters: str = "") -> Path:
    path = adaptive_profile(endpoint, root)
    path.write_text(path.read_text(encoding="utf-8").replace(
        'type = "adaptive_attacks"', f'type = "badchars_deletion"\n{parameters}',
    ), encoding="utf-8")
    return path
