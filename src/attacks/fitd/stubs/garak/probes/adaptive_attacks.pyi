"""The exact resource/template surface used from the pinned garak revision."""

from os import PathLike

ADAPTIVE_TEMPLATE: str
SUFFIXES_PATH: PathLike[str]
def _load_lines(path: PathLike[str]) -> list[str]: ...
