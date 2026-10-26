"""Pinned GGUF file in the local Hugging Face cache."""

from dataclasses import dataclass
from pathlib import Path
from typing import override

from huggingface_hub import snapshot_download

from target.gguf_asset import GGUFAsset


@dataclass(frozen=True)
class GGUFFile(GGUFAsset):
    """Keep one repository revision and quantization together for preparation and runs.

    Attributes:
        repository (str): Pinned repository.
        revision (str): Pinned commit.
        filename (str): Exact quantization file.
    """

    repository: str
    revision: str
    filename: str

    @override
    def prepare(self) -> Path:
        """Download exactly the pinned file and return its path.

        Returns:
            Path: Prepared quantization.
        """
        return self.path(False)

    @override
    def local(self) -> Path:
        """Require the pinned file to be cached before an offline run.

        Returns:
            Path: Prepared quantization.
        """
        return self.path(True)

    @override
    def identity(self) -> str:
        """Name the model advertised by the GGUF service.

        Returns:
            str: Repository, revision and file identity.
        """
        return f"{self.repository}@{self.revision}/{self.filename}"

    @override
    def commit(self) -> str:
        return self.revision

    @override
    def details(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "file": self.filename
        }

    def path(self, offline: bool) -> Path:
        try:
            path = Path(
                       snapshot_download(
                           repo_id=self.repository, revision=self.revision, allow_patterns=self.filename, local_files_only=offline
                       )
                   ) / self.filename
        except OSError as exc:
            raise FileNotFoundError(f"Prepare {self.identity()} with 'tai-qwen prepare --attacker gguf' before running offline") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Pinned GGUF file is missing: {path}")
        return path
