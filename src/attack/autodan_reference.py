"""Read native garak's bundled reference population at run time."""

from pathlib import Path
from typing import TYPE_CHECKING, override

from garak.data import path as data_path
import torch.serialization as native

from attack.autodan_corpus import AutoDANCorpus


class AutoDANReference(AutoDANCorpus):
    """Load pinned references without writing or modifying the resource file."""

    @override
    def population(self, batch: int) -> list[str]:
        """Apply native Qwen name replacements and the prompt delimiter.

        Args:
            batch (int): Number of bundled prefixes to use.

        Returns:
            list[str]: Initial reference population.

        Raises:
            ValueError: If the bundled corpus is too small.
        """
        if TYPE_CHECKING:
            def load_reference(path: Path, *, map_location: str, weights_only: bool) -> list[str]:
                """Type the pinned native tensor load without runtime changes.

                Args:
                    path (Path): Bundled tensor path.
                    map_location (str): CPU location.
                    weights_only (bool): Safe tensor unpickling flag.

                Returns:
                    list[str]: Bundled native prompts.

                Raises:
                    AssertionError: This branch is for static checking only.
                """
                raise AssertionError(
                    (path, map_location, weights_only)
                )
        else:
            load_reference = native.load
        reference = load_reference(
            data_path / "autodan" / "prompt_group.pth",
            map_location="cpu", weights_only=True
        )
        if len(reference) < batch:
            raise ValueError("AutoDAN batch exceeds pinned reference corpus")
        return [
            text.replace("ChatGPT", "Qwen/Qwen2.5-0.5B-Instruct")
            .replace("chatGPT", "Qwen/Qwen2.5-0.5B-Instruct")
            .replace("ModelKeeper", "Moderation") + " [PROMPT]:"
            for text in reference[:batch]
        ]

    @override
    def description(self) -> str:
        return "garak.data/autodan/prompt_group.pth"
