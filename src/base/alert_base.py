"""One original ALERT base request."""

from collections.abc import Mapping
from typing import override

from base.base_request import BaseRequest


class AlertBase(BaseRequest):
    """An unmodified source prompt identified by its category and source ID."""

    def __init__(
        self, identity: int, prompt: str, microcategory: str,
        origin: Mapping[str, str],
    ) -> None:
        """Preserve the original prompt and its pinned provenance.

        Args:
            identity (int): Source ID.
            prompt (str): Source text.
            microcategory (str): Canonical label.
            origin (Mapping[str, str]): Original label, group and provenance.
        """
        super().__init__()
        self.identity = identity
        self.prompt = prompt
        self.microcategory = microcategory
        self.origin = origin

    @property
    def macro(self) -> str:
        """Name the macro category of this request.

        Returns:
            str: Canonical macro label.
        """
        return self.origin["macro"]

    @override
    def text(self) -> str:
        return self.prompt

    @override
    def metadata(self) -> Mapping[str, str | int]:
        return {
            "id": self.identity,
            "source_category": self.origin["source_category"],
            "microcategory": self.microcategory,
            "macro": self.macro,
            "source_revision": self.origin["source_revision"],
            "source_sha256": self.origin["source_sha256"],
            "taxonomy_version": self.origin["taxonomy_version"],
            "taxonomy_sha256": self.origin["taxonomy_sha256"],
        }
