"""Secret-safe failures from the Chat Completions transport."""

from abc import ABC, abstractmethod


class OpenAIError(RuntimeError, ABC):
    """Secret-safe failure diagnostics, suitable for a run's journal.

    Args:
        reason (str): Local, credential-free failure description.
        status_code (int | None, default=None): Observed HTTP status.
        response_bytes (int, default=0): Number of observed response bytes.
        response_fragment (str | None, default=None): Redacted incomplete response.
    """

    def __init__(
        self,
        reason: str,
        *,
        status_code: int | None = None,
        response_bytes: int = 0,
        response_fragment: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.response_bytes = response_bytes
        self.diagnostic: dict[str, object] | None = (
            {
                "status_code": status_code,
                "response_bytes": response_bytes,
                "response_fragment": response_fragment,
            }
            if response_fragment is not None else None
        )
        detail = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"Chat Completions {reason}{detail}; response_bytes={response_bytes}")

    @abstractmethod
    def category(self) -> str:
        """Identify the failure kind while preserving the exception hierarchy.

        Returns:
            str: Failure category.
        """
