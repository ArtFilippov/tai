"""Secret-safe failures from the Chat Completions transport."""


class OpenAIError(RuntimeError):
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
        self.diagnostic = (
            {
                "status_code": status_code,
                "response_bytes": response_bytes,
                "response_fragment": response_fragment,
            }
            if response_fragment is not None else None
        )
        detail = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"Chat Completions {reason}{detail}; response_bytes={response_bytes}")


class OpenAIHTTPError(OpenAIError):
    """A non-success HTTP status; redirects are not followed."""


class OpenAIAuthenticationError(OpenAIHTTPError):
    """The endpoint rejected authentication or access (401/403)."""


class OpenAIRateLimitError(OpenAIHTTPError):
    """The endpoint limited the request (429); no implicit retry is made."""


class OpenAITransportError(OpenAIError):
    """Connection, TLS, HTTP framing, or worker failure."""


class OpenAITimeoutError(OpenAITransportError, TimeoutError):
    """The configured whole-exchange deadline expired."""


class OpenAIResponseError(OpenAIError):
    """A successful HTTP response did not contain a usable text completion."""


class OpenAIJSONError(OpenAIResponseError):
    """The response was not valid JSON text."""


class OpenAIContentError(OpenAIResponseError):
    """The response did not contain exactly one assistant text completion."""
