"""Translate detached HTTP failure observations to typed adapter exceptions."""

from http import HTTPStatus

from ..open_ai_authentication_error import OpenAIAuthenticationError
from ..open_ai_connection_error import OpenAIConnectionError
from ..open_ai_content_error import OpenAIContentError
from ..open_ai_json_error import OpenAIJSONError
from ..open_ai_http_error import OpenAIHTTPError
from ..open_ai_rate_limit_error import OpenAIRateLimitError
from ..open_ai_response_error import OpenAIResponseError
from ..open_ai_status_error import OpenAIStatusError
from ..open_ai_timeout_error import OpenAITimeoutError
from ..open_ai_transport_error import OpenAITransportError
from .cancelled import Cancelled

CANCEL = "cancel"
HTTP_ERROR = "http"

def cancelled(status: int | None, size: int, fragment: str | None) -> Exception:
    if size:
        return Cancelled(diagnostic={
            "status_code": status, "response_bytes": size,
            "response_fragment": fragment,
        })
    return Cancelled()


def http(status: int | None) -> Exception:
    errors: dict[int | None, type[OpenAIHTTPError]] = {
        HTTPStatus.UNAUTHORIZED: OpenAIAuthenticationError,
        HTTPStatus.FORBIDDEN: OpenAIAuthenticationError,
        HTTPStatus.TOO_MANY_REQUESTS: OpenAIRateLimitError,
    }
    return errors.get(status, OpenAIStatusError)("HTTP request rejected", status_code=status)


def response(kind: str, status: int | None, size: int) -> Exception:
    errors: dict[str, tuple[type[OpenAIResponseError], str]] = {
        "json": (OpenAIJSONError, "invalid JSON"),
        "content": (OpenAIContentError, "expected one assistant text completion"),
    }
    error, reason = errors[kind]
    return error(reason, status_code=status, response_bytes=size)


def transport(kind: str, status: int | None, size: int, fragment: str | None) -> Exception:
    errors: dict[str, tuple[type[OpenAITransportError], str]] = {
        "timeout": (OpenAITimeoutError, "request timed out"),
    }
    error, reason = errors.get(kind, (OpenAIConnectionError, kind))
    return error(reason, status_code=status, response_bytes=size, response_fragment=fragment)


def failure(kind: str, status: int | None, size: int, fragment: str | None) -> Exception:
    if kind in {CANCEL, HTTP_ERROR}:
        return cancelled(status, size, fragment) if kind == CANCEL else http(status)
    if kind in {"json", "content"}:
        return response(kind, status, size)
    return transport(kind, status, size, fragment)
