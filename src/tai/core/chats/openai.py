"""Direct, synchronous OpenAI-compatible Chat Completions conversations."""

import json
import math
import subprocess
import sys
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

from tai.core.primitives.diagnostics import DiagnosticObject, DiagnosticRedactor

from .conversation import ChatCancelled, ChatTurn, LLMChat


class OpenAIError(RuntimeError):
    """Secret-safe failure diagnostics, suitable for a run's journal."""

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
            if response_fragment is not None
            else None
        )
        detail = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(
            f"Chat Completions {reason}{detail}; response_bytes={response_bytes}"
        )


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


class OpenAIChat(LLMChat):
    """One independently configured role and its completed conversation.

    ``base_url`` is the API root (for example, ``https://host/v1``).
    Authorization is explicit: ``api_key`` becomes a Bearer header, while
    ``authorization`` is a complete Authorization header value. No credentials,
    proxies, model configuration, or agent session are discovered implicitly.
    ``timeout`` bounds the entire exchange, including connection and body reads.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        authorization: str | None = None,
        system: str = "",
        generation: Mapping[str, Any] | None = None,
        timeout: float = 120.0,
        opencode_session: bool = False,
    ) -> None:
        try:
            url = urlsplit(base_url)
            valid_url = (
                url.scheme in ("http", "https")
                and bool(url.hostname)
                and url.username is None
                and url.password is None
                and not url.query
                and not url.fragment
            )
            _ = url.port
        except ValueError:
            valid_url = False
        if not valid_url or any(character.isspace() for character in base_url):
            raise ValueError(
                "base_url must be an HTTP(S) API root "
                "without credentials, query or fragment"
            )
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if not isinstance(system, str):
            raise ValueError("system must be a string")
        if not isinstance(opencode_session, bool):
            raise ValueError("opencode_session must be a boolean")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if api_key is not None and authorization is not None:
            raise ValueError("configure api_key or authorization, not both")
        for credential in (api_key, authorization):
            if credential is not None and (
                not isinstance(credential, str)
                or not credential.strip()
                or not credential.isascii()
                or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in credential
                )
            ):
                raise ValueError(
                    "authorization must be a nonempty single-line ASCII value"
                )
        parameters = dict(generation or {})
        if {"model", "messages"} & parameters.keys():
            raise ValueError("generation cannot override model or messages")
        if parameters.get("n", 1) != 1 or parameters.get("stream", False) is not False:
            raise ValueError("only n=1 and stream=False are supported")
        try:
            self._generation_json = json.dumps(
                {**parameters, "n": 1, "stream": False}, allow_nan=False
            )
        except (TypeError, ValueError):
            raise ValueError("generation must contain finite JSON values") from None
        super().__init__(lambda turns: "", model=model, system=system)
        self._base_url = base_url.rstrip("/")
        self._authorization = (
            f"Bearer {api_key}" if api_key is not None else authorization
        )
        self._redactor = DiagnosticRedactor(
            value
            for value in (
                api_key,
                authorization,
                authorization.split(" ", 1)[-1] if authorization else None,
            )
            if value
        )
        self._authorization_mode = (
            "api_key"
            if api_key is not None
            else "explicit"
            if authorization is not None
            else "none"
        )
        self._timeout = timeout
        self._opencode_session = opencode_session

    def settings(self) -> DiagnosticObject:
        return self._diagnostic_settings(self._redactor)

    def _diagnostic_settings(self, redactor: DiagnosticRedactor) -> DiagnosticObject:
        # A run supplies both roles' credentials before any replacement, so
        # overlapping credentials and resulting key collisions share one policy.
        settings = {
            **super()._settings(),
            "backend": "openai",
            "transport": "openai_chat_completions",
            "base_url": self._base_url,
            "authorization": self._authorization_mode,
            "timeout": self._timeout,
            "opencode_session": self._opencode_session,
            "generation": redactor.snapshot(
                json.loads(self._generation_json), mask_fields=True
            ),
        }
        return redactor.snapshot(settings)

    def _generate(self, turns: tuple[ChatTurn, ...], cancellation: Event) -> str:
        if cancellation.is_set():
            raise ChatCancelled()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"tai/{version('tai')}",
        }
        if self._authorization is not None:
            headers["Authorization"] = self._authorization
        if self._opencode_session:
            # send owns the exchange lock, so reset cannot rotate this identity
            # until the entire HTTP exchange (including cleanup) has finished.
            headers["x-opencode-session"] = self.conversation_id
        request = json.dumps(
            {
                "url": self._base_url + "/chat/completions",
                "headers": headers,
                "timeout": self._timeout,
                "body": {
                    **json.loads(self._generation_json),
                    "model": self.model,
                    "messages": [
                        {"role": turn.role, "content": turn.message.to_string()}
                        for turn in turns
                    ],
                    "stream": False,
                    "n": 1,
                },
            }
        ).encode()
        deadline = monotonic() + self._timeout
        try:
            worker = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("http_worker.py"))],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # Terminal Ctrl+C belongs to User.stop(), not to the transport
                # child: killing it first would turn cancellation into FAILED.
                start_new_session=True,
            )
        except OSError:
            raise OpenAITransportError("could not start HTTP worker") from None
        completed = Event()
        output = (b"", b"")
        communication_error: BaseException | None = None

        def communicate() -> None:
            nonlocal output, communication_error
            try:
                # One call owns all pending stdin writes and drains both pipes.
                # Resuming timed-out communicate(None) can strand large input.
                output = worker.communicate(request)
            except BaseException as exc:
                communication_error = exc
            finally:
                completed.set()

        io = Thread(target=communicate, name="tai-http-io")
        interrupted = None
        with worker:
            try:
                io.start()
                while not completed.is_set():
                    if cancellation.is_set():
                        interrupted = "cancel"
                        break
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        interrupted = "timeout"
                        break
                    completed.wait(min(0.02, remaining))
            finally:
                if worker.poll() is None:
                    worker.kill()
                try:
                    # Killing the sole child releases blocked stdin writes and
                    # closes its output pipes. Join the drain before returning,
                    # including cancellation and communication-error paths.
                    if io.ident is not None:
                        io.join()
                finally:
                    worker.wait()
        raw, reports = output
        response: dict[str, Any] = {}
        for line in reports.splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                # A kill can interrupt a status write. Never copy stderr text
                # (including startup tracebacks) into diagnostics.
                continue
            if isinstance(record, dict):
                response.update(record)
        status = response.get("status")
        failure = response.get("error")
        if communication_error is not None:
            failure = "worker communication failed"
        if worker.returncode or not response.get("complete") or status is None:
            failure = failure or "worker failed"
        if interrupted is not None or failure is not None:
            details: dict[str, Any] = {
                "status_code": status,
                "response_bytes": len(raw),
                "response_fragment": self._redactor.text(
                    raw.decode("utf-8", errors="replace")
                )
                if raw
                else None,
            }
            if interrupted == "cancel":
                raise ChatCancelled(diagnostic=details if raw else None)
            if interrupted == "timeout" or failure == "TimeoutError":
                raise OpenAITimeoutError("request timed out", **details)
            raise OpenAITransportError(f"transport failed: {failure}", **details)
        assert status is not None
        if not 200 <= status < 300:
            error_type = (
                OpenAIAuthenticationError
                if status in (401, 403)
                else OpenAIRateLimitError
                if status == 429
                else OpenAIHTTPError
            )
            raise error_type("HTTP request rejected", status_code=status)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError):
            raise OpenAIJSONError(
                "invalid JSON", status_code=status, response_bytes=len(raw)
            ) from None
        try:
            choices = body["choices"]
            if not isinstance(choices, list) or len(choices) != 1 or "error" in body:
                raise ValueError
            message = choices[0]["message"]
            content = message["content"]
            if message["role"] != "assistant" or not isinstance(content, str):
                raise ValueError
        except (KeyError, IndexError, TypeError, ValueError):
            raise OpenAIContentError(
                "expected one assistant text completion",
                status_code=status,
                response_bytes=len(raw),
            ) from None
        return content
