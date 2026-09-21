"""One disposable HTTP worker; invoked by filename, with input on stdin.

Keeping blocking DNS/TLS/socket operations in this process lets the caller stop
and reap *all* local I/O, rather than abandon an uninterruptible worker thread.
Only the standard library is loaded here; no provider SDK or agent is involved.

stdout carries received body bytes, flushed before the next socket read. stderr
carries only local JSON status records. Killing the worker can thus preserve the
observed body prefix without mistaking it for a complete HTTP response.
"""

import json
import sys
from functools import partial
from http.client import HTTPException, HTTPMessage, HTTPResponse, IncompleteRead
from typing import IO, NoReturn, TypedDict, Unpack, cast, override
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, OpenerDirector, ProxyHandler, Request, build_opener


class _RedirectMetadata(TypedDict, total=False):
    msg: str
    headers: HTTPMessage


def _redirect_values(
    response: tuple[str | HTTPMessage, ...], metadata: _RedirectMetadata,
) -> tuple[str, HTTPMessage]:
    if response:
        message = cast(str, response[0])
    else:
        message = metadata.get("msg")
        assert message is not None
    if len(response) > 1:
        headers = cast(HTTPMessage, response[1])
    else:
        headers = metadata.get("headers")
        assert headers is not None
    return message, headers


class _NoRedirect(HTTPRedirectHandler):
    # Apply urllib's ordinary HTTP failure policy to every redirect status. This
    # rejects redirects before a new Request can forward configured authorization.
    @override
    def http_error_302(
        self, req: Request, fp: IO[bytes], code: int,
        *response: str | HTTPMessage, **metadata: Unpack[_RedirectMetadata],
    ) -> NoReturn:
        message, headers = _redirect_values(response, metadata)
        raise HTTPError(req.full_url, code, message, headers, fp)

    @override
    def http_error_301(
        self, req: Request, fp: IO[bytes], code: int,
        *response: str | HTTPMessage, **metadata: Unpack[_RedirectMetadata],
    ) -> NoReturn:
        self.http_error_302(req, fp, code, *response, **metadata)

    @override
    def http_error_303(
        self, req: Request, fp: IO[bytes], code: int,
        *response: str | HTTPMessage, **metadata: Unpack[_RedirectMetadata],
    ) -> NoReturn:
        self.http_error_302(req, fp, code, *response, **metadata)

    @override
    def http_error_307(
        self, req: Request, fp: IO[bytes], code: int,
        *response: str | HTTPMessage, **metadata: Unpack[_RedirectMetadata],
    ) -> NoReturn:
        self.http_error_302(req, fp, code, *response, **metadata)

    @override
    def http_error_308(
        self, req: Request, fp: IO[bytes], code: int,
        *response: str | HTTPMessage, **metadata: Unpack[_RedirectMetadata],
    ) -> NoReturn:
        self.http_error_302(req, fp, code, *response, **metadata)


def _report(**data: object) -> None:
    print(json.dumps(data), file=sys.stderr, flush=True)


class _WorkerRequest(TypedDict):
    url: str
    body: dict[str, object]
    headers: dict[str, str]
    timeout: float


def _stream(response: HTTPResponse) -> None:
    _report(status=response.status)
    output = sys.stdout.buffer
    # read1 emits available bytes without waiting for the entire Content-Length.
    for chunk in iter(partial(response.read1, 65536), b""):
        _ = output.write(chunk)
        _ = output.flush()
    if response.length not in (None, 0):
        raise IncompleteRead(b"")
    _report(complete=True)


def _exchange(request: _WorkerRequest, opener: OpenerDirector) -> None:
    with opener.open(
        Request(
            request["url"], data=json.dumps(request["body"]).encode(),
            headers=request["headers"], method="POST",
        ), timeout=request["timeout"],
    ) as response:
        _stream(cast(HTTPResponse, response))


def main() -> None:
    request = cast(_WorkerRequest, json.load(sys.stdin))
    try:
        _exchange(request, build_opener(ProxyHandler({}), _NoRedirect()))
    except HTTPError as exc:
        _report(status=exc.code, complete=True)
        exc.close()
    except URLError as exc:
        _report(error=type(exc.reason).__name__)
    except (OSError, HTTPException) as exc:
        # Provider bodies, URLs, headers and exception strings can echo secrets.
        # Only the local exception class goes into the status channel. Body
        # bytes already sent on stdout are redacted by the parent for diagnostics.
        _report(error=type(exc).__name__)


if __name__ == "__main__":
    main()
