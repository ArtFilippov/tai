"""Stream one isolated HTTP exchange without leaking provider details on failure."""

import json
import sys
from functools import partial
from http.client import HTTPException, HTTPResponse, IncompleteRead
from typing import IO, cast, override
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .http_worker_contract import HTTPWorkerContract


class HTTPWorker(HTTPWorkerContract):
    """Own the status channel and body stream of one disposable exchange.

    Args:
        redirect (HTTPRedirectHandler): Policy for redirected HTTP requests.
        channel (IO[str]): Local status channel, separate from response bytes.
    """

    def __init__(self, redirect: HTTPRedirectHandler, channel: IO[str]) -> None:
        super().__init__()
        self.redirect = redirect
        self.channel = channel

    def report(self, **data: object) -> None:
        print(json.dumps(data), file=self.channel, flush=True)

    def stream(self, response: HTTPResponse) -> None:
        self.report(status=response.status)
        output = sys.stdout.buffer
        for chunk in iter(partial(response.read1, 65536), b""):
            _ = output.write(chunk)
            _ = output.flush()
        if response.length not in (None, 0):
            raise IncompleteRead(b"")
        self.report(complete=True)

    def exchange(self, request: dict[str, object]) -> None:
        url, headers, timeout = request["url"], request["headers"], request["timeout"]
        if not isinstance(url, str) or not isinstance(headers, dict) or not isinstance(
            timeout, (int, float)
        ):
            raise ValueError("invalid HTTP worker request")
        with build_opener(ProxyHandler({}), self.redirect).open(
            Request(
                url, data=json.dumps(request["body"]).encode(),
                headers=cast(dict[str, str], headers), method="POST",
            ), timeout=timeout,
        ) as response:
            self.stream(cast(HTTPResponse, response))

    @override
    def run(self) -> None:
        request = cast(dict[str, object], json.load(sys.stdin))
        try:
            self.exchange(request)
        except HTTPError as exc:
            self.report(status=exc.code, complete=True)
            exc.close()
        except URLError as exc:
            self.report(error=type(exc.reason).__name__)
        except (OSError, HTTPException) as exc:
            # Exception text, provider bodies and URLs can echo credentials.
            self.report(error=type(exc).__name__)
