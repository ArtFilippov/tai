"""Prevent urllib from forwarding credentials on redirected requests."""

from http.client import HTTPMessage
from typing import IO, NoReturn, override
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request


class NoRedirect(HTTPRedirectHandler):
    """Report the original redirect status instead of following its URL."""

    @override
    def redirect_request(
        self, req: Request, fp: IO[bytes], code: int, msg: str,
        *response: str | HTTPMessage, **metadata: str | HTTPMessage,
    ) -> NoReturn:
        headers = response[0] if response else metadata.get("headers")
        if isinstance(headers, HTTPMessage):
            raise HTTPError(req.full_url, code, msg, headers, fp)
        raise ValueError("invalid HTTP redirect response")
