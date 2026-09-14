"""One disposable HTTP worker; invoked by filename, with input on stdin.

Keeping blocking DNS/TLS/socket operations in this process lets the caller stop
and reap *all* local I/O, rather than abandon an uninterruptible worker thread.
Only the standard library is loaded here; no provider SDK or agent is involved.

stdout carries received body bytes, flushed before the next socket read. stderr
carries only local JSON status records. Killing the worker can thus preserve the
observed body prefix without mistaking it for a complete HTTP response.
"""

import http.client
import json
import sys
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # In particular, never forward configured authorization to another URL.
        return None


def _report(**data: object) -> None:
    print(json.dumps(data), file=sys.stderr, flush=True)


def main() -> None:
    request = json.load(sys.stdin)
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(
            Request(
                request["url"],
                data=json.dumps(request["body"]).encode(),
                headers=request["headers"],
                method="POST",
            ),
            timeout=request["timeout"],
        ) as response:
            _report(status=response.status)
            # read1 returns available bytes instead of waiting for the entire
            # Content-Length. Keep no unread body buffer in our worker loop.
            while chunk := response.read1(65536):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            # Unlike read(), read1 does not reject an early Content-Length EOF.
            if response.length not in (None, 0):
                raise http.client.IncompleteRead(b"")
            _report(complete=True)
    except HTTPError as exc:
        _report(status=exc.code, complete=True)
        exc.close()
    except (OSError, URLError, http.client.HTTPException) as exc:
        # Provider bodies, URLs, headers and exception strings can echo secrets.
        # Only the local exception class goes into the status channel. Body
        # bytes already sent on stdout are redacted by the parent for diagnostics.
        cause = exc.reason if isinstance(exc, URLError) else exc
        _report(error=type(cause).__name__)


if __name__ == "__main__":
    main()
