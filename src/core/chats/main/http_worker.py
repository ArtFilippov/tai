"""Assemble the disposable transport process without any implicit credentials."""

import sys

from core.chats.http_worker import HTTPWorker
from core.chats.no_redirect import NoRedirect


def main() -> None:
    HTTPWorker(NoRedirect(), sys.stderr).run()


if __name__ == "__main__":
    main()
