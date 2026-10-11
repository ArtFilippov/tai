"""Complete cancellation outcome that retains uncommitted evidence."""

from typing import override

from core.primitives.messages import Message

from ..chat_cancelled import ChatCancelled


class Cancelled(ChatCancelled):
    """Represent one interrupted exchange with its observed response."""

    @override
    def observed(self) -> Message | None:
        """Return the uncommitted response, if one was observed.

        Returns:
            Message | None: The observed response, if any.
        """
        return self.response
