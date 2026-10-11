"""Wire a concrete textual result into the installed command."""

from app.cli import main as run
from core.primitives.messages.message.text_message import TextMessage


def main() -> int:
    return run(create=TextMessage)
