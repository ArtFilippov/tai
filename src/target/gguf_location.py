"""Selected bind address of a local GGUF service."""

from dataclasses import dataclass
from typing import override

from target.gguf_address import GGUFAddress


@dataclass(frozen=True)
class GGUFLocation(GGUFAddress):
    """Keep the bind and connect address together.

    Attributes:
        name (str): Selected host.
        number (int): Selected port.
    """

    name: str
    number: int

    @override
    def host(self) -> str:
        return self.name

    @override
    def port(self) -> int:
        return self.number

    @override
    def origin(self) -> str:
        return f"http://{self.name}:{self.number}"
