"""One compatible token context and causal inference capability."""

from typing import Protocol

from target.token_context import TokenContext
from target.token_inference import TokenInference


class TokenSource(TokenContext, TokenInference, Protocol):
    """Keep token IDs, their template and their model together."""
