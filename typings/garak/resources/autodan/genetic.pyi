"""Pinned GA signature for structural local mutation generation."""

from typing import Protocol
from garak.attempt import Conversation, Message

class Mutation(Protocol):
    def generate(self, prompt: Conversation) -> list[Message]: ...

def autodan_ga(
    control_prefixes: list[str], score_list: list[float], num_elites: int,
    batch_size: int, crossover_rate: float = ..., num_points: int = ...,
    mutation: float = ..., if_softmax: bool = ...,
    mutation_generator: Mutation = ...,
) -> list[str]: ...
