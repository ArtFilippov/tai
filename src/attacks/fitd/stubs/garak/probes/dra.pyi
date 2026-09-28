"""Native DRA generation surface without the corpus-loading constructor."""

from .base import Probe

class DRA(Probe):
    moderation_cache: dict[str, bool]
    letter_map: dict[str, list[str]]
    all_templates: list[str]
    all_behaviors: list[str]
    toxic_trunc: float
    benign_trunc: float
    benign_full_token_prob: float
    follow_prompt_cap: bool
    soft_probe_prompt_cap: int
    def __init__(self, config_root: object = ...) -> None: ...
    def _prefix_gen(self, question: str) -> tuple[str, str]: ...
    def _suffix_gen(self, question: str, template: str) -> str: ...
    def _generate_prompts(self) -> list[str]: ...
