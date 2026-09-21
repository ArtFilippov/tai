"""The initial FITD profile shared by execution, observation, and artifacts."""

from dataclasses import dataclass

GARAK_REVISION = "3f50ea5ff9cd7050099940647c15c39b07a93392"


@dataclass(frozen=True, slots=True)
class FITDProfile:
    generations: int = 1
    max_calls_per_conv: int = 10
    num_level_queries: int = 3


FITD_PROFILE = FITDProfile()


@dataclass(frozen=True)
class EnglishLanguage:
    """The supported English profile needs no translation service or models.

    Attributes:
        target_lang (str, default='en'): The language of the pinned profile.
    """

    target_lang: str = "en"

    @staticmethod
    def get_text(prompts: list[str]) -> list[str]:
        return prompts
