"""The initial FITD profile shared by execution, observation, and artifacts."""

from dataclasses import asdict, dataclass

from attacks.shared.contracts import StrategyConfiguration


@dataclass(frozen=True, slots=True)
class FITDProfile:
    generations: int = 1
    max_calls_per_conv: int = 10
    num_level_queries: int = 3

    def configuration(self) -> StrategyConfiguration:
        return StrategyConfiguration("fitd", asdict(self), self.max_calls_per_conv)


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
