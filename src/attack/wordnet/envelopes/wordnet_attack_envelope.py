"""Bind one WordNet search to its original request and journal."""

from attack.attack_envelope import AttackEnvelope
from attack.attack_setup import AttackSetup
from attack.envelopes.qwen_attack_envelope import QwenAttackEnvelope
from attack.wordnet.wordnet_attack import WordnetAttack
from attack.wordnet.wordnet_budget import WordnetBudget
from attack.wordnet.wordnet_lexicon import WordnetLexicon
from attack.wordnet.wordnet_probe import WordnetProbe
from base.base_request import BaseRequest
from journal.journal import Journal
from target.model_chat import ModelChat


class WordnetAttackEnvelope(AttackEnvelope):
    """Compose isolated Attacker, Target and judge conversations."""

    def __init__(self, base: BaseRequest, setup: AttackSetup, journal: Journal,
                 lexicon: WordnetLexicon, *, budget: WordnetBudget) -> None:
        """Bind an explicit request and a prepared, bounded lexical search.

        Args:
            base (BaseRequest): Original request and provenance.
            setup (AttackSetup): Independent role model resources.
            journal (Journal): Independent run journal.
            lexicon (WordnetLexicon): Prepared lexical data.
            budget (WordnetBudget): Limits and traversal seed.
        """
        super().__init__(
            WordnetAttack(
                QwenAttackEnvelope(
                    base, setup, journal,
                    WordnetProbe(
                        setup.text, ModelChat(setup.attacker), lexicon,
                        budget=budget, journal=journal
                    )
                )
            )
        )
