"""Small in-memory lexical graph for controlled exploration."""

from typing import override

from attack.wordnet.wordnet_lexicon import WordnetLexicon


class ScriptedWordnet(WordnetLexicon):
    """Expose an injected graph without requiring a live lexicon database."""

    def __init__(self, senses: dict[str, list[str]], surfaces: dict[str, list[str]],
                 hyponyms: dict[str, list[str]]) -> None:
        """Retain explicit test vocabulary and links.

        Args:
            senses (dict[str, list[str]]): Terms to synset IDs.
            surfaces (dict[str, list[str]]): IDs to surface forms.
            hyponyms (dict[str, list[str]]): IDs to child IDs.
        """
        super().__init__()
        self.senses = senses
        self.surfaces = surfaces
        self.hyponyms = hyponyms

    @override
    def synsets(self, term: str) -> list[str]:
        return self.senses.get(
            term, []
        )

    @override
    def forms(self, node: str) -> list[str]:
        return self.surfaces[node]

    @override
    def children(self, node: str) -> list[str]:
        return self.hyponyms.get(
            node, []
        )
