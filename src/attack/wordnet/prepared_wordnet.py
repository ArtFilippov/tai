"""Offline access to an explicitly prepared wn database."""

from typing import override

import wn

from attack.wordnet.wordnet_lexicon import WordnetLexicon


class PreparedWordnet(WordnetLexicon):
    """Read the pinned garak lexicon without fetching resources at inference."""

    def __init__(self, wordnet: wn.Wordnet) -> None:
        """Keep an already opened, explicitly prepared lexicon.

        Args:
            wordnet (wn.Wordnet): Open English WordNet 2023.
        """
        super().__init__()
        self.wordnet: wn.Wordnet = wordnet

    @override
    def synsets(self, term: str) -> list[str]:
        return [node.id for node in self.wordnet.synsets(term)]

    @override
    def forms(self, node: str) -> list[str]:
        return [form for word in self.wordnet.synset(node).words() for form in word.forms()]

    @override
    def children(self, node: str) -> list[str]:
        return [child.id for child in self.wordnet.synset(node).hyponyms()]
