"""A locally installed miniature lexicon is usable without inference downloads."""

import unittest
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import wn
from wn._config import config

from attack.wordnet.prepared_wordnet import PreparedWordnet
from attack.wordnet.wordnet_source_utils import open_wordnet
from journal.parsed_journal import ParsedJournal

LMF = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE LexicalResource SYSTEM "https://globalwordnet.github.io/schemas/WN-LMF-1.0.dtd">
<LexicalResource>
  <Lexicon id="oewn" label="Local test lexicon" language="en" email="test@example.org"
           license="https://creativecommons.org/licenses/by/4.0/" version="2023">
    <LexicalEntry id="w-cafe"><Lemma writtenForm="cafe" partOfSpeech="n"/><Sense id="s-cafe" synset="n-cafe"/></LexicalEntry>
    <LexicalEntry id="w-coffee"><Lemma writtenForm="coffee" partOfSpeech="n"/><Sense id="s-coffee" synset="n-coffee"/></LexicalEntry>
    <Synset id="n-cafe" ili="in" partOfSpeech="n">
      <Definition>Place serving coffee</Definition>
      <SynsetRelation relType="hyponym" target="n-coffee"/>
    </Synset>
    <Synset id="n-coffee" ili="in" partOfSpeech="n"><Definition>Coffee place</Definition></Synset>
  </Lexicon>
</LexicalResource>
"""


class TestPreparedWordnet(unittest.TestCase):
    """Exercise a real wn database and fail fast when it is absent."""

    def test_unprepared_directory_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            with pytest.raises(
                ValueError, match="not prepared"
            ):
                _ = open_wordnet(
                    Path(directory)
                )

    def test_database_without_selected_lexicon_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            config.data_directory = Path(directory)
            _ = wn.lexicons()
            with pytest.raises(
                ValueError, match="lexicon oewn:2023 is not prepared"
            ):
                _ = open_wordnet(
                    Path(directory)
                )

    def test_local_lexicon_exposes_real_synsets_and_hyponyms(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mini.xml"
            source.write_text(
                LMF, encoding="utf-8"
            )
            config.data_directory = root
            wn.add(
                source, progress_handler=None
            )
            lexicon = PreparedWordnet(
                open_wordnet(root)
            )
            assert (
                lexicon.synsets("cafe"),
                lexicon.forms("n-cafe"),
                lexicon.children("n-cafe")
            ) == (
                ["n-cafe"], ["cafe"], ["n-coffee"]
            ), "The prepared local lexicon lost its lexical relationships"

    def test_standalone_command_runs_with_prepared_offline_lexicon(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mini.xml"
            source.write_text(
                LMF, encoding="utf-8"
            )
            config.data_directory = root
            wn.add(
                source, progress_handler=None
            )
            journal = root / "run.jsonl"
            script = (
                "import main.wordnet_cli as cli; "
                "from tests.attack.model_fixture_utils import wordnet; "
                "cli.snapshot = lambda offline: None; "
                "cli.cuda_device = lambda gpu: 'controlled'; "
                "cli.LazyQwenModel = lambda device, tokens: wordnet(); "
                "cli.main()"
            )
            subprocess.run(
                [
                    sys.executable, "-c", script,
                    str(journal),
                    "Explain a café", "--lexicon-dir",
                    str(root),
                    "--gpu", "0", "--calls", "1"
                ],
                check=True, capture_output=True, text=True
            )
            assert (
                ParsedJournal(journal).read()["text"],
                ParsedJournal(journal).read()["calls"]
            ) == (
                "A café answer",
                {
                    "attacker": 1,
                    "target": 1,
                    "judge": 1
                }
            ), "The standalone command could not recover its offline exploration"
