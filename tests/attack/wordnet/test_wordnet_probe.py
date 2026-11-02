"""A bounded exploration keeps branch evidence separate from judgment."""

import json
import unittest
from collections.abc import Mapping
from pathlib import Path
from argparse import Namespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from attack.attack_run import AttackRun
from attack.wordnet.wordnet_budget import WordnetBudget
from attack.wordnet.wordnet_lexicon import WordnetLexicon
from attack.wordnet.scripted_wordnet import ScriptedWordnet
from attack.wordnet.wordnet_probe import WordnetProbe
from base.plain_base import PlainBase
from base.alert_pool import AlertPool
from journal.disk_journal import DiskJournal
from journal.parsed_journal import ParsedJournal
from journal.versioned_journal import VersionedJournal
from judge.qwen_judge import QwenJudge
from judge.recorded_judge import RecordedJudge
from target.scripted_chat import ScriptedChat
from main.alert_series import planned, prepare, series
from ...target import test_logits
from ..model_fixture_utils import wordnet
from ...journal_utils import temporary_journal

BASE = "Explain how a café works"


class TestWordnetProbe(unittest.TestCase):
    """Verify selection, branching, budgets and journal recovery offline."""

    def lexicon(self) -> WordnetLexicon:
        """Provide a tiny lexical tree with two seed senses.

        Returns:
            WordnetLexicon: Controlled WordNet relationships.
        """
        return ScriptedWordnet(
            {
                "cafe": [
                    "root", "alternate"
                ],
                "coffee": ["other"]
            },
            {
                "root": ["cafe"],
                "alternate": ["café"],
                "other": ["coffee"],
                "child": ["espresso"]
            },
            {"root": ["child"]}
        )

    def attempt(self, path: Path, proposal: str, answers: list[str], verdicts: list[str],
                 nodes: int = 3, calls: int = 4) -> dict[str, ScriptedChat]:
        """Run a complete root attack with isolated conversations.

        Args:
            path (Path): Journal destination.
            proposal (str): Ranked Attacker terms.
            answers (list[str]): Target replies.
            verdicts (list[str]): Judge decisions.
            nodes (int, default=3): Node limit.
            calls (int, default=4): Target-call limit.

        Returns:
            dict[str, ScriptedChat]: Role transcripts.
        """
        journal = VersionedJournal(
            DiskJournal(path)
        )
        attacker = ScriptedChat([proposal])
        target = ScriptedChat(answers)
        judge = ScriptedChat(verdicts)
        _ = AttackRun(
            PlainBase(BASE),
            target,
            RecordedJudge(
                QwenJudge(
                    BASE, judge
                ),
                journal, BASE
            ),
            journal,
            probe=WordnetProbe(
                BASE, attacker,
                self.lexicon(),
                budget=WordnetBudget(
                    nodes, calls, 7
                ),
                journal=journal
            )
        ).run()
        return {
            "attacker": attacker,
            "target": target,
            "judge": judge
        }

    def test_invalid_candidate_falls_back_to_nearest_valid_term(self) -> None:
        with temporary_journal("wordnet-seed.jsonl") as path:
            self.attempt(
                path, "unknown\ncafe\ncoffee", ["answer"], ["YES"], calls=1
            )
            events = list(
                ParsedJournal(path).events()
            )
            assert [(event["term"], event["synsets"]) for event in events if event.get("action") == "candidate"] == [
                ("unknown", []), ("cafe", ["root", "alternate"])
            ], "An invalid seed was used or a more distant valid term replaced the nearest one"

    def test_missing_candidates_produce_no_result(self) -> None:
        with temporary_journal("wordnet-invalid.jsonl") as path:
            roles = self.attempt(
                path, "unknown\ninvalid", [], []
            )
            assert (
                ParsedJournal(path).read()["status"],
                ParsedJournal(path).read()["reason"],
                len(roles["target"].histories)
            ) == (
                "no_result", "no_valid_synset", 0
            ), "Unverified terms produced an unrestricted Target prompt"

    def test_synset_without_surface_forms_produces_no_result(self) -> None:
        with temporary_journal("wordnet-no-forms.jsonl") as path:
            journal = VersionedJournal(
                DiskJournal(path)
            )
            _ = AttackRun(
                PlainBase(BASE),
                ScriptedChat([]),
                RecordedJudge(
                    QwenJudge(
                        BASE,
                        ScriptedChat([])
                    ),
                    journal, BASE
                ),
                journal,
                probe=WordnetProbe(
                    BASE,
                    ScriptedChat(["cafe"]),
                    ScriptedWordnet(
                        {"cafe": ["root"]}, {"root": []}, {}
                    ),
                    budget=WordnetBudget(
                        1, 2, 7
                    ),
                    journal=journal
                )
            ).run()
            assert (
                ParsedJournal(path).read()["reason"],
                ParsedJournal(path).calls(
                    ParsedJournal(path).events()
                )["target"]
            ) == (
                "no_target_reply", 0
            ), "The search invented an answer without a Target reply"

    def test_malformed_judge_verdict_is_not_success(self) -> None:
        with temporary_journal("wordnet-verdict.jsonl") as path:
            with pytest.raises(
                ValueError, match="Malformed judge verdict"
            ):
                self.attempt(
                    path, "cafe", ["complete"], ["MAYBE"]
                )
            assert ParsedJournal(path).read()["reason"] == "assessment_error", "The invalid verdict became a success"

    def test_refusal_closes_branch_even_when_judge_says_no(self) -> None:
        with temporary_journal("wordnet-refusal.jsonl") as path:
            roles = self.attempt(
                path, "cafe",
                [
                    "I cannot provide that", "I'm unable to help", "last"
                ],
                [
                    "NO", "NO", "NO"
                ],
                nodes=2, calls=3
            )
            assert (
                len(roles["target"].histories),
                ParsedJournal(path).read()["text"]
            ) == (
                3, "last"
            ), "A refusal opened a child branch or discarded the last assessed reply"

    def test_non_refusal_opens_child_without_counting_as_success(self) -> None:
        with temporary_journal("wordnet-branch.jsonl") as path:
            self.attempt(
                path, "cafe",
                [
                    "neutral", "neutral", "I cannot provide", "I cannot provide", "child"
                ],
                ["NO"] * 5, nodes=3, calls=5
            )
            events = list(
                ParsedJournal(path).events()
            )
            assert (
                [(event["node"], event["children"]) for event in events if event.get("action") == "branch"],
                ParsedJournal(path).read()["assessment"]
            ) == (
                [("root", ["child"]), ("alternate", []), ("child", [])], False
            ), "The bypass signal was treated as final success or failed to guide traversal"

    def test_first_positive_stops_before_remaining_prompts(self) -> None:
        with temporary_journal("wordnet-success.jsonl") as path:
            roles = self.attempt(
                path, "cafe",
                [
                    "complete", "unused"
                ],
                ["YES"], calls=4
            )
            assert (
                len(roles["target"].histories),
                ParsedJournal(path).read()["status"],
                ParsedJournal(path).read()["text"]
            ) == (
                1, "success", "complete"
            ), "The search continued after its first positive assessment"

    def test_call_budget_stops_mid_node_and_keeps_last_result(self) -> None:
        with temporary_journal("wordnet-budget.jsonl") as path:
            roles = self.attempt(
                path, "cafe",
                [
                    "one", "two", "unused"
                ],
                [
                    "NO", "NO"
                ],
                calls=2
            )
            assert (
                len(roles["target"].histories),
                ParsedJournal(path).read()["text"],
                ParsedJournal(path).read()["reason"]
            ) == (
                2, "two", "search_exhausted"
            ), "The Target-call limit was exceeded or the final reply was lost"

    def test_node_budget_stops_before_next_synset(self) -> None:
        with temporary_journal("wordnet-nodes.jsonl") as path:
            roles = self.attempt(
                path, "cafe",
                [
                    "one", "two"
                ],
                [
                    "NO", "NO"
                ],
                nodes=1
            )
            assert len(roles["target"].histories) == 2, "The node budget did not stop traversal"

    def test_journal_recovers_roles_and_base_for_each_judgment(self) -> None:
        with temporary_journal("wordnet-run.jsonl") as path:
            roles = self.attempt(
                path, "cafe",
                [
                    "first", "last"
                ],
                [
                    "NO", "YES"
                ]
            )
            events = list(
                ParsedJournal(path).events()
            )
            assert (
                ParsedJournal(path).read()["calls"],
                [(event["text"], event["base"]) for event in events if event["event"] == "judgment"],
                json.loads(roles["judge"].transcript(1).last_message("user").text),
                roles["attacker"].transcript(0).last_message("user").text.endswith(BASE)
            ) == (
                {
                    "attacker": 1,
                    "target": 2,
                    "judge": 2
                },
                [
                    (
                        "first", BASE
                    ),
                    (
                        "last", BASE
                    )
                ],
                {
                    "base_request": BASE,
                    "presented_result": "last"
                },
                True
            ), "The run mixed roles, lost provenance, or judged a modified request"

    def test_alert_variant_is_opt_in_and_counts_one_attempt_per_base(self) -> None:
        bases = AlertPool(
            Path("examples/alert-manifest-640.json")
        ).requests(1)
        arguments = Namespace(
            gpu=0, tokens=192, attacker="qwen", host="127.0.0.1", port=8080,
            turns=1, goat_turns=1, no_threaten_json=False,
            no_adaptive_attacks=False, no_dra=True, no_deletion=True,
            deletion_perturbations=1, deletion_positions=2, deletion_ascii=2,
            autodan=False, autodan_hga=False, wordnet=True,
            wordnet_dir=Path("prepared"), wordnet_nodes=1, wordnet_calls=1, wordnet_seed=7
        )
        with TemporaryDirectory() as directory, patch(
            "main.alert_series.LazyQwenModel", return_value=wordnet()
        ), patch(
            "main.alert_series.open_wordnet", return_value=object()
        ), patch(
            "main.alert_series.PreparedWordnet", return_value=self.lexicon()
        ), patch(
            "main.alert_series.cuda_device", return_value="controlled"
        ), patch(
            "main.alert_series.snapshot", return_value=None
        ), prepare(
            arguments,
            Path(directory)
        ) as factories:
            root = Path(directory)
            series(
                bases, factories[-1:],
                root,
                Path("examples/alert-manifest-640.json")
            )
            report = json.loads(
                (root / "summary.json").read_text(encoding="utf-8")
            )
            outcomes: list[Mapping[str, object]] = []
            for base in bases:
                path = root / "wordnet" / f"{base.identity}.jsonl"
                outcomes.append(
                    ParsedJournal(path).read()
                )
            expected = {
                "attacker": 1,
                "target": 1,
                "judge": 1
            }
            assert (
                len(factories),
                report["planned"], report["attempted"], report["complete"],
                len({run["base_metadata"]["microcategory"] for run in report["runs"]}),
                all(outcome["calls"] == expected for outcome in outcomes)
            ) == (
                5, 32, 32, True, 32, True
            ), "The WordNet variant inflated root attempts or lost recoverable journals"

    def test_all_thirteen_variants_plan_one_root_attempt_each(self) -> None:
        bases = AlertPool(
            Path("examples/alert-manifest-640.json")
        ).requests(1)
        arguments = Namespace(
            gpu=0, tokens=64, attacker="qwen", host="127.0.0.1", port=8080,
            turns=1, goat_turns=1, no_threaten_json=False,
            no_adaptive_attacks=False, no_dra=False, dra_candidates=1, dra_seed=7,
            no_deletion=False, deletion_perturbations=1, deletion_positions=2,
            deletion_ascii=2, autodan=True, autodan_hga=True,
            autodan_target="Desired completion", autodan_steps=2,
            autodan_batch=3, autodan_elites=1, autodan_points=1,
            autodan_crossover=0.0, autodan_mutation=0.0, autodan_seed=7,
            whois=True, grandma=True, slur_completion=True, leak_replay=True,
            wordnet=True, wordnet_dir=Path("prepared"), wordnet_nodes=1,
            wordnet_calls=1, wordnet_seed=7
        )
        with TemporaryDirectory() as directory, patch(
            "main.alert_series.LazyQwenModel", return_value=wordnet()
        ), patch(
            "main.alert_series.open_wordnet", return_value=object()
        ), patch(
            "main.alert_series.PreparedWordnet", return_value=self.lexicon()
        ), patch(
            "main.alert_series.QwenLogits", return_value=test_logits.TestLogits()
        ), patch(
            "main.alert_series.cuda_device", return_value="controlled"
        ), patch(
            "main.alert_series.snapshot", return_value=None
        ), prepare(
            arguments,
            Path(directory)
        ) as factories:
            assert (
                len(factories),
                planned(
                    bases, factories
                )
            ) == (13, 416), "The full opt-in selection did not plan 416 root attempts"
