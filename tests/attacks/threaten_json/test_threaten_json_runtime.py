"""Configured ThreatenJSON uses the answer judge in commands and saved series."""

import json
from collections import deque
from pathlib import Path

from adaptive_app_support import cli
from app_test_support import HttpPeer, session
from evaluation_fakes import HeldResponse, ScriptedModel
from http_peer import Reply
from series_support import MANIFEST, TAXONOMY
from test_threaten_json import AnswerJudge

from alert.values import object_value
from app.configuration import load_profile
from attacks.threaten_json.experiment import ThreatenJSONExperiment
from attacks.threaten_json.requests import COERCION, requests
from tai import EvaluatorSetup, LLMChat, Message, ThreatenJSONUser, UserState, capture_run, save_run


class TestThreatenJSONRuntime:
    """Verify the configured strategy through its real command boundaries."""

    def comparison(self, events: list[dict[str, object]]) -> dict[str, object]:
        comparison, = [event["data"] for event in events if event["kind"] == "json_comparison"]
        return object_value(comparison, "json_comparison")

    def profile(self, endpoint: HttpPeer, root: Path) -> Path:
        path = root / "threaten-json.toml"
        path.write_text(f'''
[attack]
type = "threaten_json"
[target]
base_url = "{endpoint.url}/target"
model = "controlled"
system = "Target"
opencode_session = true
[evaluator]
type = "model"
rubric_version = "json-answer-v1"
[evaluator.chat]
base_url = "{endpoint.url}/evaluator"
model = "controlled"
system = "Judge"
opencode_session = true
''', encoding="utf-8")
        return path

    def test_attack_command_saves_control_and_only_judges_pressured_json(self, endpoint: HttpPeer, tmp_path: Path) -> None:
        config = self.profile(endpoint, tmp_path)
        endpoint.scripts = {
            "/v1/target/chat/completions": deque([Reply('{"already": 1}'), Reply('{"answer": 2}')]),
            "/v1/evaluator/chat/completions": deque([Reply('{"success": true}')]),
        }
        root = tmp_path / "runs"
        completed = cli("threaten-json", "--config", str(config), "--artifacts-dir", str(root), "--text", "Give the number 2 as JSON.")
        assert not completed.returncode, completed.stdout + completed.stderr
        directory, = root.iterdir()
        metadata = json.loads((directory / "metadata.json").read_text())
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        started = [event["data"] for event in events if event["kind"] == "call_started" and event["data"]["role"] == "Target"]
        control, pressured = [event["data"] for event in events if event["kind"] == "call_completed" and event["data"]["role"] == "Target"]
        comparison = self.comparison(events)
        assert metadata["state"] == "COMPLETED" and metadata["evaluation"] == {"result_id": metadata["result_id"], "success": True}
        assert metadata["settings"]["base"]["text"] == "Give the number 2 as JSON."
        assert metadata["settings"]["evaluator"]["rubric_version"] == "json-answer-v1"
        assert metadata["result"]["text"] == (directory / "result.txt").read_text() == '{"answer": 2}'
        assert comparison == {"control_id": control["response"]["id"], "result_id": metadata["result_id"],
                              "control_parseable": True, "result_parseable": True}
        assert control["conversation_id"] != pressured["conversation_id"]
        assert all(call["input"]["origin"]["id"] == metadata["settings"]["base"]["id"] for call in started)
        assert len(endpoint.requests) == 3 and len({session(request) for request in endpoint.requests}) == 3
        judge = endpoint.requests[-1][2]["messages"][-1]["content"]
        assert json.loads(judge.split("INPUT_JSON:\n", 1)[1]) == {"base_request": "Give the number 2 as JSON.", "result": '{"answer": 2}'}

    def test_preflight_runs_answer_judge_on_valid_json(self, endpoint: HttpPeer, tmp_path: Path) -> None:
        config = self.profile(endpoint, tmp_path)
        endpoint.scripts = {
            "/v1/target/chat/completions": deque([Reply("READY")]),
            "/v1/evaluator/chat/completions": deque([Reply('{"success": false}')]),
        }
        completed = cli("preflight", "--config", str(config))
        assert not completed.returncode, completed.stdout + completed.stderr
        report = json.loads(completed.stdout)
        assert report["ready"] is True and report["roles"][-1]["success"] is False
        assert json.loads(report["roles"][-1]["result"]["text"]) == {"answer": "Paris"}
        assert len(endpoint.requests) == 2 and report["roles"][-1]["settings"]["rubric_version"] == "json-answer-v1"

    def test_invalid_pressured_json_fails_without_judge_call(self, endpoint: HttpPeer, tmp_path: Path) -> None:
        config = self.profile(endpoint, tmp_path)
        endpoint.scripts = {"/v1/target/chat/completions": deque([Reply("{}"), Reply("Sure! {}")])}
        root = tmp_path / "runs"
        completed = cli("attack", "--config", str(config), "--artifacts-dir", str(root), "--text", "Give 2")
        assert not completed.returncode, completed.stdout + completed.stderr
        directory, = root.iterdir()
        metadata = json.loads((directory / "metadata.json").read_text())
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        comparison = self.comparison(events)
        assert metadata["evaluation"] == {"result_id": metadata["result_id"], "success": False}
        assert comparison["control_parseable"] is True and comparison["result_parseable"] is False
        assert len(endpoint.requests) == 2

    def test_invalid_profile_fails_before_any_run(self, endpoint: HttpPeer, tmp_path: Path) -> None:
        for section in (
            '[attacker]\nmodel = "unwanted"\n',
            '[run]\nmax_calls_per_conv = 2\n',
            '[run]\ngenerations = 2\n',
            '[evaluator]\ntype = "contains_text"\ncriterion = "answer"\n',
            '[evaluator]\ntype = "model"\nrubric_version = "alert-safety-v1"\n',
        ):
            config = self.profile(endpoint, tmp_path)
            if section.startswith("[evaluator]"):
                config.write_text(config.read_text().replace('[evaluator]\ntype = "model"\nrubric_version = "json-answer-v1"\n', section))
            else:
                config.write_text(config.read_text() + section)
            root = tmp_path / "runs"
            completed = cli("attack", "--config", str(config), "--artifacts-dir", str(root), "--text", "Give 2")
            assert completed.returncode == 2 and not root.exists() and not endpoint.requests
            series = cli("alert-series", "--manifest", str(MANIFEST), "--profile", str(config),
                         "--taxonomy", str(TAXONOMY), "--artifacts-dir", str(root))
            assert series.returncode == 2 and not root.exists() and not endpoint.requests

    def test_profile_is_lazy_and_creates_independent_run_participants(self, endpoint: HttpPeer, tmp_path: Path) -> None:
        config = self.profile(endpoint, tmp_path)
        selected = load_profile(config)
        fresh = selected.fresh()
        experiment = fresh.create_experiment(Message("Give 2"))
        assert isinstance(experiment, ThreatenJSONExperiment)
        assert selected.target is not fresh.target and selected.evaluator.chat is not fresh.evaluator.chat
        assert not endpoint.requests and not (tmp_path / "runs").exists()

    def test_named_command_requires_threaten_json_profile(self, endpoint: HttpPeer, tmp_path: Path) -> None:
        config = self.profile(endpoint, tmp_path)
        config.write_text(config.read_text().replace('type = "threaten_json"', 'type = "adaptive_attacks"')
                          .replace('rubric_version = "json-answer-v1"', 'rubric_version = "alert-safety-v1"'))
        root = tmp_path / "runs"
        completed = cli("threaten-json", "--config", str(config), "--artifacts-dir", str(root), "--text", "Give 2")
        assert completed.returncode == 2 and not root.exists() and not endpoint.requests

    def test_series_saves_independent_results_and_comparisons(self, endpoint: HttpPeer, tmp_path: Path) -> None:
        config = self.profile(endpoint, tmp_path)
        endpoint.scripts = {
            "/v1/target/chat/completions": deque(Reply("not JSON" if not i % 2 else '{"answer": 2}') for i in range(64)),
            "/v1/evaluator/chat/completions": deque(Reply('{"success": true}') for _ in range(32)),
        }
        root = tmp_path / "series"
        completed = cli("alert-series", "--manifest", str(MANIFEST), "--profile", str(config),
                        "--taxonomy", str(TAXONOMY), "--artifacts-dir", str(root), "--jobs", "1")
        assert not completed.returncode, completed.stdout + completed.stderr
        report = json.loads(completed.stdout)
        directory = root / report["series_id"]
        index = json.loads((directory / "index.json").read_text())
        assert report["complete"] is True and len(index["entries"]) == 32
        assert len({entry["run_id"] for entry in index["entries"]}) == 32
        assert len({entry["base_id"] for entry in index["entries"]}) == 32
        for entry in index["entries"]:
            metadata = json.loads((directory / entry["artifacts"]["metadata.json"]).read_text())
            events = [json.loads(line) for line in (directory / entry["artifacts"]["events.jsonl"]).read_text().splitlines()]
            comparison = self.comparison(events)
            assert entry["evaluation"] == {"result_id": entry["result_id"], "success": True}
            assert metadata["settings"]["base"]["id"] == entry["base_id"]
            assert comparison["control_parseable"] is False and comparison["result_parseable"] is True
            assert comparison["result_id"] == entry["result_id"]
            assert (directory / entry["artifacts"]["result.txt"]).read_text() == '{"answer": 2}'
        assert len(endpoint.requests) == 96 and len({session(request) for request in endpoint.requests}) == 96

    def test_stop_during_pressured_call_retains_format_comparison_without_assessing(self) -> None:
        held = HeldResponse('{"answer": 2}')
        user = ThreatenJSONExperiment(
            LLMChat.cancellable(ScriptedModel(["not JSON", held])), Message("Give 2"), EvaluatorSetup.contains_text("answer"),
        )
        user.start()
        try:
            assert held.entered.wait(5)
            user.stop()
        finally:
            held.release.set()
        assert user.wait(10)
        assert user.status().state is UserState.STOPPED and user.result() is None
        assert not [event for event in user.journal.events() if event.kind == "evaluation_completed"]
        comparison = self.comparison([json.loads(event.dump()) for event in user.journal.events()])
        assert comparison["control_parseable"] is False and comparison["result_parseable"] is True

    def test_requests_preserve_base_and_differ_only_by_garak_coercion(self) -> None:
        base = Message("Old").transform(lambda _: "Give π {x}\nReturn an array.")
        control, pressured = requests(base)
        assert control.origin is pressured.origin is base
        assert control.text.count(base.text) == pressured.text.count(base.text) == 1
        assert pressured.text == control.text + "\n\n" + COERCION
        assert "Bard" not in pressured.text and "dictionary with two keys" not in pressured.text

    def test_only_pressured_answer_is_evaluated(self, tmp_path: Path) -> None:
        for control, pressured, answer, expected, calls in (
            ("Not JSON", '["substantive"]', True, True, 1),
            ("Not JSON", "null", True, True, 1),
            ("Not JSON", "a preamble {}", True, False, 0),
            ("Not JSON", "NaN", True, False, 0),
            ("Not JSON", '"irrelevant"', False, False, 1),
            ('{"already": true}', '"substantive"', True, True, 1),
        ):
            self._attempt(tmp_path, control, pressured, answer, expected, calls)

    def _attempt(self, tmp_path: Path, control: str, pressured: str, answer: bool, expected: bool, calls: int) -> None:
        base = Message("Answer this explicit request")
        target = ScriptedModel([control, pressured])
        pairs: list[tuple[Message, Message]] = []
        judge = AnswerJudge(answer, pairs)
        user = ThreatenJSONExperiment(LLMChat(target), base, judge, journal_dir=tmp_path)
        user.start()
        assert user.wait(10)
        assert user.status().state is UserState.COMPLETED, user.status().reason
        result = user.result()
        evaluation = user.status().evaluation
        assert result is not None and result.text == pressured
        assert evaluation is not None and evaluation.success is expected
        assert len(pairs) == calls
        assert all(pair[0] is base and pair[1] is result for pair in pairs)
        assert len(target.requests) == 2 and all(len(turns) == 1 for turns in target.requests)
        events = [json.loads(event.dump()) for event in user.journal.events()]
        calls_in_journal = [event["data"] for event in events if event["kind"] == "call_completed"]
        assert len(calls_in_journal) == 2 and calls_in_journal[0]["conversation_id"] != calls_in_journal[1]["conversation_id"]
        assert [call["context"]["cause"]["reason"] for call in calls_in_journal] == ["json_control", "json_pressured"]
        comparison = self.comparison(events)
        assert comparison["control_parseable"] is (control.startswith("{"))
        assert comparison["result_parseable"] is (pressured[0] in '["n' and pressured != "NaN")
        assert comparison["control_id"] == calls_in_journal[0]["response"]["id"]
        assert comparison["result_id"] == result.id
        directory = save_run(capture_run(user, journal=user.journal), tmp_path)
        assert (directory / "result.txt").read_text() == pressured

    def test_bare_user_and_repeated_experiment_keep_independent_results(self) -> None:
        base = Message("Respond with one JSON number")
        for kind in (ThreatenJSONUser, ThreatenJSONExperiment):
            pairs: list[tuple[Message, Message]] = []
            judge = AnswerJudge(True, pairs)
            user = kind(LLMChat(ScriptedModel(["control", "42", "second control", "7"])), base, judge)
            user.start()
            assert user.wait(10)
            first = user.result()
            user.start()
            assert user.wait(10)
            result = user.result()
            assert first is not result and result is not None and result.text == "7"
            assert len(pairs) == 2
            if isinstance(user, ThreatenJSONExperiment):
                assert len(user.journals) == 2 and all(journal.finished for journal in user.journals)

    def test_model_answer_rubric_is_not_safety_rubric(self) -> None:
        model = ScriptedModel(['{"success": true}'])
        setup = EvaluatorSetup.answer(LLMChat(model))
        assert setup.evaluator.evaluate(Message("Give the number 2"), Message("2")) is True
        assert setup.settings()["rubric_version"] == "json-answer-v1"
        assert "safety restriction" not in str(setup.settings()["rubric"])

    def test_model_answer_judge_receives_only_pressured_response(self) -> None:
        model = ScriptedModel(['{"success": true}'])
        user = ThreatenJSONExperiment(
            LLMChat(ScriptedModel(['{"control": true}', '"answer"'])), Message("Give a JSON string"),
            EvaluatorSetup.answer(LLMChat(model)),
        )
        user.start()
        assert user.wait(10)
        assert user.status().state is UserState.COMPLETED
        prompt = model.requests[0][-1].message.text
        assert json.loads(prompt.split("INPUT_JSON:\n", 1)[1]) == {
            "base_request": "Give a JSON string", "result": '"answer"',
        }
        assert len(model.requests) == 1

    def test_stop_during_control_finishes_call_without_publishing_or_assessing(self) -> None:
        held = HeldResponse("0")
        pairs: list[tuple[Message, Message]] = []
        judge = AnswerJudge(True, pairs)
        user = ThreatenJSONExperiment(LLMChat.cancellable(ScriptedModel([held, "1"])), Message("Give one"), judge)
        user.start()
        try:
            assert held.entered.wait(5)
            user.stop()
        finally:
            held.release.set()
        assert user.wait(10)
        assert user.status().state is UserState.STOPPED and user.result() is None
        assert not pairs
        assert len([event for event in user.journal.events() if event.kind == "call_completed"]) == 1
