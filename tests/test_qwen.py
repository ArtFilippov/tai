"""Public local-chat contracts, using controlled libraries without weights or CUDA."""

import json
import subprocess
import sys
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tai import (
    ChatCancelled,
    ContainsTextEvaluator,
    FITDUser,
    Message,
    ModelEvaluationError,
    ModelEvaluator,
    QwenChat,
    QwenContextError,
    QwenError,
    UserState,
    qwen,
)
from tai._config import load_profile
from tai._preflight import run_preflight


class Tokens:
    def __init__(self, tokens):
        self.tokens = tokens
        self.shape = (1, len(tokens))

    def __getitem__(self, index):
        row, span = index
        assert row == 0
        return self.tokens[span]


class Inputs(dict):
    def to(self, device):
        self.device = device
        return self


class LocalLibraries:
    def __init__(self):
        self.loads = []
        self.requests = []
        self.generations = []
        self.decoded = []
        self.imports = []
        self.cuda_available = True
        self.load_action = lambda: None
        self.generate_action = lambda criteria: None
        self.sync_action = lambda: None
        self.inference = False
        self.input_length = 7
        self.limit = 32768
        self.responses = deque()
        self.torch = SimpleNamespace(
            float16="torch.float16",
            bfloat16="torch.bfloat16",
            float32="torch.float32",
            inference_mode=self.inference_mode,
            cuda=SimpleNamespace(
                is_available=lambda: self.cuda_available,
                device_count=lambda: 2,
                is_bf16_supported=lambda: False,
                device=self.device,
                synchronize=lambda device: self.sync_action(),
            ),
        )
        self.transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=self.tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=self.model),
            GenerationConfig=SimpleNamespace,
            StoppingCriteria=object,
            StoppingCriteriaList=list,
        )

    def imports_for(self, name):
        self.imports.append(name)
        return {"torch": self.torch, "transformers": self.transformers}[name]

    @contextmanager
    def inference_mode(self):
        assert not self.inference
        self.inference = True
        try:
            yield
        finally:
            self.inference = False

    @contextmanager
    def device(self, name):
        assert name in ("cuda:0", "cuda:1")
        yield

    def tokenizer(self, path, **kwargs):
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        self.loads.append(("tokenizer", path))
        return SimpleNamespace(
            apply_chat_template=self.template,
            decode=self.decode,
            model_max_length=self.limit,
            pad_token_id=99,
        )

    def model(self, path, **kwargs):
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        assert kwargs["use_safetensors"] is True
        self.loads.append(("model", path))
        self.load_action()
        library = self

        class Model:
            dtype = kwargs["torch_dtype"]
            config = SimpleNamespace(max_position_embeddings=library.limit)
            generation_config = SimpleNamespace(bos_token_id=99, eos_token_id=[99])
            evaluating = False

            def to(self, device):
                self.device = device

            def eval(self):
                self.evaluating = True

            def generate(self, **kwargs):
                assert self.evaluating and library.inference
                assert "past_key_values" not in kwargs
                library.generations.append((self.device, kwargs))
                library.generate_action(kwargs["stopping_criteria"])
                return Tokens(kwargs["input_ids"].tokens + [42, 99])

        return Model()

    def template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
            "truncation": False,
        }
        self.requests.append(messages)
        return Inputs(
            input_ids=Tokens([1] * self.input_length),
            attention_mask=Tokens([1] * self.input_length),
        )

    def decode(self, tokens, **kwargs):
        assert tokens == [42, 99], "Input prompt must not be decoded as assistant text"
        assert kwargs == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        self.decoded.append(tokens)
        return self.responses.popleft() if self.responses else "yes"


@pytest.fixture
def local(tmp_path, monkeypatch):
    libraries = LocalLibraries()
    monkeypatch.setattr(qwen, "import_module", libraries.imports_for)
    revision = uuid4().hex + "12345678"
    cache = tmp_path / "cache"
    snapshot = cache / "models--Qwen--test" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
    ):
        (snapshot / name).touch()
    kwargs = dict(
        model="Qwen/test", revision=revision, cache_dir=cache, device="cuda:0"
    )
    return libraries, kwargs, snapshot


def test_roles_history_provenance_and_branching_with_one_lazy_load(local):
    lib, kwargs, snapshot = local
    target = QwenChat(**kwargs, system="Target instructions")
    attacker = QwenChat(
        **kwargs,
        system="Attacker instructions",
        generation={"do_sample": True, "temperature": 0.7},
    )
    assert target.settings()["loaded"] is False
    assert attacker.settings()["actual_dtype"] is None
    assert not lib.loads and not lib.imports
    original = Message("first π\n")
    point = target.checkpoint()
    response = target.send(original)
    assert response.to_string() == "yes"
    assert json.loads(response.dump())["origin"]["id"] == original.id
    branch = target.checkpoint()
    target.send(Message("discard"))
    target.restore(branch)
    target.send(Message("retained"))
    attacker.send(Message("separate"))
    assert lib.requests[2] == [
        {"role": "system", "content": "Target instructions"},
        {"role": "user", "content": "first π\n"},
        {"role": "assistant", "content": "yes"},
        {"role": "user", "content": "retained"},
    ]
    assert lib.requests[3] == [
        {"role": "system", "content": "Attacker instructions"},
        {"role": "user", "content": "separate"},
    ]
    assert target.conversation_id != attacker.conversation_id
    with pytest.raises(ValueError, match="Foreign"):
        attacker.restore(point)
    target.restore(point)
    target.send(Message("restored"))
    old_id = target.conversation_id
    target.reset()
    assert target.conversation_id != old_id
    with pytest.raises(ValueError, match="stale"):
        target.restore(branch)
    target.send(Message("new conversation"))
    assert len(lib.requests[-1]) == 2
    assert lib.loads == [("tokenizer", str(snapshot)), ("model", str(snapshot))]
    assert lib.generations[0][1]["generation_config"].do_sample is False
    assert lib.generations[3][1]["generation_config"].temperature == 0.7
    assert target.settings()["actual_dtype"] == "float16"
    settings = target.settings()
    settings["generation"]["max_new_tokens"] = 1
    assert target.settings()["generation"]["max_new_tokens"] == 512


@pytest.mark.parametrize("override", [{"device": "cuda:1"}, {"dtype": "float32"}])
def test_distinct_device_or_dtype_requires_distinct_weights(local, override):
    lib, kwargs, _ = local
    target = QwenChat(**kwargs)
    attacker = QwenChat(**{**kwargs, **override})
    target.send(Message("one"))
    attacker.send(Message("two"))
    assert len(lib.loads) == 4


def in_thread(chat, outcomes):
    def send():
        try:
            outcomes.append(chat.send(Message("request")))
        except Exception as exc:
            outcomes.append(exc)

    worker = Thread(target=send)
    worker.start()
    return worker


def test_cancel_waiter_does_not_cancel_active_role_and_serializes_generation(local):
    lib, kwargs, _ = local
    target, attacker = QwenChat(**kwargs), QwenChat(**kwargs)
    entered, release, waiting = Event(), Event(), Event()

    def hold(criteria):
        entered.set()
        assert release.wait(3)
        assert not criteria[0]()

    lib.generate_action = hold
    active, queued = [], []
    first = in_thread(target, active)
    try:
        assert entered.wait(2)
        # Signal when send has entered its generator, before waiting for weights.
        original = attacker._generate

        def generate(turns, cancellation):
            waiting.set()
            return original(turns, cancellation)

        attacker._generate = generate
        second = in_thread(attacker, queued)
        assert waiting.wait(1)
        assert not attacker.history()
        assert len(lib.generations) == 1
        attacker.cancel()
        second.join(1)
        assert not second.is_alive()
        assert isinstance(queued[0], ChatCancelled)
        assert queued[0].response is None
        assert not attacker.history()
        assert first.is_alive()
    finally:
        release.set()
        first.join(3)
    assert isinstance(active[0], Message)
    lib.generate_action = lambda criteria: None
    attacker.send(Message("after cancellation"))
    assert len(lib.loads) == 2


def test_active_cancellation_stops_decoding_and_waits_for_cuda_cleanup(local):
    lib, kwargs, _ = local
    chat = QwenChat(**kwargs)
    entered, cleaning, release = Event(), Event(), Event()

    def decode(criteria):
        entered.set()
        for _ in range(300):
            if criteria[0]():
                return
            Event().wait(0.01)
        pytest.fail("generation did not observe cancellation")

    def cleanup():
        if entered.is_set():
            cleaning.set()
            assert release.wait(3)

    lib.generate_action = decode
    lib.sync_action = cleanup
    outcomes = []
    worker = in_thread(chat, outcomes)
    try:
        assert entered.wait(2)
        chat.cancel()
        assert cleaning.wait(2)
        assert worker.is_alive() and not outcomes
        assert not chat.history()
        with pytest.raises(RuntimeError, match="active exchange"):
            chat.reset()
    finally:
        release.set()
        worker.join(3)
    assert isinstance(outcomes[0], ChatCancelled)
    assert outcomes[0].response is None
    assert not lib.decoded and not chat.history()
    lib.generate_action = lambda criteria: None
    chat.send(Message("retry"))
    assert len(lib.loads) == 2


def test_precancelled_exchange_does_not_import_or_load(local):
    lib, kwargs, _ = local
    chat = QwenChat(**kwargs)
    cancellation = Event()
    cancellation.set()
    with pytest.raises(ChatCancelled):
        chat.send(Message("cancelled"), cancellation=cancellation)
    assert not lib.imports and not lib.loads and not chat.history()


def test_loading_is_background_work_and_user_stop_waits_for_it(local, tmp_path):
    lib, kwargs, _ = local
    entered, release = Event(), Event()

    def loading():
        entered.set()
        assert release.wait(3)

    lib.load_action = loading
    target, attacker = QwenChat(**kwargs), QwenChat(**kwargs)
    user = FITDUser(
        target,
        attacker,
        Message("puzzle"),
        ContainsTextEvaluator("key"),
        artifacts_dir=tmp_path / "runs",
        max_calls_per_conv=1,
    )
    user.start()
    try:
        assert entered.wait(2)
        user.stop()
        assert user.status().state is UserState.STOPPING
        assert user.events() and user.result() is None
        assert not user.wait(0.01)
        assert not lib.generations
    finally:
        release.set()
        assert user.wait(3)
    assert user.status().state is UserState.STOPPED
    assert not target.history() and not attacker.history()
    assert not lib.generations
    # The interrupted load was not published; a subsequent run can load and finish.
    lib.load_action = lambda: None
    user.start()
    assert user.wait(3)
    assert user.status().state is UserState.COMPLETED
    assert user.result() is not None
    loads = len(lib.loads)
    user.start()
    assert user.wait(3)
    assert user.status().state is UserState.COMPLETED
    assert len(lib.loads) == loads
    directory = tmp_path / "runs" / user.status().run_id
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["final_settings"]["Target"]["actual_dtype"] == "float16"
    assert (directory / "result.txt").read_text() == user.result().to_string()


@pytest.mark.parametrize("phase", ["load", "generate"])
def test_library_errors_remain_errors_and_do_not_complete_exchange(local, phase):
    lib, kwargs, _ = local
    chat = QwenChat(**kwargs)

    def fail(*args):
        raise RuntimeError("CUDA out of memory (controlled)")

    if phase == "load":
        lib.load_action = fail
    else:
        lib.generate_action = fail
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        chat.send(Message("request"))
    assert not chat.history()
    lib.load_action = lambda: None
    lib.generate_action = lambda criteria: None
    assert chat.send(Message("retry")).to_string() == "yes"


def test_context_window_is_checked_without_truncation(local):
    lib, kwargs, _ = local
    lib.input_length = 32700
    chat = QwenChat(**kwargs)
    with pytest.raises(QwenContextError, match="input=32700.*limit=32768"):
        chat.send(Message("entire long message"))
    assert lib.requests[0][-1]["content"] == "entire long message"
    assert not lib.generations and not chat.history()


def test_missing_local_files_gives_explicit_preparation_command(local):
    lib, kwargs, snapshot = local
    (snapshot / "model.safetensors").unlink()
    chat = QwenChat(**kwargs)
    with pytest.raises(QwenError, match="hf download Qwen/test --revision"):
        chat.send(Message("request"))
    assert not lib.loads and not chat.history()


def test_missing_dependencies_and_cuda_do_not_fall_back(local, monkeypatch):
    lib, kwargs, _ = local
    chat = QwenChat(**kwargs)

    def missing(name):
        raise ImportError("missing optional library")

    monkeypatch.setattr(qwen, "import_module", missing)
    with pytest.raises(QwenError, match="uv sync --locked --extra qwen"):
        chat.send(Message("request"))
    monkeypatch.setattr(qwen, "import_module", lib.imports_for)
    lib.cuda_available = False
    with pytest.raises(QwenError, match="CUDA device cuda:0 is unavailable"):
        chat.send(Message("request"))
    assert not lib.loads and not chat.history()


def test_import_and_observation_do_not_import_optional_libraries():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tai._config import load_profile; "
            "from pathlib import Path; import sys; "
            "p = load_profile(Path('examples/fitd-alert-qwen.toml')); "
            "from tai import Message; p.create_evaluator(Message('base')).settings(); "
            "p.target.settings(); "
            "assert not {'torch', 'transformers', 'huggingface_hub'} "
            "& sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr


def config_for(local, tmp_path):
    _, kwargs, _ = local
    text = Path("examples/fitd-qwen.toml").read_text()
    text = text.replace("Qwen/Qwen2.5-0.5B-Instruct", kwargs["model"])
    text = text.replace("7ae557604adf67be50417f59c2c2f167def9a775", kwargs["revision"])
    text = text.replace("~/.cache/tai/huggingface", str(kwargs["cache_dir"]))
    config = tmp_path / "qwen.toml"
    config.write_text(text)
    return config


def test_preflight_uses_public_chats_and_reports_effective_cuda_settings(
    local, tmp_path
):
    lib, _, _ = local
    profile = load_profile(config_for(local, tmp_path))
    lib.responses.extend(["A real-shaped answer", "READY"])
    report = tmp_path / "preflight.json"
    assert run_preflight(profile, report) == 0
    data = json.loads(report.read_text())
    assert data["ready"] is True
    assert [r["response"] for r in data["roles"]] == ["A real-shaped answer", "READY"]
    assert len({r["conversation_id"] for r in data["roles"]}) == 2
    assert all(r["final_settings"]["actual_dtype"] == "float16" for r in data["roles"])
    assert all(r["elapsed_seconds"] >= 0 for r in data["roles"])
    assert len(lib.generations) == 2 and len(lib.loads) == 2


@pytest.mark.parametrize(
    "field",
    [
        'base_url="https://example.com"',
        'api_key="forbidden"',
        'authorization="forbidden"',
        'api_key_env="FORBIDDEN"',
        'authorization_env="FORBIDDEN"',
        "opencode_session=true",
        "timeout=1",
        "max_tokens=512",
    ],
)
def test_qwen_rejects_http_fields_before_execution(local, tmp_path, field):
    lib, _, _ = local
    config = config_for(local, tmp_path)
    config.write_text(config.read_text().replace("[target]\n", f"[target]\n{field}\n"))
    with pytest.raises(ValueError, match="target"):
        load_profile(config)
    assert not lib.imports


@pytest.mark.parametrize(
    "generation",
    [
        {"max_tokens": 10},
        {"stream": False},
        {"n": 1},
        {"num_return_sequences": 2},
        {"past_key_values": []},
        {"max_new_tokens": 0},
        {"max_new_tokens": True},
        {"do_sample": 1},
        {"temperature": 0.5},
        {"do_sample": True, "temperature": 0},
        {"do_sample": True, "top_p": 1.1},
        {"do_sample": True, "top_k": -1},
        {"repetition_penalty": float("nan")},
        {"do_sample": True, "top_k": True},
    ],
)
def test_invalid_generation_parameters_fail_before_import(local, generation):
    lib, kwargs, _ = local
    with pytest.raises(ValueError):
        QwenChat(**kwargs, generation=generation)
    assert not lib.imports


@pytest.mark.parametrize(
    "override",
    [
        {"device": "cpu"},
        {"device": "cuda"},
        {"revision": "main"},
        {"dtype": "auto"},
        {"cache_dir": ""},
        {"system": 42},
        {"model": "../local"},
    ],
)
def test_invalid_local_configuration_fails_before_import(local, override):
    lib, kwargs, _ = local
    with pytest.raises(ValueError):
        QwenChat(**{**kwargs, **override})
    assert not lib.imports


def test_backend_choice_and_http_compatibility(local, tmp_path):
    lib, _, _ = local
    config = config_for(local, tmp_path)
    text = config.read_text()
    for backend in ("other", 12):
        config.write_text(
            text.replace('backend = "qwen"', f"backend = {json.dumps(backend)}")
        )
        with pytest.raises(ValueError, match="backend"):
            load_profile(config)
    for explicit in ("", 'backend="openai"'):
        config.write_text(f"""[target]
{explicit}
base_url="http://localhost:8080/v1"
model="http-model"
[attacker]
base_url="http://localhost:8080/v1"
model="http-model"
[evaluator]
type="contains_text"
criterion="key"
""")
        assert load_profile(config).target.settings()["backend"] == "openai"
    assert not lib.imports


def test_three_role_preflight_and_real_fitd_reuse_one_load_across_base_requests(
    local, tmp_path
):
    lib, kwargs, snapshot = local
    text = Path("examples/fitd-alert-qwen.toml").read_text()
    text = text.replace("Qwen/Qwen2.5-0.5B-Instruct", kwargs["model"])
    text = text.replace("7ae557604adf67be50417f59c2c2f167def9a775", kwargs["revision"])
    text = text.replace("~/.cache/tai/huggingface", str(kwargs["cache_dir"]))
    config = tmp_path / "alert.toml"
    config.write_text(text)
    profile = load_profile(config)
    lib.responses.extend(["READY", "READY", '{"success": false}'])
    report = tmp_path / "preflight.json"
    assert run_preflight(profile, report) == 0
    data = json.loads(report.read_text())
    assert [r["role"] for r in data["roles"]] == ["Target", "Attacker", "Evaluator"]
    assert all(r["final_settings"]["actual_dtype"] == "float16" for r in data["roles"])
    assert len({r["conversation_id"] for r in data["roles"]}) == 3
    for base in (Message("First base"), Message("Second base")):
        lib.responses.extend(["Initial", "L1", "L2", "L3"])
        for response in ("One", "Two", "Three", "Four", "Presented"):
            lib.responses.extend([response, "yes"])
        lib.responses.append('{"success": true}')
        user = profile.create_user(base, tmp_path / "runs")
        user.start()
        assert user.wait(10)
        assert user.status().state is UserState.COMPLETED, user.status().reason
        assert user.status().evaluation.success is True
        assert user.status().progress.evaluator_calls == 1
        assert user.status().progress.max_calls_per_conv == 20
        assert len(lib.requests[-1]) == 2
        sent = json.loads(lib.requests[-1][-1]["content"].split("INPUT_JSON:\n")[1])
        assert sent == {"base_request": base.to_string(), "result": "Presented"}
    assert lib.loads == [("tokenizer", str(snapshot)), ("model", str(snapshot))]
    assert len(lib.generations) == 33
    assert not lib.responses


def test_evaluator_qwen_context_overflow_is_not_truncated_or_scored(local):
    lib, kwargs, _ = local
    lib.input_length = 32760
    chat = QwenChat(**kwargs, generation={"max_new_tokens": 32})
    evaluator = ModelEvaluator(chat, Message("Explicit base"))
    full = "Complete result π\n" * 3000
    with pytest.raises(
        ModelEvaluationError, match="QwenContextError.*no messages were truncated"
    ):
        evaluator.is_success(Message(full))
    assert not lib.generations and not chat.history()
    sent = json.loads(lib.requests[-1][-1]["content"].split("INPUT_JSON:\n")[1])
    assert sent == {"base_request": "Explicit base", "result": full}
