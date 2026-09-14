"""A background User running the pinned FITD strategy."""

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, TextIO
from uuid import uuid4

from tai._profile import FITD_PROFILE
from tai.chats import LLMChat, ModelRole
from tai.evaluators import Evaluator
from tai.messages import Message
from tai.users import Evaluation, Progress, RunEvent, User, UserState, UserStatus


class FITDUser(User):
    """One FITD experiment with one task and at most ten library rounds.

    The presented text is the most recent Target response in the retained
    conversation. This is FITDUser's composition policy, not a User constraint.
    """

    def __init__(
        self,
        target: LLMChat,
        attacker: LLMChat,
        base: Message,
        evaluator: Evaluator,
        *,
        artifacts_dir: str | Path = "runs",
    ) -> None:
        if target is attacker:
            raise ValueError(
                "Target and Attacker must be independent LLMChat instances"
            )
        self._participants: dict[ModelRole, tuple[LLMChat, str]] = {
            "Target": (target, "target_calls"),
            "Attacker": (attacker, "attacker_calls"),
        }
        self._base = base
        self._evaluator = evaluator
        self._artifacts_dir = Path(artifacts_dir)
        self._lock = Lock()
        self._status = UserStatus(
            progress=Progress(max_calls_per_conv=FITD_PROFILE.max_calls_per_conv)
        )
        self._result: Message | None = None
        self._events: list[RunEvent] = []
        self._done = Event()
        self._journal: TextIO | None = None

    def start(self) -> None:
        with self._lock:
            if self._status.state is not UserState.IDLE:
                raise RuntimeError("User has already started")
            self._status = replace(
                self._status, state=UserState.RUNNING, run_id=str(uuid4())
            )
            worker = Thread(target=self._run, name=f"fitd-{self._status.run_id}")
            worker.start()

    def status(self) -> UserStatus:
        with self._lock:
            return self._status

    def result(self) -> Message | None:
        with self._lock:
            return self._result

    def events(self, after: int = 0) -> tuple[RunEvent, ...]:
        with self._lock:
            return tuple(self._events[after:])

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def _emit(self, kind: str, **data: Any) -> None:
        run_id = self.status().run_id
        assert run_id is not None
        event = RunEvent(
            run_id,
            len(self._events) + 1,
            kind,
            datetime.now(UTC).isoformat(),
            json.dumps(data, ensure_ascii=False),
        )
        if self._journal is not None:
            self._journal.write(event.dump() + "\n")
            self._journal.flush()
        with self._lock:
            self._events.append(event)

    def _send(self, role: ModelRole, message: Message) -> Message:
        chat, counter = self._participants[role]
        call_id = str(uuid4())
        self._emit(
            "call_started",
            role=role,
            call_id=call_id,
            input=json.loads(message.dump()),
        )
        try:
            response = chat.send(message)
        except Exception as exc:
            self._emit("call_failed", role=role, call_id=call_id, error=str(exc))
            raise
        # Persist each completed exchange before FITD does any further work.
        self._emit(
            "call_completed",
            role=role,
            call_id=call_id,
            response=json.loads(response.dump()),
        )
        with self._lock:
            progress = self._status.progress
            self._status = replace(
                self._status,
                progress=replace(progress, **{counter: getattr(progress, counter) + 1}),
            )
        return response

    def _publish(self, result: Message) -> None:
        with self._lock:
            if self._result is result:
                return
            self._result = result
            self._status = replace(self._status, result_id=result.id, evaluation=None)
        self._emit("result_published", result=json.loads(result.dump()))

    def _evaluate(self) -> bool:
        result = self.result()
        if result is None:
            raise RuntimeError("FITD requested evaluation without a result")
        self._emit("evaluation_started", result_id=result.id)
        success = self._evaluator.is_success(result)
        if not isinstance(success, bool):
            raise TypeError("Evaluator.is_success must return bool")
        evaluation = Evaluation(result.id, success)
        self._emit("evaluation_completed", **asdict(evaluation))
        with self._lock:
            self._status = replace(self._status, evaluation=evaluation)
        return success

    def _run(self) -> None:
        run_id = self.status().run_id
        assert run_id is not None
        run_dir = self._artifacts_dir / run_id
        state, reason = UserState.COMPLETED, "round_limit"
        settings: dict[str, Any] | None = None
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            with (run_dir / "events.jsonl").open("w", encoding="utf-8") as journal:
                self._journal = journal
                try:
                    from tai._fitd import GARAK_REVISION, run_fitd

                    settings = {
                        "garak_revision": GARAK_REVISION,
                        **asdict(FITD_PROFILE),
                        "base": json.loads(self._base.dump()),
                        **{
                            role: self._chat_settings(chat)
                            for role, (chat, _) in self._participants.items()
                        },
                    }
                    self._emit("run_started", settings=settings)
                    reason = run_fitd(
                        self._base, self._send, self._publish, self._evaluate
                    )
                except Exception as exc:
                    state, reason = UserState.FAILED, f"{type(exc).__name__}: {exc}"
                    self._emit("run_error", error=reason)
                result = self.result()
                if result is not None:
                    (run_dir / "result.txt").write_text(
                        result.to_string(), encoding="utf-8"
                    )
                self._emit("run_finished", state=state.value, reason=reason)
                metadata = {
                    **asdict(self.status()),
                    "state": state.value,
                    "reason": reason,
                    "result": json.loads(result.dump()) if result is not None else None,
                    "settings": settings,
                }
                (run_dir / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        except Exception as exc:
            state, reason = UserState.FAILED, f"{type(exc).__name__}: {exc}"
        finally:
            self._journal = None
            with self._lock:
                self._status = replace(self._status, state=state, reason=reason)
            self._done.set()

    @staticmethod
    def _chat_settings(chat: LLMChat) -> dict[str, Any]:
        return {
            "model": chat.model,
            "initial_history": [
                {"role": turn.role, "message": json.loads(turn.message.dump())}
                for turn in chat.history()
            ],
        }
