"""Foreground User control and rendering through injected observation/storage ports."""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from pathlib import Path

from core.runs import RunSnapshot, UserState, UserStatus

from .contracts import JournalView, RunHandle, StopRequests
from .failure_capture import Failure
from .lifecycle import POLL_SECONDS, join_admitted

CALL_STARTED = "call_started"


def status_text(status: UserStatus) -> str:
    attacker = (f", Attacker={status.progress.attacker_calls}"
                if status.progress.max_calls_per_conv is not None else "")
    return (
        f"Состояние: {status.state}; завершённые обращения: "
        f"Target={status.progress.target_calls}{attacker}, "
        f"Evaluator={status.progress.evaluator_calls}"
    )


def _evaluation(snapshot: RunSnapshot) -> str:
    result, final = snapshot.result, snapshot.status
    evaluation = final.evaluation
    if result is not None and evaluation is not None and evaluation.result_id == result.id == final.result_id:
        return "Оценка результата: " + ("успех" if evaluation.success else "неуспех")
    return "Оценка результата: отсутствует; нет валидной оценки финального текста"


def show_result(snapshot: RunSnapshot, show: Callable[[str], None]) -> None:
    show(status_text(snapshot.status))
    show(f"Причина завершения: {snapshot.status.reason}")
    if snapshot.result is None:
        show("Результат: отсутствует (None)")
    else:
        text = snapshot.result.to_string()
        show(f"Результат Message {snapshot.result.id} ({len(text)} символов):")
        show(text)
    show(_evaluation(snapshot))


@dataclass
class ProgressDisplay:
    show: Callable[[str], None]
    previous: UserStatus | None = None
    cursor: int = 0

    def initial(self, journal: JournalView, directory: Path) -> None:
        status = journal.status()
        self.show(f"Запуск: {status.run_id}")
        self.show(f"Артефакты: {directory / str(status.run_id)}")
        limit = status.progress.max_calls_per_conv
        policy = (f"Лимит FITD: {limit} библиотечных раундов; это не общий бюджет обращений. "
                  if limit is not None else "Перебор вариантов: до первого успеха или исчерпания стратегии. ")
        self.show(policy + "Ctrl+C — остановить после текущего вызова.")

    def update(self, journal: JournalView) -> None:
        if (status := journal.status()) != self.previous:
            self.show(status_text(status))
            self.previous = status
        for event in journal.events(after=self.cursor):
            self.cursor = event.sequence
            if event.kind == CALL_STARTED:
                data = event.data
                detail = (f"{data['role']} ({data['call_id']})"
                          if isinstance(data, dict) and {"role", "call_id"} <= data.keys()
                          else "диагностические поля скрыты")
                self.show(f"Обращение: {detail}")


def _monitor(handle: RunHandle, display: ProgressDisplay, stop: StopRequests, directory: Path) -> None:
    handle.start()
    journal = handle.journal()
    display.initial(journal, directory)
    done = False
    for _ in count():
        if stop.interrupted and not done:
            handle.stop()
        display.update(journal)
        if done:
            return
        done = handle.wait(POLL_SECONDS)


def _execute(handle: RunHandle, display: ProgressDisplay, stop: StopRequests, directory: Path) -> bool:
    failure = Failure()
    try:
        _monitored_attempt(handle, display, stop, directory, failure=failure)
    finally:
        join_admitted(handle)
    if failure.error is not None:
        display.show(f"Ошибка выполнения: {type(failure.error).__name__}: {failure.error}")
    return failure.error is not None


def _monitored_attempt(
    handle: RunHandle, display: ProgressDisplay, stop: StopRequests, directory: Path, *, failure: Failure,
) -> None:
    with failure:
        _monitor(handle, display, stop, directory)


def _save(handle: RunHandle, show: Callable[[str], None]) -> bool:
    with Failure() as failure:
        handle.save()
    if failure.error is not None:
        show(f"Ошибка сохранения: {type(failure.error).__name__}: {failure.error}")
    return failure.error is not None


def _control(handle: RunHandle, display: ProgressDisplay, stop: StopRequests, directory: Path) -> int:
    execution_error = _execute(handle, display, stop, directory)
    if handle.status().run_id is None:
        return 1
    snapshot = handle.snapshot()
    storage_error = _save(handle, display.show)
    show_result(snapshot, display.show)
    if stop.interrupted or snapshot.status.state is UserState.STOPPED:
        return 130
    return int(execution_error or storage_error or snapshot.status.state is not UserState.COMPLETED)


def control_foreground(handle: RunHandle, display: ProgressDisplay, stop: StopRequests, directory: Path) -> int:
    with Failure() as failure:
        return _control(handle, display, stop, directory)
    display.show(f"Ошибка выполнения: {type(failure.error).__name__}: {failure.error}")
    return 1
