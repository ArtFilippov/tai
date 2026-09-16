"""Foreground control and read-only presentation of the public User interface."""

import signal
from collections.abc import Callable
from pathlib import Path
from types import FrameType

from tai import User, UserState, UserStatus


def _status(status: UserStatus) -> str:
    return (
        f"Состояние: {status.state}; завершённые обращения: "
        f"Target={status.progress.target_calls}, "
        f"Attacker={status.progress.attacker_calls}, "
        f"Evaluator={status.progress.evaluator_calls}"
    )


def run_foreground(
    user: User, *, artifacts_dir: Path, redact: Callable[[str], str] = str
) -> int:
    """Run on the main thread; SIGINT requests stop outside the signal handler.

    The handler never takes a User lock or raises through Thread.join. Repeated
    interrupts still let the worker finish its cancellation and artifact writes.
    """
    interrupted = False

    def show(text: str) -> None:
        safe = redact(text)
        print(safe, end="" if safe.endswith("\n") else "\n", flush=True)

    def interrupt(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupted
        interrupted = True

    previous_handler = signal.signal(signal.SIGINT, interrupt)
    started = False
    try:
        user.start()
        started = True
        initial = user.status()
        show(f"Запуск: {initial.run_id}")
        show(f"Артефакты: {artifacts_dir / str(initial.run_id)}")
        show(
            f"Лимит FITD: {initial.progress.max_calls_per_conv} библиотечных раундов; "
            "это не общий бюджет обращений. Ctrl+C — остановить."
        )
        previous = None
        cursor = 0
        while True:
            if interrupted:
                user.stop()
            status = user.status()
            if status != previous:
                show(_status(status))
                previous = status
            for event in user.events(after=cursor):
                cursor = event.sequence
                if event.kind == "call_started":
                    data = event.data
                    detail = (
                        f"{data['role']} ({data['call_id']})"
                        if isinstance(data, dict) and {"role", "call_id"} <= data.keys()
                        else "диагностические поля скрыты"
                    )
                    show(f"Обращение: {detail}")
            if user.wait(0.05):
                break
        final = user.status()
        if final != previous:
            show(_status(final))
        show(f"Причина завершения: {final.reason}")
        result = user.result()
        if result is None:
            show("Результат: отсутствует (None)")
        else:
            show(f"Результат Message {result.id} ({len(result.to_string())} символов):")
            show(result.to_string())
        evaluation = final.evaluation
        if (
            result is not None
            and evaluation is not None
            and evaluation.result_id == result.id == final.result_id
        ):
            show("Оценка результата: " + ("успех" if evaluation.success else "неуспех"))
        else:
            show("Оценка результата: отсутствует; финальный текст не оценивался")
        if interrupted or final.state is UserState.STOPPED:
            return 130
        return 0 if final.state is UserState.COMPLETED else 1
    finally:
        try:
            if started:
                user.stop()
                user.wait()
        finally:
            signal.signal(signal.SIGINT, previous_handler)
