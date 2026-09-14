"""Foreground control and read-only presentation of the public User interface."""

import signal
from pathlib import Path
from types import FrameType

from tai import User, UserState, UserStatus


def _status(status: UserStatus) -> None:
    print(
        f"Состояние: {status.state}; завершённые обращения: "
        f"Target={status.progress.target_calls}, "
        f"Attacker={status.progress.attacker_calls}",
        flush=True,
    )


def run_foreground(user: User, *, artifacts_dir: Path) -> int:
    """Run on the main thread; SIGINT requests stop outside the signal handler.

    The handler never takes a User lock or raises through Thread.join. Repeated
    interrupts still let the worker finish its cancellation and artifact writes.
    """
    interrupted = False

    def interrupt(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupted
        interrupted = True

    previous_handler = signal.signal(signal.SIGINT, interrupt)
    started = False
    try:
        user.start()
        started = True
        initial = user.status()
        print(f"Запуск: {initial.run_id}", flush=True)
        print(f"Артефакты: {artifacts_dir / str(initial.run_id)}", flush=True)
        print(
            f"Лимит FITD: {initial.progress.max_calls_per_conv} библиотечных раундов; "
            "это не общий бюджет обращений. Ctrl+C — остановить.",
            flush=True,
        )
        previous = None
        cursor = 0
        while True:
            if interrupted:
                user.stop()
            status = user.status()
            if status != previous:
                _status(status)
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
                    print(f"Обращение: {detail}", flush=True)
            if user.wait(0.05):
                break
        final = user.status()
        if final != previous:
            _status(final)
        print(f"Причина завершения: {final.reason}", flush=True)
        result = user.result()
        if result is None:
            print("Результат: отсутствует (None)", flush=True)
        else:
            print(
                f"Результат Message {result.id} ({len(result.to_string())} символов):",
                flush=True,
            )
            text = result.to_string()
            print(text, end="" if text.endswith("\n") else "\n", flush=True)
        evaluation = final.evaluation
        if (
            result is not None
            and evaluation is not None
            and evaluation.result_id == result.id == final.result_id
        ):
            print(
                "Оценка результата: " + ("успех" if evaluation.success else "неуспех"),
                flush=True,
            )
        else:
            print(
                "Оценка результата: отсутствует; финальный текст не оценивался",
                flush=True,
            )
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
