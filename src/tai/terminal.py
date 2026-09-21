"""Foreground control of User; progress and diagnostics come from its journal."""

import signal
from collections.abc import Callable
from pathlib import Path
from types import FrameType

from tai import FITDExperiment, UserState, UserStatus
from tai.storage import capture_run, save_run


def _status(status: UserStatus) -> str:
    return (
        f"Состояние: {status.state}; завершённые обращения: "
        f"Target={status.progress.target_calls}, "
        f"Attacker={status.progress.attacker_calls}, "
        f"Evaluator={status.progress.evaluator_calls}"
    )


def run_foreground(
    user: FITDExperiment, *, artifacts_dir: Path, redact: Callable[[str], str] = str
) -> int:
    """Run on the main thread; SIGINT requests stop outside the signal handler.

    The handler never takes a User lock or raises through Thread.join. Repeated
    interrupts still let the worker finish its current call, then save externally.
    """
    interrupted = False

    def show(text: str) -> None:
        safe = redact(text)
        print(safe, end="" if safe.endswith("\n") else "\n", flush=True)

    def interrupt(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupted
        interrupted = True

    previous_handler = signal.signal(signal.SIGINT, interrupt)
    try:
        execution_error = False
        try:
            user.start()
            journal = user.journal
            initial = journal.status()
            show(f"Запуск: {initial.run_id}")
            show(f"Артефакты: {artifacts_dir / str(initial.run_id)}")
            show(
                f"Лимит FITD: {initial.progress.max_calls_per_conv} "
                "библиотечных раундов; это не общий бюджет обращений. "
                "Ctrl+C — остановить после текущего вызова."
            )
            previous = None
            cursor = 0
            done = False
            while True:
                if interrupted and not done:
                    user.stop()
                status = journal.status()
                if status != previous:
                    show(_status(status))
                    previous = status
                for event in journal.events(after=cursor):
                    cursor = event.sequence
                    if event.kind == "call_started":
                        data = event.data
                        detail = (
                            f"{data['role']} ({data['call_id']})"
                            if isinstance(data, dict)
                            and {"role", "call_id"} <= data.keys()
                            else "диагностические поля скрыты"
                        )
                        show(f"Обращение: {detail}")
                if done:
                    break
                done = user.wait(0.05)
        except Exception as exc:
            execution_error = True
            show(f"Ошибка выполнения: {type(exc).__name__}: {exc}")
        finally:
            # start() can raise after admitting work. Always join it before saving.
            if user.status().run_id is not None:
                try:
                    if not user.wait(0):
                        user.stop()
                finally:
                    user.wait()
        if user.status().run_id is None:
            return 1
        snapshot = user.snapshot()
        final, result = snapshot.status, snapshot.result
        storage_error = False
        try:
            record = capture_run(user, journal=user.journal)
            save_run(record, artifacts_dir)
        except Exception as exc:
            storage_error = True
            show(f"Ошибка сохранения: {type(exc).__name__}: {exc}")
        show(_status(final))
        show(f"Причина завершения: {final.reason}")
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
        return int(
            execution_error or storage_error or final.state is not UserState.COMPLETED
        )
    except Exception as exc:
        show(f"Ошибка выполнения: {type(exc).__name__}: {exc}")
        return 1
    finally:
        signal.signal(signal.SIGINT, previous_handler)
