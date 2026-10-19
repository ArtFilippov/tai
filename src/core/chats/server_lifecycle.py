"""Coordinate admission, lazy startup, and closure of one shared vLLM process."""

from abc import ABC
from collections.abc import Callable
from concurrent.futures import Future
from threading import Condition, Event, Thread

class ServerLifecycle(ABC):
    """Share a single startup result without locking admitted HTTP exchanges."""

    def __init__(self) -> None:
        self.state: str = "not_started"
        super().__init__()
        self.condition: Condition = Condition()
        self.closing: Event = Event()
        self.ready: Future[None] = Future()
        self.threads: list[Thread] = []
        self.active: int = 0

    def start(self, invoke: Callable[[], None]) -> None:
        with self.condition:
            if not self.threads:
                self.state = "starting"
                self.threads.append(Thread(target=invoke, name="tai-vllm-start"))
                self.threads[0].start()

    def started(self) -> None:
        self.ready.set_result(None)
        self.notify()

    def failed(self, error: Exception) -> None:
        self.ready.set_exception(error)
        self.notify()

    def notify(self) -> None:
        with self.condition:
            self.condition.notify_all()

    def wait_ready(self) -> bool:
        with self.condition:
            return self.condition.wait_for(self.ready.done, 0.02)

    def admit(self) -> bool:
        with self.condition:
            if self.closing.is_set():
                return False
            self.active += 1
            return True

    def release(self) -> None:
        with self.condition:
            self.active -= 1
            self.condition.notify_all()

    def request_close(self) -> None:
        self.closing.set()

    def join(self) -> None:
        with self.condition:
            _ = self.condition.wait_for(lambda: not self.active)
        for thread in self.threads:
            thread.join()
