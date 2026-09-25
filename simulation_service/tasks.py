"""Async task lifecycle: submit / query / cancel / result with bounded concurrency."""

import threading

from .models import SimulationParameters, SimulationResult, TaskSnapshot


class TaskManager:
    """Manages a queue of simulation tasks with a concurrency limit."""

    def __init__(self, max_concurrent: int = 2, max_queued: int = 64) -> None:
        self._lock = threading.Lock()
        self._max_concurrent = max_concurrent
        self._max_queued = max_queued

    def submit(self, params: SimulationParameters) -> TaskSnapshot:
        raise NotImplementedError

    def get_status(self, task_id: str) -> TaskSnapshot | None:
        raise NotImplementedError

    def cancel(self, task_id: str) -> TaskSnapshot | None:
        raise NotImplementedError

    def get_result(self, task_id: str) -> SimulationResult | None:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError
