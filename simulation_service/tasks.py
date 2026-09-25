"""Async task lifecycle: submit / query / cancel / result with bounded concurrency.

Concurrency semantics:
  * at most ``max_concurrent`` simulations run at once; extra submissions are
    queued (FIFO) instead of failing;
  * the queue itself is bounded by ``max_queued``; submissions beyond that are
    rejected with RESOURCE_LIMIT_EXCEEDED so resources stay bounded;
  * cancel is idempotent: cancelling a queued task removes it before it runs,
    cancelling a running task sets a flag the solver polls, and cancelling a
    terminal task is a no-op returning the current snapshot.
"""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .errors import ErrorReason
from .models import SimulationParameters, SimulationResult, TaskSnapshot, TaskStatus
from .solver import SolverCancelled, run_simulation, validate_parameters

_TERMINAL = {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}


@dataclass
class _Task:
    task_id: str
    params: SimulationParameters
    status: TaskStatus = TaskStatus.QUEUED
    progress: float = 0.0
    error_reason: ErrorReason | None = None
    error_detail: str | None = None
    result: SimulationResult | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: object | None = None

    def snapshot(self) -> TaskSnapshot:
        return TaskSnapshot(
            task_id=self.task_id,
            status=self.status,
            progress=self.progress,
            error_reason=self.error_reason,
            error_detail=self.error_detail,
        )


class TaskManager:
    """Manages a bounded queue of simulation tasks."""

    def __init__(self, max_concurrent: int = 2, max_queued: int = 64) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._lock = threading.Lock()
        self._max_queued = max_queued
        self._tasks: dict[str, _Task] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix="sim"
        )

    # -- submission ------------------------------------------------------

    def submit(self, params: SimulationParameters) -> TaskSnapshot:
        """Validate and enqueue a simulation.

        Invalid / unstable / over-limit parameter combinations produce a task
        in FAILED state with a distinguishable reason; they never compute.
        """
        task = _Task(task_id=uuid.uuid4().hex, params=params)

        validation = validate_parameters(params)
        if not validation.ok:
            task.status = TaskStatus.FAILED
            task.error_reason = validation.reason
            task.error_detail = validation.detail
            with self._lock:
                self._tasks[task.task_id] = task
            return task.snapshot()

        with self._lock:
            queued = sum(1 for t in self._tasks.values() if t.status == TaskStatus.QUEUED)
            if queued >= self._max_queued:
                task.status = TaskStatus.FAILED
                task.error_reason = ErrorReason.RESOURCE_LIMIT_EXCEEDED
                task.error_detail = f"task queue is full (max_queued={self._max_queued})"
                self._tasks[task.task_id] = task
                return task.snapshot()
            self._tasks[task.task_id] = task
            task.future = self._executor.submit(self._run, task)
            return task.snapshot()

    # -- worker ----------------------------------------------------------

    def _run(self, task: _Task) -> None:
        with self._lock:
            if task.status != TaskStatus.QUEUED:  # cancelled before start
                return
            task.status = TaskStatus.RUNNING

        def should_cancel() -> bool:
            return task.cancel_event.is_set()

        def on_progress(fraction: float) -> None:
            with self._lock:
                task.progress = fraction

        try:
            final_state = run_simulation(task.params, should_cancel, on_progress)
        except SolverCancelled:
            with self._lock:
                task.status = TaskStatus.CANCELLED
                task.error_reason = ErrorReason.CANCELLED
                task.error_detail = "task was cancelled while running"
        except FloatingPointError as exc:
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error_reason = ErrorReason.NUMERICALLY_UNSTABLE
                task.error_detail = str(exc)
        except Exception as exc:  # noqa: BLE001 - mapped to a distinguishable reason
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error_reason = ErrorReason.INTERNAL_ERROR
                task.error_detail = f"{type(exc).__name__}: {exc}"
        else:
            with self._lock:
                task.status = TaskStatus.SUCCEEDED
                task.progress = 1.0
                task.result = SimulationResult(
                    task_id=task.task_id,
                    final_state=[float(v) for v in final_state],
                    final_time=task.params.dt * task.params.num_steps,
                    steps_completed=task.params.num_steps,
                )

    # -- queries ---------------------------------------------------------

    def get_status(self, task_id: str) -> TaskSnapshot | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.snapshot() if task else None

    def get_result(self, task_id: str) -> SimulationResult | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.SUCCEEDED:
                return None
            return task.result

    # -- cancellation ----------------------------------------------------

    def cancel(self, task_id: str) -> TaskSnapshot | None:
        """Cancel a task. Idempotent; terminal tasks are returned unchanged."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if task.status in _TERMINAL:
                return task.snapshot()
            task.cancel_event.set()
            future = task.future
            if task.status == TaskStatus.QUEUED and future is not None:
                if future.cancel():
                    task.status = TaskStatus.CANCELLED
                    task.error_reason = ErrorReason.CANCELLED
                    task.error_detail = "task was cancelled while queued"
                    return task.snapshot()

        # Running (or a queued task the executor already picked up): wait for
        # the worker to observe the cancel flag so the returned snapshot is
        # already in a terminal state.
        if future is not None:
            future.result(timeout=30)
        with self._lock:
            return task.snapshot()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
