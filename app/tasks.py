"""任务管理器：有界并发执行、排队、取消、状态与结果查询。

语义：
- 最多 max_concurrent 个任务同时执行，其余进入有界队列等待；
  队列满时 submit 抛出 ResourceLimitError。
- cancel 幂等：对排队任务直接标记取消（不会被执行），对运行中任务
  通过取消事件让求解器在安全点退出；对终态任务不产生任何影响。
- 所有状态迁移只在管理器内部发生，查询入口读取同一份状态，保证一致。
"""
from __future__ import annotations

import queue
import threading
import uuid
from typing import Optional

from app.models import (
    ErrorReason,
    SimulationRequest,
    SimulationResult,
    TaskStatus,
    TaskView,
)
from app.solver import (
    CancelledError,
    InvalidParamsError,
    ResourceLimitError,
    StabilityError,
    check_stability,
    run_simulation,
)

_VALIDATION_ERRORS = {
    InvalidParamsError: ErrorReason.INVALID_PARAMS,
    StabilityError: ErrorReason.UNSTABLE,
    ResourceLimitError: ErrorReason.RESOURCE_LIMIT,
}


class _Task:
    def __init__(self, task_id: str, req: SimulationRequest) -> None:
        self.task_id = task_id
        self.req = req
        self.status = TaskStatus.QUEUED
        self.progress = 0.0
        self.error_reason: Optional[ErrorReason] = None
        self.error_detail: Optional[str] = None
        self.final_state: Optional[list[float]] = None
        self.checksum: Optional[float] = None
        self.cancel_event = threading.Event()


class TaskManager:
    def __init__(self, max_concurrent: int = 2, max_queue: int = 16) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent 必须 >= 1")
        self._max_queue = max_queue
        self._lock = threading.Lock()
        self._tasks: dict[str, _Task] = {}
        self._queue: queue.Queue[Optional[_Task]] = queue.Queue()
        self._shutdown = False
        self._running = 0
        self._max_running_observed = 0
        self._workers = [
            threading.Thread(target=self._worker, name=f"sim-worker-{i}", daemon=True)
            for i in range(max_concurrent)
        ]
        for w in self._workers:
            w.start()

    # ---- 提交 ----
    def submit(self, req: SimulationRequest) -> TaskView:
        # 同步校验：非法/不稳定/超资源上限的参数在提交时即被拒绝，
        # 不会创建任务、不会占用队列。
        check_stability(req)
        with self._lock:
            if self._shutdown:
                raise ResourceLimitError("服务正在关闭，拒绝新任务")
            queued = sum(1 for t in self._tasks.values() if t.status == TaskStatus.QUEUED)
            if queued >= self._max_queue:
                raise ResourceLimitError(
                    f"排队任务数已达上限 {self._max_queue}，请稍后重试"
                )
            task = _Task(task_id=uuid.uuid4().hex, req=req)
            self._tasks[task.task_id] = task
        self._queue.put(task)
        return self._view(task)

    # ---- 查询 ----
    def get(self, task_id: str) -> Optional[TaskView]:
        with self._lock:
            task = self._tasks.get(task_id)
            return self._view(task) if task else None

    def result(self, task_id: str) -> Optional[SimulationResult]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return SimulationResult(
                task_id=task.task_id,
                status=task.status,
                error_reason=task.error_reason,
                error_detail=task.error_detail,
                nx=task.req.nx,
                num_steps=task.req.num_steps,
                final_state=task.final_state,
                checksum=task.checksum,
            )

    # ---- 取消（幂等） ----
    def cancel(self, task_id: str) -> Optional[TaskView]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if task.status == TaskStatus.QUEUED:
                # 排队中：直接迁移到终态，worker 取出时会跳过
                task.status = TaskStatus.CANCELLED
                task.error_reason = ErrorReason.CANCELLED
                task.error_detail = "任务在排队时被取消"
            elif task.status == TaskStatus.RUNNING:
                # 运行中：置取消事件，求解器在安全点退出并自行迁移状态
                task.cancel_event.set()
            # 终态任务：幂等，不做任何事
            return self._view(task)

    # ---- 关闭 ----
    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
        for _ in self._workers:
            self._queue.put(None)
        for w in self._workers:
            w.join(timeout=10)

    @property
    def max_running_observed(self) -> int:
        """运行期间观测到的最大并发执行数（用于验证并发上限）。"""
        with self._lock:
            return self._max_running_observed

    # ---- 内部 ----
    def _worker(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                return
            with self._lock:
                if task.status == TaskStatus.CANCELLED:
                    continue  # 排队期间被取消，直接跳过，不重复计算
                task.status = TaskStatus.RUNNING
                self._running += 1
                self._max_running_observed = max(self._max_running_observed, self._running)
            try:
                final = run_simulation(
                    task.req,
                    progress_cb=lambda p: self._set_progress(task, p),
                    should_cancel=task.cancel_event.is_set,
                )
            except CancelledError:
                self._finish(task, TaskStatus.CANCELLED, ErrorReason.CANCELLED,
                           "任务在运行中被取消")
            except tuple(_VALIDATION_ERRORS) as exc:
                self._finish(task, TaskStatus.FAILED,
                             _VALIDATION_ERRORS[type(exc)], str(exc))
            except Exception as exc:  # 含数值发散等内部错误
                self._finish(task, TaskStatus.FAILED, ErrorReason.INTERNAL_ERROR, str(exc))
            else:
                with self._lock:
                    task.final_state = [float(v) for v in final]
                    task.checksum = float(final.sum())
                self._finish(task, TaskStatus.SUCCEEDED, None, None)

    def _set_progress(self, task: _Task, p: float) -> None:
        with self._lock:
            task.progress = p

    def _finish(
        self,
        task: _Task,
        status: TaskStatus,
        reason: Optional[ErrorReason],
        detail: Optional[str],
    ) -> None:
        with self._lock:
            self._running -= 1
            task.status = status
            task.error_reason = reason
            task.error_detail = detail
            if status in (TaskStatus.SUCCEEDED,):
                task.progress = 1.0

    @staticmethod
    def _view(task: _Task) -> TaskView:
        return TaskView(
            task_id=task.task_id,
            status=task.status,
            progress=task.progress,
            error_reason=task.error_reason,
            error_detail=task.error_detail,
        )
