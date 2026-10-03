"""任务管理器：有界并发执行、排队、取消、状态与结果查询，并支持批量扫参。

语义：
- 最多 max_concurrent 个任务同时执行，其余进入有界队列等待；
  队列满时 submit 抛出 ResourceLimitError。批量扫参展开出的每一条组合都是
  普通任务，进入同一个队列、受同一个并发/队列上限约束，不享受任何特权。
- cancel 幂等：对排队任务直接标记取消（不会被执行），对运行中任务
  通过取消事件让求解器在安全点退出；对终态任务不产生任何影响。
- 整批取消 = 对批内每一条应用同一条取消规则；单条失败/取消不影响其他组合。
- 所有状态迁移只在管理器内部发生，查询入口读取同一份状态，保证一致。
"""
from __future__ import annotations

import queue
import threading
import uuid
from typing import Optional

from app.batch import BatchValidationError, validate_batch
from app.models import (
    BatchItemResult,
    BatchItemView,
    BatchRequest,
    BatchResult,
    BatchStatus,
    BatchView,
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
    def __init__(
        self,
        task_id: str,
        req: SimulationRequest,
        batch_id: Optional[str] = None,
        index: Optional[int] = None,
        axis_values: Optional[dict[str, float | int]] = None,
    ) -> None:
        self.task_id = task_id
        self.req = req
        self.status = TaskStatus.QUEUED
        self.progress = 0.0
        self.error_reason: Optional[ErrorReason] = None
        self.error_detail: Optional[str] = None
        self.final_state: Optional[list[float]] = None
        self.checksum: Optional[float] = None
        self.cancel_event = threading.Event()
        # 批次归属信息（单任务为 None）
        self.batch_id = batch_id
        self.index = index
        self.axis_values = axis_values


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

    # ---- 提交（单任务） ----
    def submit(self, req: SimulationRequest) -> TaskView:
        # 同步校验：非法/不稳定/超资源上限的参数在提交时即被拒绝，
        # 不会创建任务、不会占用队列。
        check_stability(req)
        task = _Task(task_id=uuid.uuid4().hex, req=req)
        with self._lock:
            if self._shutdown:
                raise ResourceLimitError("服务正在关闭，拒绝新任务")
            self._admit(1)  # 排队容量检查（与批次共用同一条容量规则）
            self._tasks[task.task_id] = task
            self._queue.put(task)  # 持锁入队，确保容量检查与入队是原子的
        return self._view(task)

    # ---- 提交（批量扫参） ----
    def submit_batch(self, request: BatchRequest) -> BatchView:
        # 提交前整批原子校验：展开方式不合法、组合数超限、任意一条组合
        # 合法性/资源/稳定性不通过，都会在这里抛出，且不创建任何任务。
        combos = validate_batch(request)
        batch_id = uuid.uuid4().hex
        new_tasks: list[_Task] = []
        for index, (params, axis_values) in enumerate(combos):
            new_tasks.append(
                _Task(
                    task_id=uuid.uuid4().hex,
                    req=SimulationRequest(**params),
                    batch_id=batch_id,
                    index=index,
                    axis_values=dict(axis_values),
                )
            )
        with self._lock:
            if self._shutdown:
                raise ResourceLimitError("服务正在关闭，拒绝新批次")
            # 整批一次性申请排队名额：容量不足则整批拒绝，绝不部分受理
            self._admit(len(new_tasks))
            for task in new_tasks:
                self._tasks[task.task_id] = task
                self._queue.put(task)
        return self.get_batch(batch_id)  # type: ignore[return-value]

    def _admit(self, n_new: int) -> None:
        """调用方持锁：检查排队容量是否还能容纳 n_new 个新任务。"""
        queued = sum(1 for t in self._tasks.values() if t.status == TaskStatus.QUEUED)
        if queued + n_new > self._max_queue:
            raise ResourceLimitError(
                f"排队容量不足：当前排队 {queued} 个，本次需要 {n_new} 个名额，"
                f"队列上限 {self._max_queue}；批次不会部分受理，请稍后重试或收窄扫描轴"
            )

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
            self._cancel_locked(task)
            return self._view(task)

    # ---- 批量查询 ----
    def get_batch(self, batch_id: str) -> Optional[BatchView]:
        with self._lock:
            tasks = self._batch_tasks_locked(batch_id)
            if not tasks:
                return None
            return self._batch_view_locked(batch_id, tasks)

    def batch_result(self, batch_id: str) -> Optional[BatchResult]:
        with self._lock:
            tasks = self._batch_tasks_locked(batch_id)
            if not tasks:
                return None
            status = self._aggregate_status_locked(tasks)
            items = [self._batch_item_result_locked(t) for t in tasks]
            return BatchResult(
                batch_id=batch_id,
                status=status,
                total=len(tasks),
                items=items,
            )

    # ---- 批量取消 ----
    def cancel_batch(self, batch_id: str) -> Optional[BatchView]:
        """整批取消：逐条应用单任务取消规则；幂等，终态组合不受影响。"""
        with self._lock:
            tasks = self._batch_tasks_locked(batch_id)
            if not tasks:
                return None
            for task in tasks:
                self._cancel_locked(task)
            return self._batch_view_locked(batch_id, tasks)

    def cancel_batch_item(
        self, batch_id: str, index: int
    ) -> Optional[BatchItemView]:
        """只取消批次中的某一条；幂等，不影响同批其他组合。"""
        with self._lock:
            tasks = self._batch_tasks_locked(batch_id)
            if not tasks or index < 0 or index >= len(tasks):
                return None
            task = tasks[index]
            self._cancel_locked(task)
            return self._batch_item_view_locked(task)

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
    @staticmethod
    def _cancel_locked(task: "_Task") -> None:
        """调用方持锁：单条取消原语，对任意状态幂等。"""
        if task.status == TaskStatus.QUEUED:
            # 排队中：直接迁移到终态，worker 取出时会跳过
            task.status = TaskStatus.CANCELLED
            task.error_reason = ErrorReason.CANCELLED
            task.error_detail = "任务在排队时被取消"
        elif task.status == TaskStatus.RUNNING:
            # 运行中：置取消事件，求解器在安全点退出并自行迁移状态
            task.cancel_event.set()
        # 终态任务：幂等，不做任何事

    def _batch_tasks_locked(self, batch_id: str) -> list[_Task]:
        """调用方持锁：返回批内全部任务（按下标排序，下标即展开顺序）。"""
        tasks = [t for t in self._tasks.values() if t.batch_id == batch_id]
        tasks.sort(key=lambda t: t.index)
        return tasks

    @staticmethod
    def _aggregate_status_locked(tasks: list[_Task]) -> BatchStatus:
        queued = sum(1 for t in tasks if t.status == TaskStatus.QUEUED)
        running = sum(1 for t in tasks if t.status == TaskStatus.RUNNING)
        if queued:
            return BatchStatus.PENDING
        if running:
            return BatchStatus.RUNNING
        return BatchStatus.COMPLETED

    @staticmethod
    def _params_locked(task: _Task) -> dict:
        # 与提交时的 SimulationRequest 完全一致的字段集，保证「对号入座」
        return task.req.model_dump(mode="json")

    def _batch_item_view_locked(self, task: _Task) -> BatchItemView:
        return BatchItemView(
            index=task.index,  # type: ignore[arg-type]
            task_id=task.task_id,
            status=task.status,
            progress=task.progress,
            error_reason=task.error_reason,
            error_detail=task.error_detail,
            params=self._params_locked(task),
            axis_values=dict(task.axis_values or {}),
        )

    def _batch_item_result_locked(self, task: _Task) -> BatchItemResult:
        return BatchItemResult(
            index=task.index,  # type: ignore[arg-type]
            task_id=task.task_id,
            status=task.status,
            error_reason=task.error_reason,
            error_detail=task.error_detail,
            nx=task.req.nx,
            num_steps=task.req.num_steps,
            final_state=task.final_state,
            checksum=task.checksum,
            params=self._params_locked(task),
            axis_values=dict(task.axis_values or {}),
        )

    def _batch_view_locked(self, batch_id: str, tasks: list[_Task]) -> BatchView:
        counts = {s: 0 for s in TaskStatus}
        progress_sum = 0.0
        items: list[BatchItemView] = []
        for task in tasks:
            counts[task.status] += 1
            # 终态但失败/取消的组合没有完整进度，按 1.0 计入均值，
            # 使总进度在整批结束时必然为 1.0；进行中按求解器自报进度。
            progress_sum += 1.0 if task.status in (
                TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED
            ) else task.progress
            items.append(self._batch_item_view_locked(task))
        return BatchView(
            batch_id=batch_id,
            status=self._aggregate_status_locked(tasks),
            total=len(tasks),
            queued=counts[TaskStatus.QUEUED],
            running=counts[TaskStatus.RUNNING],
            succeeded=counts[TaskStatus.SUCCEEDED],
            failed=counts[TaskStatus.FAILED],
            cancelled=counts[TaskStatus.CANCELLED],
            progress=progress_sum / len(tasks),
            items=items,
        )

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
