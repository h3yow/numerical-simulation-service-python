"""批量扫参：在基础参数上按扫描轴做笛卡尔积展开，复用同一套执行容量与排队规则。

提交规则（任一不满足则整批在提交阶段被拒绝，不会受理任何一条）：
- 扫描轴字段必须是可扫描字段，values 非空且轴内无重复取值；
- 展开得到的组合数不得超过上限；
- 展开后总计算量 sum(nx*num_steps) 不得超过上限；
- 每个组合都必须通过 check_stability（非法/超资源/不稳定原因可区分）。
"""
from __future__ import annotations

import itertools
import math
import threading
import uuid
from typing import Optional

from pydantic import ValidationError

from app.models import (
    BatchItemResult,
    BatchItemView,
    BatchRequest,
    BatchResultView,
    BatchStatus,
    BatchView,
    ErrorReason,
    SCANNABLE_FIELDS,
    SimulationRequest,
    TaskStatus,
    TERMINAL_STATES,
)
from app.solver import (
    InvalidParamsError,
    ResourceLimitError,
    StabilityError,
    check_stability,
)
from app.tasks import TaskManager

#: 单批次组合数上限
DEFAULT_MAX_COMBINATIONS = 256
#: 单批次总计算量上限 sum(nx*num_steps)
DEFAULT_MAX_TOTAL_WORK = 200_000_000

#: check_stability 抛出的异常到可区分原因的映射
_REASON_BY_ERROR = {
    InvalidParamsError: ErrorReason.INVALID_PARAMS,
    StabilityError: ErrorReason.UNSTABLE,
    ResourceLimitError: ErrorReason.RESOURCE_LIMIT,
}
_HTTP_BY_REASON = {
    ErrorReason.INVALID_PARAMS: 400,
    ErrorReason.UNSTABLE: 422,
    ErrorReason.RESOURCE_LIMIT: 429,
}


class BatchValidationError(Exception):
    """批量请求在提交阶段被拒绝。携带可区分原因与逐组合违规明细。"""

    def __init__(
        self,
        reason: ErrorReason,
        detail: str,
        violations: Optional[list[dict]] = None,
    ) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.violations = violations or []
        self.http_status = _HTTP_BY_REASON[reason]


def _value_key(v: object) -> tuple[str, object]:
    """把轴取值规范化为可比较的去重键。"""
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        # 枚举在 JSON 入口是字符串；其余类型（list/dict/None 等）一律不支持
        raise BatchValidationError(
            ErrorReason.INVALID_PARAMS,
            f"扫描轴取值类型不支持: {v!r}（只接受数值或字符串）",
        )
    if isinstance(v, (int, float)):
        f = float(v)
        if not math.isfinite(f):
            raise BatchValidationError(
                ErrorReason.INVALID_PARAMS,
                f"扫描轴取值必须是有限值: {v!r}",
            )
        return ("num", f)
    return ("str", v)


def _expand_and_validate(
    req: BatchRequest,
    max_combinations: int,
    max_total_work: int,
) -> list[tuple[dict, SimulationRequest]]:
    """展开笛卡尔积并在提交期完成全部校验。

    返回 [(params_json_dict, SimulationRequest), ...]，按轴顺序稳定排序。
    任何一条不合法都抛 BatchValidationError（整批拒绝，不做部分受理）。
    """
    # 1) 轴结构：字段可扫描、字段不重复、轴内取值不重复
    seen_fields: set[str] = set()
    normalized_axes: list[tuple[str, list]] = []
    for axis in req.axes:
        if axis.field not in SCANNABLE_FIELDS:
            raise BatchValidationError(
                ErrorReason.INVALID_PARAMS,
                f"扫描轴字段 {axis.field!r} 不可扫描，"
                f"允许的字段: {', '.join(SCANNABLE_FIELDS)}",
            )
        if axis.field in seen_fields:
            raise BatchValidationError(
                ErrorReason.INVALID_PARAMS,
                f"扫描轴字段 {axis.field!r} 出现了多条，每个字段只能指定一条轴",
            )
        seen_fields.add(axis.field)
        if len(axis.values) == 0:
            raise BatchValidationError(
                ErrorReason.INVALID_PARAMS,
                f"扫描轴 {axis.field} 的 values 不能为空",
            )
        keys: set[tuple[str, object]] = set()
        for v in axis.values:
            key = _value_key(v)
            if key in keys:
                raise BatchValidationError(
                    ErrorReason.INVALID_PARAMS,
                    f"扫描轴 {axis.field} 出现重复取值: {v!r}，轴内取值必须唯一",
                )
            keys.add(key)
        normalized_axes.append((axis.field, list(axis.values)))

    # 2) 组合数上限（在校验完轴结构后、展开前即可确定）
    n_combinations = 1
    for _, values in normalized_axes:
        n_combinations *= len(values)
    if n_combinations > max_combinations:
        raise BatchValidationError(
            ErrorReason.RESOURCE_LIMIT,
            f"展开后组合数 {n_combinations} 超过单批上限 {max_combinations}，"
            f"请收窄扫描轴或拆分批次",
        )

    # 3) 逐组合构造 + 提交期数值/资源/稳定性校验（含 diffusion=0 等边界）
    base_dict = req.base.model_dump()
    combos: list[tuple[dict, SimulationRequest]] = []
    violations: list[dict] = []
    value_lists = [values for _, values in normalized_axes]
    field_names = [name for name, _ in normalized_axes]
    for index, picked in enumerate(itertools.product(*value_lists)):
        params = dict(base_dict)
        params.update(dict(zip(field_names, picked)))
        try:
            sim_req = SimulationRequest.model_validate(params)
        except ValidationError as exc:
            violations.append({
                "index": index,
                "params": params,
                "error_reason": ErrorReason.INVALID_PARAMS.value,
                "detail": f"参数类型/取值不合法: {exc.errors()[0]['msg']}",
            })
            continue
        try:
            check_stability(sim_req)
        except tuple(_REASON_BY_ERROR) as exc:
            violations.append({
                "index": index,
                "params": sim_req.model_dump(mode="json"),
                "error_reason": _REASON_BY_ERROR[type(exc)].value,
                "detail": str(exc),
            })
            continue
        combos.append((sim_req.model_dump(mode="json"), sim_req))

    if violations:
        # 整批拒绝；HTTP 层原因取遍历顺序中第一条违规的原因，明细逐条返回
        first = violations[0]
        raise BatchValidationError(
            ErrorReason(first["error_reason"]),
            f"{len(violations)}/{n_combinations} 个组合未通过提交期校验，"
            f"整批不予受理；首条：{first['detail']}",
            violations=violations,
        )

    # 4) 整批总计算量上限
    total_work = sum(sim_req.nx * sim_req.num_steps for _, sim_req in combos)
    if total_work > max_total_work:
        raise BatchValidationError(
            ErrorReason.RESOURCE_LIMIT,
            f"展开后总计算量 sum(nx*num_steps)={total_work} "
            f"超过单批上限 {max_total_work}，请收窄扫描轴或拆分批次",
        )

    return combos


class _Item:
    """批次内的一条组合。未派发时状态由 BatchManager 本地维护。"""

    def __init__(self, item_id: str, params: dict, req: SimulationRequest) -> None:
        self.item_id = item_id
        self.params = params
        self.req = req
        self.task_id: Optional[str] = None
        self.dispatched = False
        self.status = TaskStatus.QUEUED
        self.progress = 0.0
        self.error_reason: Optional[ErrorReason] = None
        self.error_detail: Optional[str] = None
        self.final_state: Optional[list[float]] = None
        self.checksum: Optional[float] = None


class _Batch:
    def __init__(self, batch_id: str, items: list[_Item]) -> None:
        self.batch_id = batch_id
        self.items = items
        self.abort = threading.Event()
        self.cancelled = False  # 是否发生过整批取消（终止后续派发）


class BatchManager:
    def __init__(
        self,
        task_manager: TaskManager,
        max_combinations: int = DEFAULT_MAX_COMBINATIONS,
        max_total_work: int = DEFAULT_MAX_TOTAL_WORK,
    ) -> None:
        self._tm = task_manager
        self._max_combinations = max_combinations
        self._max_total_work = max_total_work
        self._lock = threading.Lock()
        self._batches: dict[str, _Batch] = {}
        self._threads: list[threading.Thread] = []
        self._closed = False

    # ---- 提交 ----
    def submit(self, req: BatchRequest) -> BatchView:
        # 提交期原子校验：结构 / 组合数 / 逐组合稳定性与资源 / 总工作量，
        # 全部通过才受理，任何一条不合法都整批拒绝。
        combos = _expand_and_validate(req, self._max_combinations, self._max_total_work)
        with self._lock:
            if self._closed:
                raise BatchValidationError(
                    ErrorReason.RESOURCE_LIMIT, "批量服务正在关闭，拒绝新批次"
                )
            batch_id = uuid.uuid4().hex
            items = [
                _Item(item_id=f"{i:04d}", params=params, req=sim_req)
                for i, (params, sim_req) in enumerate(combos)
            ]
            batch = _Batch(batch_id=batch_id, items=items)
            self._batches[batch_id] = batch
            t = threading.Thread(
                target=self._dispatch_loop,
                args=(batch,),
                name=f"batch-dispatch-{batch_id[:8]}",
                daemon=True,
            )
            self._threads.append(t)
            t.start()
            return self._build_view_locked(batch)

    def get(self, batch_id: str) -> Optional[BatchView]:
        raise NotImplementedError

    def result(self, batch_id: str) -> Optional[BatchResultView]:
        raise NotImplementedError

    def cancel(self, batch_id: str) -> Optional[BatchView]:
        raise NotImplementedError

    def cancel_item(self, batch_id: str, item_id: str) -> Optional[BatchView]:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError

    # ---- 派发：复用 TaskManager 的有界并发与排队，不绕过任何上限 ----
    def _dispatch_loop(self, batch: _Batch) -> None:
        while True:
            with self._lock:
                item = next((it for it in batch.items if not it.dispatched), None)
                if item is None:
                    return
                if batch.cancelled or batch.abort.is_set():
                    # 整批取消 / 服务关闭：仍未派发的直接终止，不再占用容量
                    item.dispatched = True
                    item.status = TaskStatus.CANCELLED
                    item.error_reason = ErrorReason.CANCELLED
                    item.error_detail = "批次已取消，组合在排队派发前终止"
                    continue
            # 等待共享队列出现空位；满了就排队，绝不绕过并发上限，也不失败
            if not self._tm.wait_for_space(
                should_abort=lambda: batch.abort.is_set() or batch.cancelled
            ):
                with self._lock:
                    item.dispatched = True
                    item.status = TaskStatus.CANCELLED
                    item.error_reason = ErrorReason.CANCELLED
                    item.error_detail = "批次已取消，组合在排队派发前终止"
                continue
            with self._lock:
                if batch.abort.is_set() or batch.cancelled:
                    item.dispatched = True
                    item.status = TaskStatus.CANCELLED
                    item.error_reason = ErrorReason.CANCELLED
                    item.error_detail = "批次已取消，组合在排队派发前终止"
                    continue
                try:
                    view = self._tm.submit(item.req)
                except ResourceLimitError:
                    # 空位可能被并发的单任务提交占用：回到等待，绝不丢弃
                    item.dispatched = False
                    continue
                item.dispatched = True
                item.task_id = view.task_id
                item.status = view.status

    # ---- 查询 ----
    def get(self, batch_id: str) -> Optional[BatchView]:
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                return None
            self._sync_locked(batch)
            return self._build_view_locked(batch)

    def result(self, batch_id: str) -> Optional[BatchResultView]:
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                return None
            self._sync_locked(batch)
            item_views = [self._build_item_result_locked(it) for it in batch.items]
            counts = self._counts_locked(batch)
            return BatchResultView(
                batch_id=batch.batch_id,
                status=self._aggregate_status_locked(batch),
                total=len(batch.items),
                counts=counts,
                items=item_views,
            )

    # ---- 取消（幂等） ----
    def cancel(self, batch_id: str) -> Optional[BatchView]:
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                return None
            self._sync_locked(batch)
            # 已到终态的批次：无效果，不回退状态
            if all(it.status in TERMINAL_STATES for it in batch.items):
                return self._build_view_locked(batch)
            batch.cancelled = True
            batch.abort.set()  # 唤醒可能在等待空位的派发线程
            for it in batch.items:
                self._cancel_one_locked(batch, it)
            self._sync_locked(batch)
            return self._build_view_locked(batch)

    def cancel_item(self, batch_id: str, item_id: str) -> Optional[BatchView]:
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                return None
            item = next((it for it in batch.items if it.item_id == item_id), None)
            if item is None:
                return None
            self._sync_locked(batch)
            self._cancel_one_locked(batch, item)
            self._sync_locked(batch)
            return self._build_view_locked(batch)

    # ---- 关闭 ----
    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            for batch in self._batches.values():
                batch.abort.set()
        for t in self._threads:
            t.join(timeout=10)

    # ---- 内部 ----
    def _cancel_one_locked(self, batch: _Batch, item: _Item) -> None:
        if not item.dispatched or item.task_id is None:
            # 尚未派发到执行器：本地标记终态，派发线程不会再送出
            if item.status in TERMINAL_STATES:
                return
            item.dispatched = True
            item.status = TaskStatus.CANCELLED
            item.error_reason = ErrorReason.CANCELLED
            item.error_detail = "组合在排队派发前被取消"
            return
        # 已派发：委托给 TaskManager，排队/运行/终态语义与单任务完全一致
        self._tm.cancel(item.task_id)
        self._copy_task_state_locked(item)

    def _sync_locked(self, batch: _Batch) -> None:
        """从 TaskManager 同步已派发条目的最新状态/进度/结果（读取同一份状态）。"""
        for it in batch.items:
            if it.dispatched and it.task_id is not None:
                self._copy_task_state_locked(it)

    def _copy_task_state_locked(self, item: _Item) -> None:
        view = self._tm.get(item.task_id)
        if view is None:
            return
        item.status = view.status
        item.progress = view.progress
        item.error_reason = view.error_reason
        item.error_detail = view.error_detail
        if view.status in TERMINAL_STATES:
            res = self._tm.result(item.task_id)
            if res is not None:
                item.final_state = res.final_state
                item.checksum = res.checksum

    @staticmethod
    def _counts_locked(batch: _Batch) -> dict[str, int]:
        counts = {s.value: 0 for s in TaskStatus}
        for it in batch.items:
            counts[it.status.value] += 1
        return counts

    @staticmethod
    def _aggregate_status_locked(batch: _Batch) -> BatchStatus:
        statuses = [it.status for it in batch.items]
        terminal = [s for s in statuses if s in TERMINAL_STATES]
        if len(terminal) < len(statuses):
            # 仍有未终态条目
            if not any(s == TaskStatus.RUNNING for s in statuses) and not terminal:
                return BatchStatus.QUEUED
            return BatchStatus.RUNNING
        n_succ = statuses.count(TaskStatus.SUCCEEDED)
        n_canc = statuses.count(TaskStatus.CANCELLED)
        n_fail = statuses.count(TaskStatus.FAILED)
        if n_succ == len(statuses):
            return BatchStatus.SUCCEEDED
        if n_succ == 0 and n_canc > 0 and n_fail == 0:
            return BatchStatus.CANCELLED
        if n_canc == 0 and n_fail > 0 and n_succ == 0:
            return BatchStatus.FAILED
        if n_canc > 0:
            return BatchStatus.PARTIAL_CANCELLED
        return BatchStatus.PARTIAL_FAILED

    def _build_view_locked(self, batch: _Batch) -> BatchView:
        counts = self._counts_locked(batch)
        n_done = sum(
            counts[s.value] for s in TERMINAL_STATES
        )
        total = len(batch.items)
        return BatchView(
            batch_id=batch.batch_id,
            status=self._aggregate_status_locked(batch),
            progress=(n_done / total) if total else 1.0,
            total=total,
            counts=counts,
            items=[self._build_item_view_locked(it) for it in batch.items],
        )

    @staticmethod
    def _build_item_view_locked(it: _Item) -> BatchItemView:
        return BatchItemView(
            item_id=it.item_id,
            task_id=it.task_id,
            params=it.params,
            status=it.status,
            progress=it.progress,
            error_reason=it.error_reason,
            error_detail=it.error_detail,
        )

    @staticmethod
    def _build_item_result_locked(it: _Item) -> BatchItemResult:
        return BatchItemResult(
            item_id=it.item_id,
            task_id=it.task_id,
            params=it.params,
            status=it.status,
            progress=it.progress,
            error_reason=it.error_reason,
            error_detail=it.error_detail,
            nx=it.req.nx,
            num_steps=it.req.num_steps,
            final_state=it.final_state,
            checksum=it.checksum,
        )
