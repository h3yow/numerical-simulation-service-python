"""批量扫参：扫描轴展开（笛卡尔积）与提交前的整批原子校验。

纯逻辑模块，不接触线程与任务队列；TaskManager 负责把通过校验的组合
作为普通任务入队，从而与单任务共用同一套并发与排队规则。

展开规则：
- 按 axes 的顺序对各轴取值做笛卡尔积，组合下标按「最后一轴变化最快」
  的顺序从 0 递增，顺序确定、与提交无关地可复现；
- 轴字段名重复、同一轴内取值重复、扫描不允许的字段、取值类型/范围
  不合法，都在提交时以 invalid_params 拒绝；
- 展开后的组合数 / 总工作量超上限，以 resource_limit 拒绝；
- 每条组合都要通过与单任务完全相同的合法性/资源/稳定性校验，任意一条
  不合法则整批拒绝（原子受理），异常中逐条给出可区分原因。
"""
from __future__ import annotations

import math
from itertools import product
from typing import Any

from pydantic import ValidationError

from app.models import BatchRequest, ErrorReason, ScanAxis, SimulationRequest
from app.solver import (
    InvalidParamsError,
    ResourceLimitError,
    StabilityError,
    check_stability,
)

# ---- 批次规模上限（提交时强制，超限直接拒绝） ----
MAX_AXES = 8
MAX_VALUES_PER_AXIS = 256
MAX_BATCH_COMBINATIONS = 256          # 展开后的组合数上限
MAX_BATCH_TOTAL_WORK = 100_000_000    # 所有组合 nx*num_steps 之和的上限

# 允许扫描的字段（标量数值字段；枚举/字符串不参与扫描）
SCANNABLE_FIELDS: tuple[str, ...] = (
    "nx",
    "dt",
    "num_steps",
    "diffusion",
    "advection",
    "domain_length",
    "noise_amplitude",
    "seed",
)
# 整数字段：扫描值必须是整数（拒绝 1.5 这类取值）
_INT_FIELDS: frozenset[str] = frozenset({"nx", "num_steps", "seed"})

_STABILITY_ERRORS = {
    InvalidParamsError: ErrorReason.INVALID_PARAMS.value,
    StabilityError: ErrorReason.UNSTABLE.value,
    ResourceLimitError: ErrorReason.RESOURCE_LIMIT.value,
}


class BatchValidationError(ValueError):
    """批次在提交阶段被拒绝。

    - reason: 顶层可区分原因（invalid_params / unstable / resource_limit），
      取展开顺序中第一条非法组合的原因，确定且可预期；
    - invalid_combinations: 每条非法组合各自的原因（展开方式本身错误时为空）。
    """

    def __init__(
        self,
        reason: str,
        detail: str,
        invalid_combinations: list[dict] | None = None,
    ) -> None:
        super().__init__(detail)
        self.reason = reason
        self.invalid_combinations = invalid_combinations or []


def _invalid(detail: str, invalid_combinations: list[dict] | None = None) -> BatchValidationError:
    return BatchValidationError(ErrorReason.INVALID_PARAMS.value, detail, invalid_combinations)


def _normalize_value(field: str, value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(f"扫描轴 {field} 的取值必须是数值, got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise _invalid(f"扫描轴 {field} 的取值必须有限, got {value!r}")
    if field in _INT_FIELDS:
        if not v.is_integer():
            raise _invalid(f"扫描轴 {field} 为整数字段, 取值必须是整数, got {value!r}")
        return int(v)
    return v


def expand_combinations(
    base: dict[str, Any], axes: list[ScanAxis]
) -> list[tuple[dict[str, Any], dict[str, float | int]]]:
    """把 base 与扫描轴做笛卡尔积展开。

    返回 [(resolved_params, axis_values), ...]，顺序按轴序/取值序确定
    （最后一条轴变化最快）。展开方式不合法时抛 BatchValidationError。
    """
    if not axes:
        raise _invalid("扫描轴 axes 至少包含一条")
    if len(axes) > MAX_AXES:
        raise _invalid(f"扫描轴数量 {len(axes)} 超过上限 {MAX_AXES}")

    seen_fields: set[str] = set()
    axis_values: list[list[int | float]] = []
    for axis in axes:
        field = axis.field
        if field not in SCANNABLE_FIELDS:
            raise _invalid(
                f"扫描字段 {field!r} 不允许扫描；可扫描字段: {', '.join(SCANNABLE_FIELDS)}"
            )
        if field in seen_fields:
            raise _invalid(f"扫描轴字段重复: {field!r}（同一字段只允许出现一次）")
        seen_fields.add(field)
        if not axis.values:
            raise _invalid(f"扫描轴 {field} 的取值列表不能为空")
        if len(axis.values) > MAX_VALUES_PER_AXIS:
            raise BatchValidationError(
                ErrorReason.RESOURCE_LIMIT.value,
                f"扫描轴 {field} 的取值数 {len(axis.values)} 超过单轴上限 "
                f"{MAX_VALUES_PER_AXIS}",
            )
        normalized: list[int | float] = []
        seen_values: set[int | float] = set()
        for raw in axis.values:
            value = _normalize_value(field, raw)
            if value in seen_values:
                raise _invalid(
                    f"扫描轴 {field} 出现重复取值 {value}（去重后再提交）"
                )
            seen_values.add(value)
            normalized.append(value)
        axis_values.append(normalized)

    # 组合数：增量相乘，超限立即拒绝，避免真的展开出巨大列表
    count = 1
    for values in axis_values:
        count *= len(values)
        if count > MAX_BATCH_COMBINATIONS:
            raise BatchValidationError(
                ErrorReason.RESOURCE_LIMIT.value,
                f"展开后的组合数至少为 {count}，超过批次上限 "
                f"{MAX_BATCH_COMBINATIONS}；请收窄扫描轴后重新提交",
            )

    fields = [axis.field for axis in axes]
    combos: list[tuple[dict[str, Any], dict[str, float | int]]] = []
    for picked in product(*axis_values):
        overrides = dict(zip(fields, picked))
        params = dict(base)
        params.update(overrides)
        combos.append((params, dict(overrides)))
    return combos


def _format_pydantic_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def validate_batch(
    request: BatchRequest,
) -> list[tuple[dict[str, Any], dict[str, float | int]]]:
    """展开并对每一条组合执行合法性/资源/稳定性校验。

    全部通过才返回展开结果；任意一条不合法则整批拒绝（原子受理），
    异常中携带每条非法组合可区分的原因与下标。
    """
    combos = expand_combinations(request.base, request.axes)

    invalid_combinations: list[dict] = []
    valid_requests: list[SimulationRequest] = []
    for index, (params, axis_values) in enumerate(combos):
        try:
            req = SimulationRequest(**params)
        except ValidationError as exc:
            invalid_combinations.append(
                {
                    "index": index,
                    "axis_values": axis_values,
                    "error_reason": ErrorReason.INVALID_PARAMS.value,
                    "detail": _format_pydantic_error(exc),
                }
            )
            continue
        try:
            check_stability(req)
        except tuple(_STABILITY_ERRORS) as exc:
            invalid_combinations.append(
                {
                    "index": index,
                    "axis_values": axis_values,
                    "error_reason": _STABILITY_ERRORS[type(exc)],
                    "detail": str(exc),
                }
            )
            continue
        valid_requests.append(req)

    if invalid_combinations:
        first = invalid_combinations[0]
        raise BatchValidationError(
            first["error_reason"],
            f"{len(invalid_combinations)}/{len(combos)} 条组合未通过提交前校验，"
            "整批已拒绝（不会部分受理）；详见 invalid_combinations",
            invalid_combinations,
        )

    total_work = sum(r.nx * r.num_steps for r in valid_requests)
    if total_work > MAX_BATCH_TOTAL_WORK:
        raise BatchValidationError(
            ErrorReason.RESOURCE_LIMIT.value,
            f"所有组合 nx*num_steps 之和 {total_work} 超过批次总工作量上限 "
            f"{MAX_BATCH_TOTAL_WORK}；请减少组合数或单组合网格/步数",
        )
    return combos
