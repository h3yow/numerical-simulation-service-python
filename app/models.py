"""数据模型与枚举：仿真请求、任务状态、错误原因、结果视图。"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class BoundaryCondition(str, Enum):
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"


class InitialCondition(str, Enum):
    GAUSSIAN = "gaussian"
    STEP = "step"
    RANDOM = "random"  # 需要 seed


class ErrorReason(str, Enum):
    """可区分的失败原因。"""

    INVALID_PARAMS = "invalid_params"          # 参数非法（范围/组合）
    UNSTABLE = "unstable"                      # 违反数值稳定性约束
    RESOURCE_LIMIT = "resource_limit"          # 超出资源上限（网格/步数/队列）
    CANCELLED = "cancelled"                    # 任务被取消
    INTERNAL_ERROR = "internal_error"          # 求解过程内部错误（含发散）


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}


class SimulationRequest(BaseModel):
    """一维对流-扩散仿真请求。"""

    nx: int = Field(..., description="空间网格点数")
    dt: float = Field(..., description="时间步长")
    num_steps: int = Field(..., description="时间步数")
    diffusion: float = Field(..., ge=0.0, description="扩散系数 D >= 0")
    advection: float = Field(0.0, description="对流速度 c")
    domain_length: float = Field(1.0, gt=0.0, description="区间长度 L")
    boundary: BoundaryCondition = BoundaryCondition.PERIODIC
    initial: InitialCondition = InitialCondition.GAUSSIAN
    noise_amplitude: float = Field(0.0, ge=0.0, description="叠加在初值上的随机扰动幅度")
    seed: Optional[int] = Field(None, description="随机种子；含随机成分时用于复现")


class TaskView(BaseModel):
    """任务状态视图（进度查询 / 列表）。"""

    task_id: str
    status: TaskStatus
    progress: float = 0.0  # 0.0 ~ 1.0
    error_reason: Optional[ErrorReason] = None
    error_detail: Optional[str] = None


class SimulationResult(BaseModel):
    """仿真结果。"""

    task_id: str
    status: TaskStatus
    error_reason: Optional[ErrorReason] = None
    error_detail: Optional[str] = None
    nx: Optional[int] = None
    num_steps: Optional[int] = None
    final_state: Optional[list[float]] = None
    checksum: Optional[float] = None  # 末态求和，便于复现性断言


# ============ 批量扫参 ============

#: 允许作为扫描轴的参数名（与 SimulationRequest 可扫描字段一一对应）
SCANNABLE_FIELDS = (
    "nx",
    "dt",
    "num_steps",
    "diffusion",
    "advection",
    "domain_length",
    "noise_amplitude",
    "seed",
    "boundary",
    "initial",
)

ScanField = Literal[
    "nx",
    "dt",
    "num_steps",
    "diffusion",
    "advection",
    "domain_length",
    "noise_amplitude",
    "seed",
    "boundary",
    "initial",
]


class ScanAxis(BaseModel):
    """一条扫描轴：对参数 field 依次取 values 中的值。"""

    field: ScanField
    values: list = Field(..., min_length=1, description="该轴上的候选值（至少 1 个）")


class BatchRequest(BaseModel):
    """批量扫参请求：在 base 参数上对 axes 做笛卡尔积展开。"""

    base: SimulationRequest
    axes: list[ScanAxis] = Field(default_factory=list)


class BatchStatus(str, Enum):
    """批次总体状态（对各单条状态的如实聚合）。"""

    QUEUED = "queued"              # 还没有任何一条开始执行
    RUNNING = "running"            # 已有在跑/已完成的，且仍有未终态条目
    CANCELLED = "cancelled"        # 全部条目终态，且无成功、有取消
    FAILED = "failed"              # 全部条目终态，且无成功、无取消，只有失败
    SUCCEEDED = "succeeded"        # 全部条目成功
    PARTIAL_FAILED = "partial_failed"      # 全部终态：有成功也有失败（无取消）
    PARTIAL_CANCELLED = "partial_cancelled"  # 全部终态：有取消且其余不是全失败


class BatchItemView(BaseModel):
    """批次内单条组合的状态视图。"""

    item_id: str
    task_id: Optional[str] = None   # 已派发到执行器后才有 task_id
    params: dict                    # 该组合的完整参数（对号依据）
    status: TaskStatus
    progress: float = 0.0
    error_reason: Optional[ErrorReason] = None
    error_detail: Optional[str] = None


class BatchView(BaseModel):
    """批次总体视图：批次状态、进度与每条组合的状态。"""

    batch_id: str
    status: BatchStatus
    progress: float = 0.0           # 已完成（终态）条目占比
    total: int
    counts: dict[str, int]          # 按 TaskStatus 分桶的计数
    items: list[BatchItemView]


class BatchItemResult(BaseModel):
    """批次内单条组合的结果（参数对号 + 单任务结果字段）。"""

    item_id: str
    task_id: Optional[str] = None
    params: dict
    status: TaskStatus
    progress: float = 0.0
    error_reason: Optional[ErrorReason] = None
    error_detail: Optional[str] = None
    nx: Optional[int] = None
    num_steps: Optional[int] = None
    final_state: Optional[list[float]] = None
    checksum: Optional[float] = None


class BatchResultView(BaseModel):
    """批次结果视图：整批状态 + 每条组合各自的状态与结果。"""

    batch_id: str
    status: BatchStatus
    total: int
    counts: dict[str, int]
    items: list[BatchItemResult]
