"""数据模型与枚举：仿真请求、任务状态、错误原因、结果视图、批量扫参。"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

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


# ===================== 批量扫参 =====================

class BatchStatus(str, Enum):
    """批次聚合状态。"""

    PENDING = "pending"      # 还有组合在排队
    RUNNING = "running"      # 无排队、但仍有在跑
    COMPLETED = "completed"  # 全部到达终态（成功/失败/取消的任意混合）


TERMINAL_BATCH_STATES = {BatchStatus.COMPLETED}


class ScanAxis(BaseModel):
    """一条扫描轴：对参数 field 依次取 values 中的值。"""

    field: str = Field(..., description="要扫描的参数字段名")
    values: list[float | int] = Field(..., description="该轴的取值列表（笛卡尔积展开）")


class BatchRequest(BaseModel):
    """批量扫参请求：base 参数 + 若干扫描轴，按笛卡尔积展开。"""

    base: dict[str, Any] = Field(..., description="基础参数（同 SimulationRequest）")
    axes: list[ScanAxis] = Field(..., min_length=1, description="扫描轴，至少一条")


class BatchItemView(BaseModel):
    """批次中单条组合的状态视图（对号入座：index/params/task_id）。"""

    index: int
    task_id: str
    status: TaskStatus
    progress: float = 0.0
    error_reason: Optional[ErrorReason] = None
    error_detail: Optional[str] = None
    params: dict[str, Any] = Field(..., description="该组合展开后的完整参数")
    axis_values: dict[str, float | int] = Field(..., description="该组合在各扫描轴上的取值（保留原始数值类型）")


class BatchView(BaseModel):
    """批次聚合视图：总状态、总进度、计数与每条组合的状态。"""

    batch_id: str
    status: BatchStatus
    total: int
    queued: int = 0
    running: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    progress: float = 0.0  # 各组合进度的均值
    items: list[BatchItemView]


class BatchItemResult(BaseModel):
    """批次中单条组合的结果（含对号信息）。"""

    index: int
    task_id: str
    status: TaskStatus
    error_reason: Optional[ErrorReason] = None
    error_detail: Optional[str] = None
    nx: Optional[int] = None
    num_steps: Optional[int] = None
    final_state: Optional[list[float]] = None
    checksum: Optional[float] = None
    params: dict[str, Any]
    axis_values: dict[str, float | int]


class BatchResult(BaseModel):
    """整批结果：每条组合各自的结果，互不影响。"""

    batch_id: str
    status: BatchStatus
    total: int
    items: list[BatchItemResult]
