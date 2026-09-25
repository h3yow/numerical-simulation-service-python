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
