"""Request / response schemas and task state definitions."""

from enum import Enum

from pydantic import BaseModel

from .errors import ErrorReason


class BoundaryConditionType(str, Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    PERIODIC = "periodic"


class InitialConditionType(str, Enum):
    GAUSSIAN = "gaussian"
    STEP = "step"
    RANDOM = "random"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SimulationParameters(BaseModel):
    """Parameters of one 1D advection-diffusion simulation request."""

    grid_size: int
    domain_length: float
    dt: float
    num_steps: int
    advection_velocity: float
    diffusion_coefficient: float
    boundary_condition: BoundaryConditionType
    initial_condition: InitialConditionType
    noise_amplitude: float = 0.0
    seed: int | None = None


class TaskSnapshot(BaseModel):
    """Consistent view of a task from any query entry point."""

    task_id: str
    status: TaskStatus
    progress: float
    error_reason: ErrorReason | None = None
    error_detail: str | None = None


class SimulationResult(BaseModel):
    """Final result of a succeeded simulation."""

    task_id: str
    final_state: list[float]
    final_time: float
    steps_completed: int
