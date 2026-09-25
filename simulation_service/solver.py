"""1D advection-diffusion time-stepping solver (skeleton)."""

from dataclasses import dataclass

from .errors import ErrorReason
from .models import SimulationParameters


@dataclass
class ValidationResult:
    ok: bool
    reason: ErrorReason | None = None
    detail: str | None = None


def validate_parameters(params: SimulationParameters) -> ValidationResult:
    """Validate parameters: reject invalid / unstable / over-limit combos."""
    raise NotImplementedError


def run_simulation(
    params: SimulationParameters,
    should_cancel,
    on_progress,
) -> list[float]:
    """Run the time-stepping loop; returns final state. Skeleton only."""
    raise NotImplementedError
