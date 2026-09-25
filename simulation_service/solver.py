"""1D advection-diffusion time-stepping solver.

Solves  du/dt + a * du/dx = D * d2u/dx2  on [0, L] with an explicit
FTCS scheme (first-order upwind advection, central diffusion).

Stability constraints (checked before any computation):
  * diffusion number   d = D * dt / dx^2   must satisfy d <= 0.5
  * advection CFL      c = |a| * dt / dx   must satisfy c <= 1
  * combined FTCS-upwind safety: 2*d + c <= 1
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .errors import ErrorReason
from .models import BoundaryConditionType, InitialConditionType, SimulationParameters

# Resource limits.
MIN_GRID_SIZE = 3
MAX_GRID_SIZE = 10_000
MIN_STEPS = 1
MAX_STEPS = 100_000
MAX_TOTAL_WORK = 20_000_000  # grid_size * num_steps upper bound


class SolverCancelled(Exception):
    """Raised inside the solver loop when the task is cancelled."""


@dataclass
class ValidationResult:
    ok: bool
    reason: ErrorReason | None = None
    detail: str | None = None


def validate_parameters(params: SimulationParameters) -> ValidationResult:
    """Validate parameters; distinguish invalid / unstable / over-limit."""
    if not (MIN_GRID_SIZE <= params.grid_size <= MAX_GRID_SIZE):
        return ValidationResult(
            False,
            ErrorReason.RESOURCE_LIMIT_EXCEEDED,
            f"grid_size must be in [{MIN_GRID_SIZE}, {MAX_GRID_SIZE}], got {params.grid_size}",
        )
    if not (MIN_STEPS <= params.num_steps <= MAX_STEPS):
        return ValidationResult(
            False,
            ErrorReason.RESOURCE_LIMIT_EXCEEDED,
            f"num_steps must be in [{MIN_STEPS}, {MAX_STEPS}], got {params.num_steps}",
        )
    if params.grid_size * params.num_steps > MAX_TOTAL_WORK:
        return ValidationResult(
            False,
            ErrorReason.RESOURCE_LIMIT_EXCEEDED,
            f"grid_size*num_steps={params.grid_size * params.num_steps} exceeds {MAX_TOTAL_WORK}",
        )
    if params.domain_length <= 0:
        return ValidationResult(
            False, ErrorReason.INVALID_PARAMETERS, "domain_length must be positive"
        )
    if params.dt <= 0:
        return ValidationResult(False, ErrorReason.INVALID_PARAMETERS, "dt must be positive")
    if params.diffusion_coefficient < 0:
        return ValidationResult(
            False, ErrorReason.INVALID_PARAMETERS, "diffusion_coefficient must be >= 0"
        )
    if params.noise_amplitude < 0:
        return ValidationResult(
            False, ErrorReason.INVALID_PARAMETERS, "noise_amplitude must be >= 0"
        )
    if (
        params.noise_amplitude > 0 or params.initial_condition == InitialConditionType.RANDOM
    ) and params.seed is None:
        return ValidationResult(
            False,
            ErrorReason.INVALID_PARAMETERS,
            "seed is required when randomness is involved (random initial condition or noise)",
        )

    dx = params.domain_length / (params.grid_size - 1)
    d = params.diffusion_coefficient * params.dt / dx**2
    c = abs(params.advection_velocity) * params.dt / dx
    if d > 0.5:
        return ValidationResult(
            False,
            ErrorReason.NUMERICALLY_UNSTABLE,
            f"diffusion number D*dt/dx^2={d:.4g} exceeds 0.5",
        )
    if c > 1.0:
        return ValidationResult(
            False,
            ErrorReason.NUMERICALLY_UNSTABLE,
            f"advection CFL |a|*dt/dx={c:.4g} exceeds 1",
        )
    if 2 * d + c > 1.0:
        return ValidationResult(
            False,
            ErrorReason.NUMERICALLY_UNSTABLE,
            f"combined criterion 2d+c={2 * d + c:.4g} exceeds 1",
        )
    return ValidationResult(True)


def _initial_state(params: SimulationParameters, rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(0.0, params.domain_length, params.grid_size)
    if params.initial_condition == InitialConditionType.GAUSSIAN:
        mid = params.domain_length / 2.0
        sigma = params.domain_length / 10.0
        u = np.exp(-((x - mid) ** 2) / (2 * sigma**2))
    elif params.initial_condition == InitialConditionType.STEP:
        u = np.where(x < params.domain_length / 2.0, 1.0, 0.0)
    else:  # RANDOM
        u = rng.random(params.grid_size)
    if params.noise_amplitude > 0:
        u = u + params.noise_amplitude * (rng.random(params.grid_size) - 0.5)
    return u


def _apply_boundary(u: np.ndarray, bc: BoundaryConditionType) -> None:
    if bc == BoundaryConditionType.DIRICHLET:
        u[0] = 0.0
        u[-1] = 0.0
    elif bc == BoundaryConditionType.NEUMANN:
        u[0] = u[1]
        u[-1] = u[-2]
    # PERIODIC handled by np.roll in the update step.


def run_simulation(
    params: SimulationParameters,
    should_cancel: Callable[[], bool],
    on_progress: Callable[[float], None],
) -> np.ndarray:
    """Run the explicit time-stepping loop and return the final state.

    Deterministic for a given (parameters, seed) pair: all randomness comes
    from a generator seeded with ``params.seed``.
    """
    rng = np.random.default_rng(params.seed)
    u = _initial_state(params, rng)
    _apply_boundary(u, params.boundary_condition)

    dx = params.domain_length / (params.grid_size - 1)
    d = params.diffusion_coefficient * params.dt / dx**2
    a = params.advection_velocity
    periodic = params.boundary_condition == BoundaryConditionType.PERIODIC

    for step in range(params.num_steps):
        if should_cancel():
            raise SolverCancelled()
        diffusion = d * (np.roll(u, -1) - 2 * u + np.roll(u, 1))
        if a >= 0:
            advection = -(a * params.dt / dx) * (u - np.roll(u, 1))
        else:
            advection = -(a * params.dt / dx) * (np.roll(u, -1) - u)
        u = u + diffusion + advection
        if not periodic:
            _apply_boundary(u, params.boundary_condition)
        if not np.all(np.isfinite(u)):
            raise FloatingPointError(f"solution became non-finite at step {step + 1}")
        on_progress((step + 1) / params.num_steps)
    return u
