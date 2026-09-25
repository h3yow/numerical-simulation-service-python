"""一维对流-扩散方程求解器（显式 FTCS 格式）与稳定性检查。

求解 PDE:  du/dt + c * du/dx = D * d2u/dx2,  x in [0, L]

离散格式（显式）:
    u_new[i] = u[i] + r*(u[i+1] - 2*u[i] + u[i-1])
                    - (cfl/2)*(u[i+1] - u[i-1])
    r   = D*dt/dx^2   (扩散数)
    cfl = c*dt/dx     (对流 CFL 数)

稳定性约束（显式格式必要条件）:
    r <= 1/2, |cfl| <= 1, 且 D > 0 时 cfl^2 <= 2*r
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from app.models import BoundaryCondition, InitialCondition, SimulationRequest

# ---- 资源上限 ----
MIN_NX = 3
MAX_NX = 10_000
MAX_STEPS = 200_000
MAX_WORK = 20_000_000  # nx * num_steps 上限，防止单次任务占用过多资源

# ---- 稳定性阈值 ----
MAX_DIFFUSION_NUMBER = 0.5
MAX_CFL = 1.0

_CHECK_INTERVAL = 256  # 每多少步检查一次取消/发散


class InvalidParamsError(ValueError):
    """参数非法（取值范围或组合不合法）。"""


class StabilityError(ValueError):
    """数值稳定性约束不满足。"""


class ResourceLimitError(ValueError):
    """超出资源上限。"""


class CancelledError(Exception):
    """任务在执行中被取消。"""


def check_stability(req: SimulationRequest) -> None:
    """校验参数合法性、资源上限与数值稳定性。

    依次抛出 InvalidParamsError / ResourceLimitError / StabilityError。
    """
    # 1) 参数合法性
    if not math.isfinite(req.dt) or req.dt <= 0:
        raise InvalidParamsError(f"dt 必须为正的有限值, got {req.dt}")
    if not math.isfinite(req.advection):
        raise InvalidParamsError(f"advection 必须为有限值, got {req.advection}")
    if not math.isfinite(req.diffusion) or req.diffusion < 0:
        raise InvalidParamsError(f"diffusion 必须为非负有限值, got {req.diffusion}")
    if not math.isfinite(req.domain_length) or req.domain_length <= 0:
        raise InvalidParamsError(f"domain_length 必须为正的有限值, got {req.domain_length}")
    if not math.isfinite(req.noise_amplitude) or req.noise_amplitude < 0:
        raise InvalidParamsError(f"noise_amplitude 必须为非负有限值, got {req.noise_amplitude}")
    if req.initial == InitialCondition.RANDOM and req.seed is None:
        raise InvalidParamsError("initial=random 时必须提供 seed 以保证可复现")
    if req.noise_amplitude > 0 and req.seed is None:
        raise InvalidParamsError("noise_amplitude>0 时必须提供 seed 以保证可复现")

    # 2) 资源上限
    if req.nx < MIN_NX:
        raise InvalidParamsError(f"nx 至少为 {MIN_NX}, got {req.nx}")
    if req.nx > MAX_NX:
        raise ResourceLimitError(f"nx 超过上限 {MAX_NX}, got {req.nx}")
    if req.num_steps < 1:
        raise InvalidParamsError(f"num_steps 至少为 1, got {req.num_steps}")
    if req.num_steps > MAX_STEPS:
        raise ResourceLimitError(f"num_steps 超过上限 {MAX_STEPS}, got {req.num_steps}")
    if req.nx * req.num_steps > MAX_WORK:
        raise ResourceLimitError(
            f"nx*num_steps={req.nx * req.num_steps} 超过总工作量上限 {MAX_WORK}"
        )

    # 3) 数值稳定性
    dx = req.domain_length / (req.nx - 1)
    r = req.diffusion * req.dt / (dx * dx)
    cfl = abs(req.advection) * req.dt / dx
    if r > MAX_DIFFUSION_NUMBER:
        raise StabilityError(
            f"扩散数 r=D*dt/dx^2={r:.6g} 超过上限 {MAX_DIFFUSION_NUMBER}"
        )
    if cfl > MAX_CFL:
        raise StabilityError(f"CFL 数 |c|*dt/dx={cfl:.6g} 超过上限 {MAX_CFL}")
    if req.diffusion > 0 and cfl * cfl > 2 * r + 1e-15:
        raise StabilityError(
            f"组合约束 cfl^2={cfl * cfl:.6g} > 2*r={2 * r:.6g}，显式格式不稳定"
        )


def _initial_state(req: SimulationRequest, rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(0.0, req.domain_length, req.nx)
    if req.initial == InitialCondition.GAUSSIAN:
        mid = req.domain_length / 2.0
        sigma = req.domain_length / 10.0
        u = np.exp(-((x - mid) ** 2) / (2.0 * sigma * sigma))
    elif req.initial == InitialCondition.STEP:
        u = np.where(x >= req.domain_length / 2.0, 1.0, 0.0)
    elif req.initial == InitialCondition.RANDOM:
        u = rng.random(req.nx)
    else:  # pragma: no cover - 枚举兜底
        raise InvalidParamsError(f"未知初始条件: {req.initial}")
    if req.noise_amplitude > 0:
        u = u + req.noise_amplitude * (rng.random(req.nx) - 0.5)
    return u


def run_simulation(
    req: SimulationRequest,
    progress_cb: Optional[Callable[[float], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """执行时间步进仿真，返回末态向量（长度 nx）。

    - progress_cb: 每 _CHECK_INTERVAL 步回调一次进度 (0~1)
    - should_cancel: 返回 True 时抛出 CancelledError
    - 数值发散（非有限值）时抛出 RuntimeError
    """
    check_stability(req)

    rng = np.random.default_rng(req.seed)
    u = _initial_state(req, rng).astype(np.float64)

    dx = req.domain_length / (req.nx - 1)
    r = req.diffusion * req.dt / (dx * dx)
    cfl = req.advection * req.dt / dx
    periodic = req.boundary == BoundaryCondition.PERIODIC

    for step in range(1, req.num_steps + 1):
        left = np.roll(u, 1)
        right = np.roll(u, -1)
        if not periodic:
            # Dirichlet 边界 u=0：边界点不更新，roll 引入的环绕值不会被写回
            u_new = u + r * (right - 2.0 * u + left) - 0.5 * cfl * (right - left)
            u_new[0] = 0.0
            u_new[-1] = 0.0
        else:
            u_new = u + r * (right - 2.0 * u + left) - 0.5 * cfl * (right - left)
        u = u_new

        if step % _CHECK_INTERVAL == 0 or step == req.num_steps:
            if should_cancel is not None and should_cancel():
                raise CancelledError()
            if not np.all(np.isfinite(u)):
                raise RuntimeError(f"第 {step} 步出现非有限值，数值发散")
            if progress_cb is not None:
                progress_cb(step / req.num_steps)

    return u
