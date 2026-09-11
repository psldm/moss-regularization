"""Exact analytical solution of the nonlinear vacuum damping ODE.

The substep ODE

    dv/dt = -lambda_eff |v|^2 v          (v = d x / dt)

has the closed-form solution

    v(t) = v0 / sqrt(1 + 2 * lambda_eff * |v0|^2 * t),

which preserves the direction of motion:  v_hat(t) = v_hat(0).
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ..constants import LAMBDA

__all__ = [
    "exact_damping",
    "damped_step",
    "euler_step",
    "damping_timescale",
]

ArrayLike = Union[np.ndarray, float, list]


def _as_array(v: ArrayLike) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


def _speed_sq(v: np.ndarray) -> np.ndarray:
    return np.sum(v * v, axis=-1, keepdims=True)


def exact_damping(
    v0: ArrayLike,
    t: ArrayLike,
    lambda_eff: ArrayLike = LAMBDA,
) -> np.ndarray:
    """Exact solution of dv/dt = -lambda_eff |v|^2 v.

    Parameters
    ----------
    v0 : array_like, shape (..., D) or (D,)
        Initial velocity vector(s) with D in {1, 2, 3}.
    t : array_like, scalar or broadcastable against v0[..., 0].
        Time(s) at which to evaluate the solution.
    lambda_eff : float or array_like
        Effective damping coupling.  May be a scalar or a per-particle
        array broadcastable against the leading axes of ``v0``
        (shape (N,) for v0 of shape (N, D)).

    Returns
    -------
    v : ndarray, same shape as broadcast(v0, t)
        Velocity at time t; direction is preserved exactly.
    """
    v0 = _as_array(v0)
    lam = np.asarray(lambda_eff, dtype=np.float64)
    if lam.ndim > 0:
        lam = lam[..., np.newaxis]
    sq = _speed_sq(v0)
    t_arr = np.asarray(t, dtype=np.float64)[..., np.newaxis]
    denom = np.sqrt(1.0 + 2.0 * lam * sq * t_arr)
    # Guard the v0 = 0, t = 0 limits against 0 / 0.
    zero = np.logical_or(sq == 0.0, t_arr == 0.0)
    denom = np.where(zero, 1.0, denom)
    return v0 / denom


def damped_step(
    v: ArrayLike,
    dt: float,
    lambda_eff: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Operator-splitting substep: advance v exactly over one interval dt.

    Uses the analytical solution so the substep is exact for any dt; the
    only splitting error comes from coupling with the advection step.
    """
    lam = LAMBDA if lambda_eff is None else lambda_eff
    return exact_damping(v, dt, lam)


def euler_step(
    v: ArrayLike,
    dt: float,
    lambda_eff: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Explicit Euler substep (reference integrator for tests)."""
    lam = LAMBDA if lambda_eff is None else np.asarray(lambda_eff, dtype=np.float64)
    v = _as_array(v)
    if lam.ndim > 0:
        lam = lam[..., np.newaxis]
    return v - lam * _speed_sq(v) * v * dt


def damping_timescale(
    v: ArrayLike,
    lambda_eff: ArrayLike = LAMBDA,
) -> np.ndarray:
    """Characteristic damping time tau = 1 / (lambda_eff |v|^2)."""
    v = _as_array(v)
    lam = np.asarray(lambda_eff, dtype=np.float64)
    if lam.ndim > 0:
        lam = lam[..., np.newaxis]
    sq = _speed_sq(v)
    tau = np.full_like(sq, np.inf)
    np.divide(1.0, lam * sq, out=tau, where=sq > 0.0)
    return tau
