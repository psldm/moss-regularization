"""Stability criteria and adaptive timestep selection.

Two limits bound dt for an explicit update with the vacuum damping term:

  1. Damping (explicit Euler) stability:  dt <= 2 / (lambda |v|^2),
     from |v_{n+1}| = |v_n| |1 - lambda |v_n|^2 dt| <= |v_n|.
  2. Advective CFL:                       dt <= CFL * dx / |v|_max.

``stable_dt`` returns the smallest admissible dt over an array of
velocities, and ``adaptive_dt`` combines it with a CFL limit when a grid
spacing is supplied.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..constants import LAMBDA

__all__ = [
    "damping_dt_limit",
    "cfl_dt_limit",
    "stable_dt",
    "adaptive_dt",
]


def damping_dt_limit(
    v: np.ndarray,
    lambda_eff: float = LAMBDA,
    safety: float = 0.5,
) -> float:
    """Largest dt keeping the explicit Euler damping substep monotone."""
    v = np.asarray(v, dtype=np.float64)
    speed2 = np.sum(v * v, axis=-1)
    vmax2 = float(np.max(speed2)) if speed2.size else 0.0
    if vmax2 <= 0.0:
        return np.inf
    return safety * 2.0 / (lambda_eff * vmax2)


def cfl_dt_limit(
    v: np.ndarray,
    dx: float,
    courant: float = 0.5,
) -> float:
    """Advective CFL limit for a grid cell of width dx."""
    v = np.asarray(v, dtype=np.float64)
    speed = np.sqrt(np.sum(v * v, axis=-1))
    vmax = float(np.max(speed)) if speed.size else 0.0
    if vmax <= 0.0:
        return np.inf
    return courant * dx / vmax


def stable_dt(
    v: np.ndarray,
    lambda_eff: float = LAMBDA,
    dx: Optional[float] = None,
    courant: float = 0.5,
    safety: float = 0.5,
) -> float:
    """Smallest admissible dt for explicit advection + damping steps."""
    dt_damp = damping_dt_limit(v, lambda_eff=lambda_eff, safety=safety)
    dt = dt_damp
    if dx is not None:
        dt = min(dt, cfl_dt_limit(v, dx, courant=courant))
    return dt


def adaptive_dt(
    v: np.ndarray,
    dt_max: float,
    lambda_eff: float = LAMBDA,
    dx: Optional[float] = None,
    courant: float = 0.5,
    safety: float = 0.5,
) -> float:
    """Adaptive dt: stability limits, capped at dt_max."""
    return min(dt_max, stable_dt(v, lambda_eff, dx, courant, safety))
