"""Lagrangian (N-body) system with vacuum damping regularization.

The per-particle damping coupling follows the local density:

    lambda_i = G * rho_i / c^3

and the analytical damping substep

    v <- v / sqrt(1 + 2 * lambda_i * |v|^2 * dt)

is applied inside a kick-drift-kick operator splitting:

    1. v^(*)      = v + 0.5 * a_grav(x^n) * dt
    2. v^(**)     = damp(v^(*), dt, lambda(rho^n))
    3. x^(n+1)    = x^n + v^(**) * dt
    4. rho^(n+1)  = compute_density(x^(n+1))
    5. v^(n+1)    = damp(v^(**) + 0.5 * a_grav(x^(n+1)) * dt,
                        dt, lambda(rho^(n+1)))

With damping disabled the scheme reduces to standard leapfrog KDK.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

from ..core.analytical import exact_damping
from .kernel import compute_density

__all__ = ["LagrangianSystem"]

Callbacks = Optional[Callable[["LagrangianSystem"], None]]


def _direct_gravity(
    positions: np.ndarray,
    masses: np.ndarray,
    G: float,
    softening: float,
) -> np.ndarray:
    """Pairwise gravity with Plummer softening, O(N^2), any dimension."""
    n, d = positions.shape
    acc = np.zeros((n, d))
    if n < 2:
        return acc
    for i in range(n):
        dr = positions - positions[i]
        r2 = np.sum(dr * dr, axis=1) + softening**2
        factor = G * masses / (r2 * np.sqrt(r2))
        factor[i] = 0.0
        acc[i] = np.sum(dr * factor[:, np.newaxis], axis=0)
    return acc


class LagrangianSystem:
    """Self-gravitating particle system advanced with damping-regularized KDK.

    Parameters
    ----------
    positions : (N, D) array
    velocities : (N, D) array
    masses : (N,) array
    h : float or (N,) array
        SPH smoothing length(s).
    G, c : float
        Gravitational constant and speed of light in code units.
    softening : float
        Plummer softening length for gravity.
    damping : bool
        Enable the vacuum damping substeps.
    gravity : bool
        Enable gravitational acceleration (disable for damping-only runs).
    """

    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        h: Union[float, np.ndarray],
        G: float = 1.0,
        c: float = 1.0,
        softening: float = 1e-2,
        damping: bool = True,
        gravity: bool = True,
    ) -> None:
        self.positions = np.asarray(positions, dtype=np.float64)
        self.velocities = np.asarray(velocities, dtype=np.float64)
        self.masses = np.asarray(masses, dtype=np.float64)
        if self.positions.ndim != 2 or self.velocities.shape != self.positions.shape:
            raise ValueError("positions and velocities must share shape (N, D)")
        n, d = self.positions.shape
        if self.masses.shape != (n,):
            raise ValueError(f"masses must have shape ({n},)")
        self.initial_positions = self.positions.copy()
        self.h = np.broadcast_to(np.asarray(h, dtype=np.float64), (n,)).copy()
        self.G = float(G)
        self.c = float(c)
        self.softening = float(softening)
        self.damping = bool(damping)
        self.gravity = bool(gravity)
        self.time = 0.0
        self.n_steps = 0
        self.densities = compute_density(self.positions, self.masses, self.h)
        self.acceleration = self.gravity_acceleration()

    # -- physics ----------------------------------------------------------

    def gravity_acceleration(self) -> np.ndarray:
        """Direct O(N^2) gravity with Plummer softening."""
        if not self.gravity:
            self.acceleration = np.zeros_like(self.positions)
            return self.acceleration
        self.acceleration = _direct_gravity(
            self.positions, self.masses, self.G, self.softening
        )
        return self.acceleration

    def update_densities(self) -> np.ndarray:
        self.densities = compute_density(self.positions, self.masses, self.h)
        return self.densities

    def damping_coefficients(self) -> np.ndarray:
        """lambda_i = G * rho_i / c^3 (zeros when damping is disabled)."""
        if not self.damping:
            return np.zeros_like(self.densities)
        return self.G * self.densities / self.c**3

    # -- integration ------------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance one full KDK + damping operator-splitting step."""
        a0 = self.gravity_acceleration()
        lam0 = self.damping_coefficients()

        v = self.velocities + 0.5 * a0 * dt                      # half-kick
        v = exact_damping(v, dt, lam0)                           # damp, rho^n
        self.positions = self.positions + v * dt                 # drift
        self.update_densities()                                  # rho^(n+1)

        a1 = self.gravity_acceleration()
        lam1 = self.damping_coefficients()
        v = v + 0.5 * a1 * dt                                    # second half-kick
        self.velocities = exact_damping(v, dt, lam1)             # damp, rho^(n+1)
        self.time += dt
        self.n_steps += 1

    def adaptive_dt(
        self,
        dt_max: float = np.inf,
        eta: float = 0.2,
        cfl: float = 0.5,
        damp_safety: float = 0.5,
    ) -> float:
        """dt = min(dt_max, dt_grav, dt_damp, dt_CFL).

        * dt_grav = eta * sqrt(softening / max|a_grav|)   (Plummer criterion)
        * dt_damp = damp_safety * 2 / max_i(lambda_i |v_i|^2)
        * dt_CFL  = cfl * min(h) / max|v|
        """
        v = self.velocities
        speed = np.sqrt(np.sum(v * v, axis=1))
        vmax = float(np.max(speed)) if speed.size else 0.0

        amag = np.sqrt(np.sum(self.acceleration**2, axis=1))
        amax = float(np.max(amag)) if amag.size else 0.0
        dt_grav = np.inf if amax == 0.0 else eta * np.sqrt(self.softening / amax)

        lam = self.damping_coefficients()
        lam_v2 = lam * speed**2
        lam_v2_max = float(np.max(lam_v2)) if lam_v2.size else 0.0
        dt_damp = (
            np.inf
            if lam_v2_max == 0.0
            else damp_safety * 2.0 / lam_v2_max
        )

        h_min = float(np.min(self.h))
        dt_cfl = np.inf if vmax == 0.0 else cfl * h_min / vmax

        return float(min(dt_max, dt_grav, dt_damp, dt_cfl))

    def advance(
        self,
        t_end: float,
        dt_max: Optional[float] = None,
        callback: Callbacks = None,
        **dt_kwargs,
    ) -> None:
        """Integrate from the current time to t_end with adaptive dt."""
        if dt_max is None:
            dt_max = np.inf
        while self.time < t_end - 1e-14:
            dt = self.adaptive_dt(dt_max=dt_max, **dt_kwargs)
            dt = min(dt, t_end - self.time)
            self.step(dt)
            if callback is not None:
                callback(self)
