"""Lagrangian (N-body) system with vacuum damping regularization.

The per-particle damping coupling follows the local density (code units),

    lambda_i = G * rho_i / c^3

and the analytical damping substep

    v <- v / sqrt(1 + 2 * lambda_i * |v|^2 * tau)

is applied inside a symmetric kick-damp-drift-damp-kick splitting:

    1. v^(*)      = v^n + 0.5 * a_grav(x^n) * dt            (half kick)
    2. v^(**)     = damp(v^(*), dt/2, lambda(rho^n))         (half damp, exact)
    3. x^(n+1)    = x^n + v^(**) * dt                        (drift)
    4. rho^(n+1)  = compute_density(x^(n+1))
    5. v^(***)    = damp(v^(**), dt/2, lambda(rho^(n+1)))    (half damp, exact)
    6. v^(n+1)    = v^(***) + 0.5 * a_grav(x^(n+1)) * dt     (half kick)

The two exact damping substeps cover dt/2 each, so one step damps over
exactly dt of physical time.  Because the exact flow of the damping ODE
composes, phi_a o phi_b = phi_(a+b), a damping-only run with constant
density reproduces v(T) = v0 / sqrt(1 + 2 lambda |v0|^2 T) to round-off
for *any* dt.  (Release 0.1.0 applied two substeps of length dt each and
therefore damped twice as strongly as the stated equation; see
CHANGELOG.md.)

With damping disabled the scheme reduces to standard leapfrog KDK.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

from ..core.analytical import exact_damping
from .kernel import compute_density

__all__ = ["LagrangianSystem"]

Callbacks = Optional[Callable[["LagrangianSystem"], None]]

# Pairwise (N, N, D) broadcasting is used below this many elements; larger
# systems fall back to the O(N) loop with O(N) memory per iteration.
_DENSE_MAX_ELEMENTS = 3_000_000


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
    eps2 = softening**2
    if n * n * d <= _DENSE_MAX_ELEMENTS:
        dr = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # x_j - x_i
        r2 = np.sum(dr * dr, axis=-1) + eps2
        factor = G * masses[np.newaxis, :] / (r2 * np.sqrt(r2))
        np.fill_diagonal(factor, 0.0)
        return np.einsum("ij,ijd->id", factor, dr)
    for i in range(n):
        dr = positions - positions[i]
        r2 = np.sum(dr * dr, axis=1) + eps2
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
        self.energy_to_vacuum = 0.0        # cumulative kinetic energy removed by damping
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

    # -- invariants -------------------------------------------------------

    def _kinetic(self, v: np.ndarray) -> float:
        return 0.5 * float(np.sum(self.masses * np.sum(v * v, axis=1)))

    def kinetic_energy(self) -> float:
        return self._kinetic(self.velocities)

    def potential_energy(self) -> float:
        """Plummer-softened pair potential, consistent with the force
        (zero when gravity is disabled)."""
        if not self.gravity or self.positions.shape[0] < 2:
            return 0.0
        x = self.positions
        dr = x[np.newaxis, :, :] - x[:, np.newaxis, :]
        r = np.sqrt(np.sum(dr * dr, axis=-1) + self.softening**2)
        mm = self.masses[:, np.newaxis] * self.masses[np.newaxis, :]
        iu = np.triu_indices(x.shape[0], k=1)
        return -self.G * float(np.sum(mm[iu] / r[iu]))

    def jacobian_min(self) -> float:
        """min_i J_i of the Lagrangian map (1D interval Jacobians; 3D SPH
        determinants using the smoothing lengths)."""
        from .jacobian import JacobianTracker

        q = self.initial_positions
        tracker = JacobianTracker(q[:, 0] if q.shape[1] == 1 else q, h=self.h)
        return tracker.min_jacobian(self.positions[:, 0] if q.shape[1] == 1 else self.positions)

    def diagnostics(self, potential: bool = True, jacobian: bool = True) -> dict:
        """Scalar diagnostics: E_kin, E_pot, E_vac (cumulative), E_total,
        v_max, rho_max, lam_max and J_min.  ``potential`` costs O(N^2)."""
        e_kin = self.kinetic_energy()
        e_pot = self.potential_energy() if potential else np.nan
        speed = np.sqrt(np.sum(self.velocities**2, axis=1))
        lam = self.damping_coefficients()
        out = {
            "E_kin": e_kin,
            "E_pot": e_pot,
            "E_vac": self.energy_to_vacuum,
            "E_total": e_kin + e_pot + self.energy_to_vacuum,
            "v_max": float(np.max(speed)) if speed.size else 0.0,
            "rho_max": float(np.max(self.densities)) if self.densities.size else 0.0,
            "lam_max": float(np.max(lam)) if lam.size else 0.0,
            "steps": self.n_steps,
        }
        if jacobian:
            out["J_min"] = self.jacobian_min()
        return out

    # -- integration ------------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance one symmetric kick/2 - damp/2 - drift - damp/2 - kick/2 step.

        Each exact damping substep covers dt/2, so the step damps over
        exactly dt of physical time.
        """
        half = 0.5 * dt
        a0 = self.gravity_acceleration()
        lam0 = self.damping_coefficients()

        v = self.velocities + half * a0                          # half-kick
        v_damped = exact_damping(v, half, lam0)                  # damp dt/2, rho^n
        self.energy_to_vacuum += self._kinetic(v) - self._kinetic(v_damped)
        v = v_damped
        self.positions = self.positions + v * dt                 # drift
        self.update_densities()                                  # rho^(n+1)

        a1 = self.gravity_acceleration()
        lam1 = self.damping_coefficients()
        v_damped = exact_damping(v, half, lam1)                  # damp dt/2, rho^(n+1)
        self.energy_to_vacuum += self._kinetic(v) - self._kinetic(v_damped)
        self.velocities = v_damped + half * a1                   # half-kick
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
          (accuracy of the splitting; the exact substep itself has no
          stability limit, pass ``damp_safety=np.inf`` to drop it)
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
