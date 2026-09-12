"""NumPy adapter: exact damping substep for arbitrary velocity arrays."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.analytical import exact_damping_alpha

__all__ = ["damp", "damp_with_energy"]


def damp(v: np.ndarray, dt: float, lam, alpha: float = 2.0) -> np.ndarray:
    """Exact substep v <- v (1 + alpha lam |v|^alpha dt)^(-1/alpha).

    ``v`` has shape (..., D); ``lam`` is a scalar or an array broadcastable
    against ``v.shape[:-1]`` (per-cell / per-particle coupling).
    """
    return exact_damping_alpha(np.asarray(v, dtype=np.float64), dt, lam, alpha)


def damp_with_energy(
    v: np.ndarray, dt: float, lam, alpha: float = 2.0, mass: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """Damped velocities and the kinetic energy removed (sum 1/2 m (|v|^2 - |v'|^2))."""
    v = np.asarray(v, dtype=np.float64)
    out = damp(v, dt, lam, alpha)
    m = 1.0 if mass is None else np.asarray(mass, dtype=np.float64)
    removed = 0.5 * np.sum(m * (np.sum(v * v, axis=-1) - np.sum(out * out, axis=-1)))
    return out, float(removed)
