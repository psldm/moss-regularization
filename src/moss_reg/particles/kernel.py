"""SPH smoothing kernels and local density estimation.

Cubic spline (M4) kernel in 1D and 3D:

    1D:  W(r, h) = (4 / (3 h))    * f(r / h)
    3D:  W(r, h) = (8 / (pi h^3)) * f(r / h)

    f(u) = 1 - 6 u^2 + 6 u^3          for 0 <= u < 1/2
         = 2 (1 - u)^3                for 1/2 <= u < 1
         = 0                          otherwise

with the usual normalization int W dV = 1.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "cubic_spline",
    "cubic_spline_gradient",
    "cubic_spline_grad_vector",
    "compute_density",
    "adaptive_smoothing_length",
]

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

_SPLINE_CORE = Union[float, np.ndarray]


def _spline_core(u: _SPLINE_CORE) -> _SPLINE_CORE:
    """Dimensionless cubic spline shape function f(u), u = r / h."""
    u = np.asarray(u, dtype=np.float64)
    f = np.zeros_like(u)
    low = u < 0.5
    mid = (u >= 0.5) & (u < 1.0)
    f = np.where(low, 1.0 - 6.0 * u**2 + 6.0 * u**3, f)
    f = np.where(mid, 2.0 * (1.0 - u) ** 3, f)
    return f


def _spline_core_deriv(u: _SPLINE_CORE) -> _SPLINE_CORE:
    """df/du of the cubic spline shape function."""
    u = np.asarray(u, dtype=np.float64)
    f = np.zeros_like(u)
    low = u < 0.5
    mid = (u >= 0.5) & (u < 1.0)
    f = np.where(low, -12.0 * u + 18.0 * u**2, f)
    f = np.where(mid, -6.0 * (1.0 - u) ** 2, f)
    return f


def _norm_1d(h: float) -> float:
    return 4.0 / (3.0 * h)


def _norm_3d(h: float) -> float:
    return 8.0 / (np.pi * h**3)


def cubic_spline(
    r: Union[float, np.ndarray],
    h: float,
    dim: int = 1,
) -> np.ndarray:
    """Kernel value W(r, h) in dim = 1 or dim = 3.

    Parameters
    ----------
    r : scalar or array of positive distances.
    h : smoothing length (same units as r).
    dim : spatial dimension (1 or 3).
    """
    if dim not in (1, 3):
        raise ValueError(f"cubic spline kernel only supports dim in {{1, 3}}, got {dim}")
    r = np.asarray(r, dtype=np.float64)
    h = float(h)
    norm = _norm_1d(h) if dim == 1 else _norm_3d(h)
    return norm * _spline_core(r / h)


def cubic_spline_gradient(
    r: Union[float, np.ndarray],
    h: float,
    dim: int = 1,
) -> np.ndarray:
    """Radial derivative dW/dr at distance r (scalar, direction-free)."""
    if dim not in (1, 3):
        raise ValueError(f"cubic spline kernel only supports dim in {{1, 3}}, got {dim}")
    r = np.asarray(r, dtype=np.float64)
    h = float(h)
    if dim == 1:
        return _norm_1d(h) / h * _spline_core_deriv(r / h)
    return _norm_3d(h) / h * _spline_core_deriv(r / h)


def cubic_spline_grad_vector(
    dr: np.ndarray,
    h: float,
    dim: int = 3,
) -> np.ndarray:
    """Vector gradient grad W for displacement vectors dr, shape (..., D).

    Returns zeros for |dr| >= h.  In 1D this is dW/dr * sign(dr).
    """
    if dim not in (1, 3):
        raise ValueError(f"cubic spline kernel only supports dim in {{1, 3}}, got {dim}")
    dr = np.asarray(dr, dtype=np.float64)
    r = np.sqrt(np.sum(dr * dr, axis=-1))
    dWdr = cubic_spline_gradient(r, h, dim=dim)[..., np.newaxis]
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.where(r[..., np.newaxis] > 0.0, dr / r[..., np.newaxis], 0.0)
    return dWdr * rhat


# ---------------------------------------------------------------------------
# Density estimation
# ---------------------------------------------------------------------------

def compute_density(
    positions: np.ndarray,
    masses: np.ndarray,
    h: Union[float, np.ndarray],
    dim: Optional[int] = None,
    boxsize: Optional[float] = None,
) -> np.ndarray:
    """SPH density estimate rho_i = sum_j m_j W(|x_i - x_j|, h_i).

    Neighbor search uses ``scipy.spatial.cKDTree``.  The self-term
    (j = i) is included so that the estimate integrates to the total
    mass away from boundaries.  Passing ``boxsize`` enables periodic
    boundary conditions, in which case sum_i rho_i dV == sum_i m_i.
    """
    positions = np.asarray(positions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if positions.ndim != 2:
        raise ValueError("positions must have shape (N, D)")
    n, d = positions.shape
    if d not in (1, 3):
        raise ValueError(f"only 1D and 3D are supported, got D={d}")
    dim = d if dim is None else dim
    if masses.shape != (n,):
        raise ValueError(f"masses must have shape ({n},)")

    h_arr = np.broadcast_to(np.asarray(h, dtype=np.float64), (n,)).copy()
    if np.any(h_arr <= 0.0):
        raise ValueError("smoothing lengths must be positive")

    tree = cKDTree(positions, boxsize=boxsize)
    rho = np.zeros(n)
    for i in range(n):
        js = tree.query_ball_point(positions[i], h_arr[i])
        if not js:
            continue
        dr = positions[js] - positions[i]
        if boxsize is not None:
            dr -= boxsize * np.rint(dr / boxsize)
        r = np.sqrt(np.sum(dr * dr, axis=1))
        rho[i] = np.sum(masses[js] * cubic_spline(r, h_arr[i], dim=dim))
    return rho


# ---------------------------------------------------------------------------
# Adaptive smoothing length
# ---------------------------------------------------------------------------

def adaptive_smoothing_length(
    positions: np.ndarray,
    h0: float,
    n_ngb: int = 32,
    n_iter: int = 3,
    h_min_factor: float = 0.5,
    h_max_factor: float = 4.0,
) -> np.ndarray:
    """Iteratively adjust h_i so each particle sees ~n_ngb neighbours.

    h_i <- h_i * (n_ngb / N_neighbors(h_i))^(1 / D), clipped to
    [h_min_factor, h_max_factor] * h0.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n, d = positions.shape
    if d not in (1, 3):
        raise ValueError(f"only 1D and 3D are supported, got D={d}")
    tree = cKDTree(positions)
    h = np.full(n, float(h0))
    for _ in range(max(1, n_iter)):
        counts = np.asarray(
            [len(tree.query_ball_point(positions[i], h[i])) - 1 for i in range(n)],
            dtype=np.float64,
        )
        h *= (n_ngb / np.maximum(counts, 1.0)) ** (1.0 / d)
        h = np.clip(h, h_min_factor * h0, h_max_factor * h0)
    return h
