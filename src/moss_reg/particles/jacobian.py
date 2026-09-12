"""Deformation-gradient / Jacobian tracking for shell-crossing detection.

The Lagrangian map q -> x(q, t) has Jacobian matrix

    J_ij = d x_i / d q_j,   J = det(J).

For an invertible (no shell-crossing) flow the determinant stays
strictly positive, J(q, t) > 0.  Shell crossing is signalled by J <= 0.
(Release 0.1.0 referred to a "Theorem D.1" here; no such theorem exists
in the accompanying paper, and the label is dropped.)

* 1D:  J_i = (x_{i+1} - x_i) / (q_{i+1} - q_i) over the sorted
       Lagrangian ordering.
* 3D:  SPH least-squares estimate of the full tensor gradient dx/dq
       at every particle.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.spatial import cKDTree

from .kernel import cubic_spline

__all__ = [
    "one_d_jacobians",
    "sph_deformation_gradients",
    "jacobian_determinants",
    "JacobianTracker",
]


def one_d_jacobians(
    positions: np.ndarray,
    initial_positions: np.ndarray,
) -> np.ndarray:
    """Interval Jacobians J_i = (x_{i+1} - x_i) / (q_{i+1} - q_i).

    Both arrays are sorted along the initial Lagrangian ordering before
    taking differences, so J > 0 iff the ordering is preserved.
    """
    x = np.asarray(positions, dtype=np.float64).ravel()
    q = np.asarray(initial_positions, dtype=np.float64).ravel()
    if x.shape != q.shape:
        raise ValueError("positions and initial_positions must have equal size")
    order = np.argsort(q, kind="stable")
    qs = q[order]
    xs = x[order]
    dq = np.diff(qs)
    dx = np.diff(xs)
    if np.any(dq == 0.0):
        raise ValueError("initial positions must be strictly increasing")
    return dx / dq


def sph_deformation_gradients(
    positions: np.ndarray,
    initial_positions: np.ndarray,
    h: Union[float, np.ndarray],
) -> np.ndarray:
    """Least-squares SPH estimate of the deformation gradient tensor.

    For every particle i,

        A_i = sum_j W_ij (x_j - x_i) (q_j - q_i)^T
        B_i = sum_j W_ij (q_j - q_i) (q_j - q_i)^T
        J_i = A_i B_i^{-1}

    where W is the cubic spline kernel evaluated on the current
    positions with smoothing length h.  Returns shape (N, D, D).
    """
    x = np.asarray(positions, dtype=np.float64)
    q = np.asarray(initial_positions, dtype=np.float64)
    if x.ndim != 2 or q.shape != x.shape:
        raise ValueError("positions and initial_positions must share shape (N, D)")
    n, d = x.shape
    if d not in (1, 3):
        raise ValueError(f"only 1D and 3D are supported, got D={d}")
    h_arr = np.broadcast_to(np.asarray(h, dtype=np.float64), (n,)).copy()

    tree = cKDTree(x)
    grads = np.zeros((n, d, d))
    for i in range(n):
        js = tree.query_ball_point(x[i], h_arr[i])
        if not js:
            continue
        dr = x[js] - x[i]
        dq = q[js] - q[i]
        w = cubic_spline(
            np.sqrt(np.sum(dr * dr, axis=1)), h_arr[i], dim=d
        )
        w = w[:, np.newaxis, np.newaxis]
        A = np.sum(w * dr[:, :, np.newaxis] * dq[:, np.newaxis, :], axis=0)
        B = np.sum(w * dq[:, :, np.newaxis] * dq[:, np.newaxis, :], axis=0)
        grads[i] = A @ np.linalg.pinv(B)
    return grads


def jacobian_determinants(grads: np.ndarray) -> np.ndarray:
    """det(J_i) for a stack of deformation gradient matrices."""
    grads = np.asarray(grads)
    if grads.ndim == 2:
        return np.linalg.det(grads)[None]
    return np.linalg.det(grads)


class JacobianTracker:
    """Monitor min_i J_i(t) for a Lagrangian map.

    For 1D the interval Jacobians of the sorted map are used; for 3D
    the per-particle determinants of the SPH deformation gradient.
    """

    def __init__(
        self,
        initial_positions: np.ndarray,
        h: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        self.q = np.asarray(initial_positions, dtype=np.float64)
        self.h = h

    @property
    def dim(self) -> int:
        return 1 if self.q.ndim == 1 else self.q.shape[1]

    def evaluate(self, positions: np.ndarray) -> np.ndarray:
        """Return per-element Jacobian measures (intervals in 1D, dets in 3D)."""
        if self.dim == 1:
            return one_d_jacobians(positions, self.q)
        if self.h is None:
            raise ValueError("h must be set for the 3D SPH gradient estimate")
        grads = sph_deformation_gradients(positions, self.q, self.h)
        return jacobian_determinants(grads)

    def min_jacobian(self, positions: np.ndarray) -> float:
        """min_i J_i(t); <= 0 indicates shell crossing."""
        vals = self.evaluate(positions)
        return float(np.min(vals)) if vals.size else np.inf
