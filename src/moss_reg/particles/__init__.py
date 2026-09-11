"""Lagrangian / N-body SPH density kernel, integrator and shell-crossing
monitoring.

Submodules
----------
kernel      : cubic spline SPH kernels, density estimation, adaptive h
integrator  : LagrangianSystem with gravity, damping, adaptive dt
jacobian    : deformation gradient tracking and J > 0 monitoring
"""

from .kernel import (
    adaptive_smoothing_length,
    compute_density,
    cubic_spline,
    cubic_spline_grad_vector,
    cubic_spline_gradient,
)
from .integrator import LagrangianSystem
from .jacobian import JacobianTracker, one_d_jacobians, sph_deformation_gradients

__all__ = [
    "adaptive_smoothing_length",
    "compute_density",
    "cubic_spline",
    "cubic_spline_gradient",
    "cubic_spline_grad_vector",
    "LagrangianSystem",
    "JacobianTracker",
    "one_d_jacobians",
    "sph_deformation_gradients",
]
