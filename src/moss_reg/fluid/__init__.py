"""Eulerian (spectral) solvers with the cubic damping term and coupling helpers."""

from .coupling import (
    damping_number,
    lambda_eff_si,
    lambda_field_from_density,
    physical_damping_number,
)
from .spectral import SpectralNS2D
from .spectral3d import SpectralNS3D

__all__ = [
    "SpectralNS2D",
    "SpectralNS3D",
    "damping_number",
    "lambda_eff_si",
    "lambda_field_from_density",
    "physical_damping_number",
]
