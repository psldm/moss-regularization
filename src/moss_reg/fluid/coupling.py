"""Coupling helpers: from densities and units to the number a solver needs.

The damping term in a solver is  -lam |u|^alpha u  with ``lam`` in the
solver's own units.  Three ways to obtain it:

* **code units** (G = 1, mass and length units = 1): ``lam = G rho / c^3``
  with ``rho`` the local density and ``c`` the speed of light in code
  units, exactly what the particle integrator does;
  :func:`lambda_field_from_density` does the same for a density field on
  an Eulerian grid (with an explicit length ``ell``);
* **SI**: :func:`lambda_eff_si` returns ``G rho ell / c^3`` in s m^-2; the
  length ``ell`` must be supplied by the physics (open problem, see README);
* **nondimensional**: :func:`damping_number` converts an effective
  coupling into the dimensionless ``lam_code = lam_eff U^alpha L`` of the
  nondimensionalized equations, which is the single parameter of a
  simulation and the value that must be reported with any result.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from .. import constants as K

__all__ = [
    "damping_number",
    "lambda_eff_si",
    "physical_damping_number",
    "lambda_field_from_density",
]

ArrayOrFloat = Union[float, np.ndarray]


def damping_number(lam_eff: ArrayOrFloat, U: float, L: float, alpha: float = 2.0) -> ArrayOrFloat:
    """lam_code = lam_eff U^alpha L (dimensionless damping number)."""
    return lam_eff * U**alpha * L


def lambda_eff_si(rho: ArrayOrFloat, ell: float, G: float = K.G, c: float = K.C) -> ArrayOrFloat:
    """Effective coupling G rho ell / c^3 in SI (s m^-2) for alpha = 2.

    ``ell`` is the length the vacuum-response hypothesis has to supply;
    without it G rho / c^3 is one power of length short (s m^-3).
    """
    return G * rho * ell / c**3


def physical_damping_number(rho: float, ell: float, U: float, L: float,
                            G: float = K.G, c: float = K.C) -> float:
    """lam_code for a physical flow: G rho ell U^2 L / c^3.  For any
    terrestrial flow this is far below 1e-20."""
    return float(damping_number(lambda_eff_si(rho, ell, G, c), U, L, 2.0))


def lambda_field_from_density(rho: np.ndarray, G: float = 1.0, c: float = 1.0,
                              ell: float = 1.0) -> np.ndarray:
    """Per-cell coupling lam(x) = G rho(x) ell / c^3 for an Eulerian solver
    (code units by default).  Pass the result as ``lam`` to
    :class:`~moss_reg.fluid.spectral.SpectralNS2D` /
    :class:`~moss_reg.fluid.spectral3d.SpectralNS3D`."""
    return G * np.asarray(rho, dtype=np.float64) * ell / c**3
