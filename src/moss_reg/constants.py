"""Fundamental and derived physical constants for moss regularization.

The vacuum damping term is

    dv/dt = -lambda_eff * |v|^2 * v

with the canonical coupling

    lambda = G / c^3   [s / kg].

Notes on the internal conventions (SI, CODATA 2018):
  * The "vacuum shear modulus" of the vacuum-as-elastic-medium picture is
        Xi = c^4 / (8 pi G).
    Dimensionally this is a FORCE (kg m s^-2; c^4/G is the Planck force),
    not a pressure / energy density.  A modulus in Pa requires dividing
    by an area; the theory does not currently specify which.
  * The reciprocal Planck force is fixed by the spec identity
        lambda = 1 / F_P,
    which requires  F_P = c^3 / G  (a Planck mass-flow constant,
    kg / s).  This differs from the textbook Planck force
    c^4 / G = m_P c^2 / l_P by one factor of c; both are provided.
  * The identity  lambda = 1 / (8 pi Xi c)  quoted in early drafts is
    FALSE: 1 / (8 pi Xi c) evaluates to G / c^5 = lambda / c^2, both
    numerically and dimensionally (s^3 kg^-1 m^-2 versus s kg^-1).  The
    combination is exposed as ``lambda_c5`` for reference only.
  * Open dimensional issue: with lambda_eff = lambda * rho the ODE
        dv/dt = -lambda_eff |v|^2 v
    requires [lambda_eff] = s m^-2, whereas G rho / c^3 has dimension
    s m^-3 -- one power of length is missing in the SI form.  The
    numerical benchmarks work in code units (G = 1, mass and length
    units = 1) where this is invisible.  See ``moss_reg.dimensions`` and
    ``moss-reg validate``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import numpy as np

__all__ = [
    "C",
    "G",
    "HBAR",
    "XI",
    "LAMBDA",
    "lambda_",
    "lambda_c5",
    "L_P",
    "T_P",
    "M_P",
    "RHO_P",
    "F_P",
    "F_P_STANDARD",
    "EIGHT_PI_G",
    "UnitSystem",
    "natural_units",
    "planck_units",
]

# ---------------------------------------------------------------------------
# Fundamental constants (SI)
# ---------------------------------------------------------------------------
C: float = 299792458.0            # exact speed of light, m/s
G: float = 6.67430e-11            # Newtonian constant of gravitation,
                                  # m^3 kg^-1 s^-2 (CODATA 2018)
HBAR: float = 1.054571817e-34     # reduced Planck constant, J s (exact)

EIGHT_PI_G: float = 8.0 * math.pi * G

# ---------------------------------------------------------------------------
# Derived vacuum properties
# ---------------------------------------------------------------------------
# "Vacuum shear modulus" c^4 / (8 pi G): dimension of FORCE, N = kg m s^-2
# (it is the Planck force / 8 pi, NOT a pressure).
XI: float = C**4 / EIGHT_PI_G

# Canonical damping coupling, s / kg
LAMBDA: float = G / C**3
lambda_ = LAMBDA                    # ``lambda`` is a Python keyword

# 1 / (8 pi Xi c) == G / c^5 == lambda / c^2   (NOT equal to lambda)
lambda_c5: float = G / C**5

# ---------------------------------------------------------------------------
# Planck scales (SI)
# ---------------------------------------------------------------------------
L_P: float = math.sqrt(HBAR * G / C**3)          # Planck length, m
T_P: float = L_P / C                             # Planck time, s
M_P: float = math.sqrt(HBAR * C / G)             # Planck mass, kg
RHO_P: float = M_P / L_P**3                      # Planck density, kg / m^3
F_P: float = C**3 / G                            # reciprocal of lambda, kg / s
F_P_STANDARD: float = C**4 / G                   # textbook Planck force, N


@dataclass(frozen=True)
class UnitSystem:
    """A consistent set of physical constants.

    By default this is the SI system.  Passing ``c=1, G=1`` (optionally
    ``hbar=1``) produces the dimensionless normalization used by the
    numerical scheme, in which ``lambda == 1``.
    """

    c: float = C
    G: float = G
    hbar: float = HBAR

    @property
    def eight_pi_G(self) -> float:
        return 8.0 * math.pi * self.G

    @property
    def xi(self) -> float:
        """Vacuum shear modulus in this unit system."""
        return self.c**4 / self.eight_pi_G

    @property
    def lambda_(self) -> float:
        """Damping coupling G / c^3."""
        return self.G / self.c**3

    @property
    def f_p(self) -> float:
        """Reciprocal Planck force c^3 / G."""
        return self.c**3 / self.G

    @property
    def l_p(self) -> float:
        return math.sqrt(self.hbar * self.G / self.c**3)

    @property
    def t_p(self) -> float:
        return self.l_p / self.c

    @property
    def m_p(self) -> float:
        return math.sqrt(self.hbar * self.c / self.G)

    @property
    def rho_p(self) -> float:
        return self.m_p / self.l_p**3

    @classmethod
    def natural(cls, hbar: float = 1.0) -> "UnitSystem":
        """Dimensionless normalization with c = G = 1  =>  lambda = 1."""
        return cls(c=1.0, G=1.0, hbar=hbar)


SI: UnitSystem = UnitSystem()
natural_units: UnitSystem = UnitSystem.natural()
planck_units: UnitSystem = UnitSystem.natural(hbar=1.0)


def velocity_coupling(system: UnitSystem = SI) -> float:
    """Return lambda for a given unit system."""
    return system.lambda_


def rescale(values: Union[np.ndarray, float], system: UnitSystem = SI) -> float:
    """Dimensional analysis helper placeholder (kept for API stability)."""
    return values
