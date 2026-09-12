"""Exact dimensional bookkeeping for the formulas used in moss-reg.

A :class:`Dim` stores rational exponents of (mass, length, time).  A
:class:`Quantity` couples a value with a ``Dim`` so that products,
quotients and powers propagate units automatically and sums of
mismatched units raise :class:`DimensionError`.

:func:`check_formulas` evaluates every dimensional claim made in
README.md and returns PASS / FAIL / XFAIL (known open issue) records; it
is the backend of ``moss-reg validate`` and of ``tests/test_dimensions.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Union

from . import constants as K

__all__ = [
    "Dim",
    "DimensionError",
    "Quantity",
    "Check",
    "DIMENSIONLESS",
    "MASS",
    "LENGTH",
    "TIME",
    "VELOCITY",
    "ACCELERATION",
    "DENSITY",
    "FORCE",
    "PRESSURE",
    "ENERGY",
    "ACTION",
    "ODE_COUPLING",
    "C",
    "G",
    "HBAR",
    "xi",
    "lambda_",
    "lambda_c5",
    "f_p",
    "lambda_eff",
    "planck_length",
    "planck_time",
    "planck_mass",
    "check_formulas",
    "all_passed",
]

Number = Union[int, float, Fraction]


class DimensionError(ValueError):
    """Raised when quantities of different dimension are added or compared."""


def _frac(x: Number) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x).limit_denominator(1000)


@dataclass(frozen=True)
class Dim:
    """Exponents of mass (kg), length (m) and time (s)."""

    M: Fraction = Fraction(0)
    L: Fraction = Fraction(0)
    T: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "M", _frac(self.M))
        object.__setattr__(self, "L", _frac(self.L))
        object.__setattr__(self, "T", _frac(self.T))

    def __mul__(self, other: "Dim") -> "Dim":
        return Dim(self.M + other.M, self.L + other.L, self.T + other.T)

    def __truediv__(self, other: "Dim") -> "Dim":
        return Dim(self.M - other.M, self.L - other.L, self.T - other.T)

    def __pow__(self, p: Number) -> "Dim":
        p = _frac(p)
        return Dim(self.M * p, self.L * p, self.T * p)

    @property
    def is_dimensionless(self) -> bool:
        return self.M == 0 and self.L == 0 and self.T == 0

    def __str__(self) -> str:
        parts = []
        for sym, e in (("kg", self.M), ("m", self.L), ("s", self.T)):
            if e == 0:
                continue
            parts.append(sym if e == 1 else f"{sym}^{e}")
        return " ".join(parts) if parts else "1"


DIMENSIONLESS = Dim()
MASS = Dim(M=1)
LENGTH = Dim(L=1)
TIME = Dim(T=1)
VELOCITY = LENGTH / TIME
ACCELERATION = VELOCITY / TIME
DENSITY = MASS / LENGTH**3
FORCE = MASS * ACCELERATION
PRESSURE = FORCE / LENGTH**2
ENERGY = FORCE * LENGTH
ACTION = ENERGY * TIME

# [lambda_eff] required by  dv/dt = -lambda_eff |v|^2 v  :  s m^-2
ODE_COUPLING = ACCELERATION / VELOCITY**3


@dataclass(frozen=True)
class Quantity:
    """A value with a dimension.  Arithmetic propagates the dimension."""

    value: float
    dim: Dim = DIMENSIONLESS

    def _coerce(self, other: Union["Quantity", Number]) -> "Quantity":
        if isinstance(other, Quantity):
            return other
        return Quantity(float(other), DIMENSIONLESS)

    def __mul__(self, other: Union["Quantity", Number]) -> "Quantity":
        o = self._coerce(other)
        return Quantity(self.value * o.value, self.dim * o.dim)

    __rmul__ = __mul__

    def __truediv__(self, other: Union["Quantity", Number]) -> "Quantity":
        o = self._coerce(other)
        return Quantity(self.value / o.value, self.dim / o.dim)

    def __rtruediv__(self, other: Number) -> "Quantity":
        return Quantity(float(other) / self.value, DIMENSIONLESS / self.dim)

    def __pow__(self, p: Number) -> "Quantity":
        return Quantity(self.value ** float(p), self.dim**p)

    def __neg__(self) -> "Quantity":
        return Quantity(-self.value, self.dim)

    def _same_dim(self, other: "Quantity", op: str) -> None:
        if self.dim != other.dim:
            raise DimensionError(
                f"cannot {op} [{self.dim}] and [{other.dim}]"
            )

    def __add__(self, other: "Quantity") -> "Quantity":
        self._same_dim(other, "add")
        return Quantity(self.value + other.value, self.dim)

    def __sub__(self, other: "Quantity") -> "Quantity":
        self._same_dim(other, "subtract")
        return Quantity(self.value - other.value, self.dim)

    def sqrt(self) -> "Quantity":
        return self ** Fraction(1, 2)

    def __str__(self) -> str:
        return f"{self.value:.6g} [{self.dim}]"


# ---------------------------------------------------------------------------
# SI constants as dimensioned quantities
# ---------------------------------------------------------------------------
C = Quantity(K.C, VELOCITY)
G = Quantity(K.G, LENGTH**3 / (MASS * TIME**2))
HBAR = Quantity(K.HBAR, ACTION)
EIGHT_PI = 8.0 * math.pi


def xi() -> Quantity:
    """Xi = c^4 / (8 pi G)."""
    return C**4 / (EIGHT_PI * G)


def lambda_() -> Quantity:
    """lambda = G / c^3."""
    return G / C**3


def lambda_c5() -> Quantity:
    """1 / (8 pi Xi c)  (evaluates to G / c^5)."""
    return 1.0 / (EIGHT_PI * xi() * C)


def f_p() -> Quantity:
    """F_P = c^3 / G, the reciprocal of lambda."""
    return C**3 / G


def lambda_eff(rho: Quantity) -> Quantity:
    """lambda_eff = lambda * rho = G rho / c^3."""
    return lambda_() * rho


def planck_length() -> Quantity:
    return (HBAR * G / C**3).sqrt()


def planck_time() -> Quantity:
    return planck_length() / C


def planck_mass() -> Quantity:
    return (HBAR * C / G).sqrt()


# ---------------------------------------------------------------------------
# Formula checks (backend of ``moss-reg validate``)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    name: str
    status: str          # "PASS" | "FAIL" | "XFAIL" | "XPASS"
    detail: str

    @property
    def ok(self) -> bool:
        """True unless the check is a hard FAIL."""
        return self.status != "FAIL"


def _check(name: str, passed: bool, detail: str, known_issue: bool = False) -> Check:
    if known_issue:
        status = "XPASS" if passed else "XFAIL"
    else:
        status = "PASS" if passed else "FAIL"
    return Check(name, status, detail)


def _rel_close(a: float, b: float, rel: float = 1e-12) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b))


def check_formulas() -> List[Check]:
    """Evaluate the dimensional and numerical claims of the README."""
    lam = lambda_()
    x = xi()
    lc5 = lambda_c5()
    fp = f_p()
    rho = Quantity(1000.0, DENSITY)           # any density; only the dimension matters
    leff = lambda_eff(rho)
    lp, tp, mp = planck_length(), planck_time(), planck_mass()

    checks = [
        _check(
            "lambda = G/c^3 has dimension s kg^-1",
            lam.dim == TIME / MASS,
            f"[lambda] = {lam.dim}",
        ),
        _check(
            "lambda * F_P is dimensionless and equals 1",
            (lam * fp).dim.is_dimensionless and _rel_close((lam * fp).value, 1.0),
            f"lambda * F_P = {(lam * fp)}",
        ),
        _check(
            "Xi = c^4/(8 pi G) has the dimension of a force (kg m s^-2)",
            x.dim == FORCE,
            f"[Xi] = {x.dim}; Xi = {x.value:.6g}",
        ),
        _check(
            "Xi is NOT a pressure / energy density (Pa)",
            x.dim != PRESSURE,
            f"[Xi] = {x.dim}, [Pa] = {PRESSURE}",
        ),
        _check(
            "1/(8 pi Xi c) equals G/c^5 (value and dimension)",
            lc5.dim == (G / C**5).dim and _rel_close(lc5.value, (G / C**5).value),
            f"1/(8 pi Xi c) = {lc5}; G/c^5 = {G / C**5}",
        ),
        _check(
            "identity lambda = 1/(8 pi Xi c) is rejected (differs by c^2)",
            lam.dim != lc5.dim and _rel_close(lc5.value, lam.value / K.C**2),
            f"[lambda] = {lam.dim}, [1/(8 pi Xi c)] = {lc5.dim}",
        ),
        _check(
            "Planck length / time / mass have dimensions m / s / kg",
            lp.dim == LENGTH and tp.dim == TIME and mp.dim == MASS,
            f"l_P = {lp}, t_P = {tp}, m_P = {mp}",
        ),
        _check(
            "ODE coupling: [G rho / c^3] == [s m^-2] required by dv/dt = -lambda_eff |v|^2 v",
            leff.dim == ODE_COUPLING,
            f"[G rho/c^3] = {leff.dim}, required {ODE_COUPLING}: "
            "one power of length is missing in the SI form (open issue)",
            known_issue=True,
        ),
    ]
    return checks


def all_passed(checks: List[Check]) -> bool:
    """True if no check is a hard FAIL (XFAIL known issues are tolerated)."""
    return all(c.ok for c in checks)
