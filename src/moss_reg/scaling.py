"""Scaling (homogeneity) checks for inequalities between Sobolev-type norms.

A norm term ``|D^j u|_p^k`` on R^d transforms under the two symmetries

    dilation   u(x) -> u(mu x):    |D^j u|_p -> mu^(j - d/p) |D^j u|_p
    amplitude  u    -> A u    :    |D^j u|_p -> A |D^j u|_p

An inequality  prod_i |D^{j_i} u|_{p_i}^{k_i}  <=  C  prod_l |D^{j_l} u|_{p_l}^{k_l}
with a constant C independent of u can hold only if the total dilation
exponent and the total amplitude exponent agree on the two sides: otherwise
mu -> 0 or mu -> infinity (or A -> 0 / infinity) makes the ratio LHS/RHS
unbounded for any fixed u for which the left side is nonzero.  This is the
thirty-second test that rejects the interpolation inequality of version 1
of the paper, and it is the first thing to run on any inequality produced
by a person or a model.

The module also gives the scaling of the terms of the damped Navier-Stokes
equation under the Navier-Stokes symmetry u -> mu u(mu x, mu^2 t), which is
what makes alpha = 2 the critical damping exponent.

String syntax (spaces or ``*`` separate factors)::

    |D1 u|_2^2 <= |u|_4^{4/3} |D2 u|_2^{2/3}
    |u|_6 <= |D1 u|_2
    |u|_4^4 <= |u|_2^2 |D1 u|_2^2

``D<j>`` is the j-th derivative (``u`` alone is j = 0), ``_p`` the Lebesgue
exponent (``inf`` allowed), ``^k`` the power (integer, decimal or a/b).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple, Union

__all__ = [
    "Term",
    "Inequality",
    "check_inequality",
    "parse_inequality",
    "interpolation_exponent",
    "ns_term_scaling",
    "damping_exponent_scaling",
]

Number = Union[int, float, str, Fraction]


def _frac(x: Number) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s in ("inf", "oo", "infinity"):
            return Fraction(0)          # 1/p = 0 handled via p_inv
        return Fraction(s)
    return Fraction(x).limit_denominator(10_000)


@dataclass(frozen=True)
class Term:
    """|D^j u|_p^k  (p = None means p = infinity)."""

    j: int
    p: Union[Fraction, None]
    k: Fraction

    @property
    def p_inv(self) -> Fraction:
        return Fraction(0) if self.p is None else 1 / self.p

    def dilation_exponent(self, d: int) -> Fraction:
        return self.k * (self.j - d * self.p_inv)

    def amplitude_exponent(self) -> Fraction:
        return self.k

    def __str__(self) -> str:
        p = "inf" if self.p is None else str(self.p)
        base = "u" if self.j == 0 else f"D{self.j} u"
        k = "" if self.k == 1 else f"^{{{self.k}}}"
        return f"|{base}|_{p}{k}"


@dataclass(frozen=True)
class Inequality:
    lhs: Tuple[Term, ...]
    rhs: Tuple[Term, ...]
    d: int = 3

    def exponents(self) -> Dict[str, Fraction]:
        return {
            "lhs_dilation": sum((t.dilation_exponent(self.d) for t in self.lhs), Fraction(0)),
            "rhs_dilation": sum((t.dilation_exponent(self.d) for t in self.rhs), Fraction(0)),
            "lhs_amplitude": sum((t.amplitude_exponent() for t in self.lhs), Fraction(0)),
            "rhs_amplitude": sum((t.amplitude_exponent() for t in self.rhs), Fraction(0)),
        }

    def __str__(self) -> str:
        return " ".join(map(str, self.lhs)) + " <= C " + " ".join(map(str, self.rhs))


_TERM = re.compile(r"\|\s*(?:D(\d+)\s+)?u\s*\|\s*_\s*\{?\s*([0-9./]+|inf|oo)\s*\}?\s*(?:\^\s*\{?\s*([-0-9./]+)\s*\}?)?")


def _parse_side(text: str) -> Tuple[Term, ...]:
    terms = []
    pos = 0
    for m in _TERM.finditer(text):
        gap = text[pos:m.start()].strip().strip("*").strip()
        if gap:
            raise ValueError(f"unparsed text {gap!r} in {text!r}")
        j = int(m.group(1)) if m.group(1) else 0
        p_txt = m.group(2)
        p = None if p_txt in ("inf", "oo") else _frac(p_txt)
        k = _frac(m.group(3)) if m.group(3) else Fraction(1)
        terms.append(Term(j, p, k))
        pos = m.end()
    tail = text[pos:].strip().strip("*").strip()
    if tail or not terms:
        raise ValueError(f"cannot parse side {text!r}")
    return tuple(terms)


def parse_inequality(text: str, d: int = 3) -> Inequality:
    if "<=" in text:
        left, right = text.split("<=", 1)
    elif "≤" in text:
        left, right = text.split("≤", 1)
    else:
        raise ValueError("inequality must contain '<='")
    right = re.sub(r"^\s*C\s*\*?", "", right)        # optional leading constant
    return Inequality(_parse_side(left), _parse_side(right), d)


def check_inequality(ineq: Union[str, Inequality], d: int = 3) -> Dict[str, object]:
    """Verdict on the scaling admissibility of an inequality.

    Returns a dict with the four exponents, ``homogeneous`` (both match),
    and a human-readable ``reason``.  ``homogeneous = False`` proves the
    inequality false for every constant C (for any u with nonzero left side);
    ``True`` is necessary, not sufficient, for validity.
    """
    if isinstance(ineq, str):
        ineq = parse_inequality(ineq, d)
    e = ineq.exponents()
    dil_ok = e["lhs_dilation"] == e["rhs_dilation"]
    amp_ok = e["lhs_amplitude"] == e["rhs_amplitude"]
    reasons = []
    if not dil_ok:
        gap = e["lhs_dilation"] - e["rhs_dilation"]
        direction = "mu -> 0 (spreading u out)" if gap < 0 else "mu -> infinity (concentrating u)"
        reasons.append(
            f"dilation exponents differ: LHS ~ mu^{e['lhs_dilation']}, RHS ~ mu^{e['rhs_dilation']}; "
            f"LHS/RHS ~ mu^{gap} is unbounded as {direction}"
        )
    if not amp_ok:
        reasons.append(
            f"amplitude exponents differ: LHS ~ A^{e['lhs_amplitude']}, RHS ~ A^{e['rhs_amplitude']}"
        )
    return {
        "inequality": str(ineq),
        "d": ineq.d,
        **{k: str(v) for k, v in e.items()},
        "homogeneous": dil_ok and amp_ok,
        "reason": "; ".join(reasons) if reasons else "scaling-admissible (necessary condition satisfied)",
    }


def interpolation_exponent(j: int, p: Number, m: int, r: Number, q: Number, d: int = 3) -> Fraction:
    """The Gagliardo-Nirenberg exponent theta in
    |D^j u|_p <= C |D^m u|_r^theta |u|_q^(1-theta), from
    1/p = j/d + theta (1/r - m/d) + (1 - theta)/q, i.e. the unique theta that
    makes the inequality scaling-admissible (must satisfy j/m <= theta <= 1)."""
    p, r, q = _frac(p), _frac(r), _frac(q)
    p_inv = 0 if p == 0 else 1 / p
    r_inv = 0 if r == 0 else 1 / r
    q_inv = 0 if q == 0 else 1 / q
    denom = (r_inv - Fraction(m, d)) - q_inv
    if denom == 0:
        raise ValueError("degenerate exponents")
    return (p_inv - Fraction(j, d) - q_inv) / denom


def ns_term_scaling(term: str) -> int:
    """Exponent n such that the term scales as mu^n under u -> mu u(mu x, mu^2 t)."""
    table = {
        "dt u": 3, "u.grad u": 3, "grad p": 3, "nu lap u": 3,
    }
    if term in table:
        return table[term]
    raise KeyError(f"unknown term {term!r}; use damping_exponent_scaling for |u|^alpha u")


def damping_exponent_scaling(alpha: Number) -> Dict[str, object]:
    """Scaling of lam |u|^alpha u under the Navier-Stokes symmetry: mu^(alpha+1).
    Compared with the viscous term (mu^3): alpha = 2 critical, alpha > 2
    supercritical (dominant at small scales), alpha < 2 subcritical."""
    a = _frac(alpha)
    n = a + 1
    if n == 3:
        cls = "critical (same scaling as nu lap u)"
    elif n > 3:
        cls = "supercritical damping (dominates viscosity at small scales)"
    else:
        cls = "subcritical damping (negligible at small scales)"
    return {"alpha": str(a), "exponent": str(n), "viscous_exponent": "3", "class": cls}
