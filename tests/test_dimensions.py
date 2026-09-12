"""Dimensional consistency of the fundamental formulas (README claims)."""

from fractions import Fraction

import pytest

from moss_reg import constants as K
from moss_reg import dimensions as D


def test_dim_algebra_and_repr():
    assert D.VELOCITY == D.Dim(L=1, T=-1)
    assert (D.LENGTH / D.TIME) ** 2 == D.Dim(L=2, T=-2)
    assert D.FORCE == D.Dim(M=1, L=1, T=-2)
    assert str(D.FORCE) == "kg m s^-2"
    assert str(D.DIMENSIONLESS) == "1"
    assert str(D.Dim(M=Fraction(1, 2))) == "kg^1/2"
    assert D.FORCE / D.FORCE == D.DIMENSIONLESS


def test_quantity_rejects_mismatched_sums():
    with pytest.raises(D.DimensionError):
        _ = D.C + D.G
    q = D.Quantity(2.0, D.LENGTH) + D.Quantity(3.0, D.LENGTH)
    assert q.value == 5.0 and q.dim == D.LENGTH
    assert (D.Quantity(4.0, D.LENGTH**2).sqrt()).dim == D.LENGTH


def test_lambda_is_time_per_mass():
    lam = D.lambda_()
    assert lam.dim == D.TIME / D.MASS
    assert lam.value == pytest.approx(K.LAMBDA, rel=1e-15, abs=0.0)
    assert (lam * D.f_p()).dim.is_dimensionless
    assert (lam * D.f_p()).value == pytest.approx(1.0, rel=1e-14)


def test_xi_is_a_force_not_a_pressure():
    xi = D.xi()
    assert xi.dim == D.FORCE
    assert xi.dim != D.PRESSURE
    assert xi.value == pytest.approx(K.XI, rel=1e-15, abs=0.0)


def test_readme_identity_with_xi_is_false():
    """1 / (8 pi Xi c) = G / c^5, not G / c^3."""
    lc5 = D.lambda_c5()
    assert lc5.dim == (D.G / D.C**5).dim
    assert lc5.value == pytest.approx(K.G / K.C**5, rel=1e-12, abs=0.0)
    assert lc5.dim != D.lambda_().dim
    assert lc5.dim / D.lambda_().dim == D.VELOCITY ** -2


def test_planck_scales_dimensions():
    assert D.planck_length().dim == D.LENGTH
    assert D.planck_time().dim == D.TIME
    assert D.planck_mass().dim == D.MASS
    assert D.planck_length().value == pytest.approx(K.L_P, rel=1e-12, abs=0.0)


def test_ode_coupling_requirement():
    # dv/dt = -lambda_eff |v|^2 v  =>  [lambda_eff] = s m^-2
    assert D.ODE_COUPLING == D.Dim(L=-2, T=1)


@pytest.mark.xfail(
    strict=True,
    reason="open issue: G rho / c^3 has dimension s m^-3, the ODE needs s m^-2 "
    "(one power of length missing in the SI form of lambda_eff)",
)
def test_lambda_eff_si_form_is_dimensionally_consistent_with_ode():
    rho = D.Quantity(1.0, D.DENSITY)
    assert D.lambda_eff(rho).dim == D.ODE_COUPLING


def test_check_formulas_has_no_hard_failures_and_one_known_issue():
    checks = D.check_formulas()
    assert D.all_passed(checks)
    statuses = {c.name: c.status for c in checks}
    known = [c for c in checks if c.status in ("XFAIL", "XPASS")]
    assert len(known) == 1 and known[0].status == "XFAIL"
    assert "ODE coupling" in known[0].name
    assert all(v == "PASS" for k, v in statuses.items() if k != known[0].name)
