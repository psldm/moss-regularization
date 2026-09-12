"""Spatially varying coupling lam(x) and the unit helpers."""

import numpy as np
import pytest

from moss_reg import dimensions as D
from moss_reg.diagnostics import DiagnosticsLog
from moss_reg.fluid import (
    SpectralNS2D,
    SpectralNS3D,
    damping_number,
    lambda_eff_si,
    lambda_field_from_density,
    physical_damping_number,
)


def test_constant_field_equals_scalar_coupling():
    s_scalar = SpectralNS2D(32, nu=0.01, lam=0.5)
    s_field = SpectralNS2D(32, nu=0.01, lam=np.full((32, 32), 0.5))
    for s in (s_scalar, s_field):
        s.set_taylor_green()
    r0 = s_scalar._rhs(s_scalar.u_hat, s_scalar.v_hat)
    r1 = s_field._rhs(s_field.u_hat, s_field.v_hat)
    np.testing.assert_allclose(r1[0], r0[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(r1[1], r0[1], rtol=1e-12, atol=1e-12)
    assert s_field.lam == pytest.approx(0.5) and s_field.lam_max == pytest.approx(0.5)
    assert s_field.dt_limits()["damping"] == pytest.approx(s_scalar.dt_limits()["damping"])


def test_varying_field_energy_budget_closes_2d():
    n, nu = 32, 0.005
    x = 2 * np.pi * np.arange(n) / n
    X, Y = np.meshgrid(x, x)
    lam = lambda_field_from_density(1.0 + 0.8 * np.cos(X) * np.cos(2 * Y), G=1.0, c=1.0, ell=0.5)
    assert np.all(lam >= 0.0)
    s = SpectralNS2D(n, nu=nu, lam=lam, dt_max=0.005)
    s.set_taylor_green()
    log = DiagnosticsLog()
    log.record(s.time, **s.diagnostics())
    while s.time < 1.0 - 1e-14:
        s.step(min(s.cfl_dt(), 1.0 - s.time))
        log.record(s.time, **s.diagnostics())
    b = log.budget(nu=nu)
    assert b.max_abs_residual < 1e-6
    assert b.summary()["E_vac_final"] > 0.0


def test_varying_field_energy_budget_closes_3d():
    n, nu = 16, 0.02
    s = SpectralNS3D(n, nu=nu, lam=0.0, dt_max=0.01)
    lam = 0.3 + 0.2 * np.cos(s.X[0]) * np.cos(s.X[2])
    s.set_lambda(lam)
    s.set_taylor_green()
    log = DiagnosticsLog()
    log.record(s.time, **s.diagnostics())
    while s.time < 0.5 - 1e-14:
        s.step(min(s.cfl_dt(), 0.5 - s.time))
        log.record(s.time, **s.diagnostics())
    assert log.budget(nu=nu).max_abs_residual < 1e-6


def test_field_shape_is_validated():
    with pytest.raises(ValueError):
        SpectralNS2D(16, lam=np.ones((8, 8)))


def test_unit_helpers_and_dimensions():
    rho = D.Quantity(1000.0, D.DENSITY)
    ell = D.Quantity(1.0, D.LENGTH)
    lam = D.G * rho * ell / D.C**3
    assert lam.dim == D.ODE_COUPLING                       # s m^-2 once the length is supplied
    assert lambda_eff_si(1000.0, 1.0) == pytest.approx(lam.value, rel=1e-12)
    assert physical_damping_number(1000.0, 1.0, 1e3, 1.0) == pytest.approx(lam.value * 1e6, rel=1e-12)
    assert physical_damping_number(1000.0, 1.0, 1e3, 1.0) < 1e-26
    assert damping_number(2.0, 3.0, 5.0) == pytest.approx(2.0 * 9.0 * 5.0)
    assert damping_number(2.0, 3.0, 5.0, alpha=3.0) == pytest.approx(2.0 * 27.0 * 5.0)
    np.testing.assert_allclose(lambda_field_from_density(np.array([1.0, 2.0]), G=2.0, c=2.0, ell=4.0),
                               [1.0, 2.0])
