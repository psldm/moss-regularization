"""Tests for the 2D spectral NS solver with Moss damping."""

import numpy as np
import pytest

from moss_reg.fluid.spectral import SpectralNS2D


def _run_series(n, nu, lam, t_end, dt_max=0.005, snap_dt=0.05):
    """Histories including t = 0; ``snap_dt=None`` records every step."""
    solver = SpectralNS2D(n, nu=nu, lam=lam, dt_max=dt_max)
    solver.set_taylor_green()
    times, e, o, l4 = [0.0], [], [], []
    ei, oi, li = solver.measure()
    e.append(ei)
    o.append(oi)
    l4.append(li)
    t_next = snap_dt if snap_dt is not None else 0.0
    while solver.time < t_end - 1e-14:
        solver.step(min(solver.cfl_dt(), t_end - solver.time))
        if snap_dt is None or solver.time >= t_next - 1e-14:
            times.append(solver.time)
            ei, oi, li = solver.measure()
            e.append(ei)
            o.append(oi)
            l4.append(li)
            if snap_dt is not None:
                t_next += snap_dt
    return (
        solver,
        np.asarray(times),
        np.asarray(e),
        np.asarray(o),
        np.asarray(l4),
    )


def test_taylor_green_initial_conditions():
    s = SpectralNS2D(64, nu=0.005, lam=0.0)
    s.set_taylor_green()
    e, o, l4 = s.measure()
    assert e == pytest.approx(0.25, rel=1e-12)
    assert o == pytest.approx(0.5, rel=1e-12)
    u, v = s.velocity()
    assert np.abs(u - np.sin(s.x) * np.cos(s.y)).max() < 1e-12
    assert np.abs(v + np.cos(s.x) * np.sin(s.y)).max() < 1e-12
    assert l4 > 0.0


def test_taylor_green_exact_decay_without_damping():
    nu = 0.01
    _, t, e, o, _ = _run_series(64, nu, 0.0, 3.0)
    e_exact = 0.25 * np.exp(-4.0 * nu * t)
    o_exact = 0.5 * np.exp(-4.0 * nu * t)
    assert np.abs(e - e_exact).max() / 0.25 < 1e-6
    assert np.abs(o - o_exact).max() / 0.5 < 1e-6


def test_energy_identity_with_damping():
    """E(t) = E0 - 2 nu int Omega - lam int ||u||_L4^4."""
    nu, lam = 0.005, 0.5
    _, t, e, o, l4 = _run_series(64, nu, lam, 5.0)
    predicted = 0.25 - np.trapezoid(2.0 * nu * o + lam * l4**4, t)
    residual = e[-1] - predicted
    assert abs(residual) < 2e-3


def test_damping_dissipates_faster_than_classical_ns():
    nu, lam = 0.005, 0.5
    _, t0, e0, _, _ = _run_series(64, nu, 0.0, 5.0)
    _, t1, e1, _, _ = _run_series(64, nu, lam, 5.0)
    assert e1[-1] < 0.8 * e0[-1]
    # both decay monotonically
    assert np.all(np.diff(e0) < 0.0)
    assert np.all(np.diff(e1) < 0.0)


def test_flow_stays_divergence_free():
    s = SpectralNS2D(64, nu=0.005, lam=0.5)
    s.set_taylor_green()
    for _ in range(100):
        s.step(0.005)
    u, v = s.velocity()
    ux = np.fft.ifft2(1j * s.kx * np.fft.fft2(u)).real
    vy = np.fft.ifft2(1j * s.ky * np.fft.fft2(v)).real
    assert np.abs(ux + vy).max() < 1e-10


def test_random_initial_condition_runs():
    s = SpectralNS2D(64, nu=0.005, lam=0.5)
    s.set_random(n_modes=6, seed=1)
    e0, o0, l40 = s.measure()
    assert e0 > 0.0 and o0 > 0.0
    while s.time < 0.5:
        s.step(min(s.cfl_dt(), 0.5 - s.time))
    e, o, l4 = s.measure()
    assert np.isfinite(e) and 0.0 < e <= e0
    assert np.isfinite(o) and np.isfinite(l4)


def test_enstrophy_l4_positive_and_finite():
    _, _, e, o, l4 = _run_series(64, 0.005, 0.5, 2.0)
    assert np.all(o > 0.0) and np.all(l4 > 0.0) and np.all(e > 0.0)
    assert np.all(np.isfinite(o)) and np.all(np.isfinite(l4))


# ---------------------------------------------------------------------------
# Cubic-term dealiasing, exact L4 quadrature, damping stiffness limit
# ---------------------------------------------------------------------------


def _single_mode_solver(n, k, cubic_dealias, lam=1.0):
    s = SpectralNS2D(n, nu=0.0, lam=lam, dt_max=0.005, cubic_dealias=cubic_dealias)
    s.set_field(np.cos(k * s.y), 0.0)  # u = cos(k y) x-hat: solenoidal, u.grad u = 0
    return s


def test_cubic_term_is_aliased_by_two_thirds_rule_and_fixed_by_padding():
    """u = cos(15 y) x-hat on n = 48:  |u|^2 u = (3 cos 15y + cos 45y) / 4.

    The mode 45 exceeds n/2 and aliases to |k| = 3 on the native grid,
    inside the retained band |k| <= 16 -- the 2/3 rule does not dealias
    cubic products.  The padded evaluation must not contain it.
    """
    n, k = 48, 15
    for cubic_dealias, spurious in ((False, 0.125), (True, 0.0)):
        s = _single_mode_solver(n, k, cubic_dealias)
        rhs_u, rhs_v = s._rhs(s.u_hat, s.v_hat)
        a = rhs_u / n**2  # complex Fourier amplitudes
        assert abs(a[k, 0]) == pytest.approx(3.0 / 8.0, rel=1e-10)      # -lam * 3/4 cos(15y)
        assert abs(a[3, 0]) == pytest.approx(spurious, abs=1e-10)      # aliased cos(45y)
        assert np.abs(rhs_v).max() < 1e-12


def test_l4_norm_quadrature_is_exact_on_padded_grid():
    """u = cos(12 y) x-hat on n = 48: <|u|^4> = 3/8.  cos(48 y) aliases to
    the mean on the native grid (giving 1/2); the padded quadrature is exact."""
    n, k = 48, 12
    s = _single_mode_solver(n, k, cubic_dealias=True)
    u, v = s.velocity()
    native = float(np.mean((u * u + v * v) ** 2))
    assert native == pytest.approx(0.5, rel=1e-12)          # aliased
    assert s.l4_norm() ** 4 == pytest.approx(3.0 / 8.0, rel=1e-12)  # exact
    _, _, l4 = s.measure()
    assert l4 == pytest.approx((3.0 / 8.0) ** 0.25, rel=1e-12)


def test_cfl_dt_includes_damping_stiffness():
    s = SpectralNS2D(32, nu=1e-3, lam=200.0, dt_max=1.0)
    s.set_taylor_green()
    lim = s.dt_limits(cfl=0.5)
    umax2 = 1.0  # max |u|^2 of the Taylor-Green vortex
    assert lim["damping"] == pytest.approx(0.5 * 2.0 / (3.0 * 200.0 * umax2), rel=1e-6)
    assert s.cfl_dt(0.5) <= lim["damping"]
    s0 = SpectralNS2D(32, nu=1e-3, lam=0.0, dt_max=1.0)
    s0.set_taylor_green()
    assert np.isinf(s0.dt_limits()["damping"])
    assert s0.cfl_dt() > s.cfl_dt()


def test_energy_identity_with_damping_every_step_quadrature():
    """With every-step trapezoid quadrature the residual is O(dt^2) ~ 1e-6,
    i.e. limited by the time quadrature, not by aliasing."""
    nu, lam = 0.005, 0.5
    _, t, e, o, l4 = _run_series(48, nu, lam, 3.0, snap_dt=None)
    predicted = e[0] - np.trapezoid(2.0 * nu * o + lam * l4**4, t)
    assert abs(e[-1] - predicted) < 1e-5
