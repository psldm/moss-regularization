"""Tests for the 2D spectral NS solver with Moss damping."""

import numpy as np
import pytest

from moss_reg.fluid.spectral import SpectralNS2D


def _run_series(n, nu, lam, t_end, dt_max=0.005, snap_dt=0.05):
    solver = SpectralNS2D(n, nu=nu, lam=lam, dt_max=dt_max)
    solver.set_taylor_green()
    times, e, o, l4 = [], [], [], []
    t_next = 0.0
    while solver.time < t_end - 1e-14:
        solver.step(min(solver.cfl_dt(), t_end - solver.time))
        if solver.time >= t_next - 1e-14:
            times.append(solver.time)
            ei, oi, li = solver.measure()
            e.append(ei)
            o.append(oi)
            l4.append(li)
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
