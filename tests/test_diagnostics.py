"""Sobolev norms, energy budget, diagnostics log and particle invariants."""

import json

import numpy as np
import pytest

from moss_reg.diagnostics import DiagnosticsLog, EnergyBudget, field_norms, pad_spectrum
from moss_reg.fluid.spectral import SpectralNS2D
from moss_reg.particles.integrator import LagrangianSystem


def test_field_norms_taylor_green_2d():
    n = 64
    x = 2 * np.pi * np.arange(n) / n
    X, Y = np.meshgrid(x, x)                     # X along axis 1, Y along axis 0
    u = np.sin(X) * np.cos(Y)
    v = -np.cos(X) * np.sin(Y)
    d = field_norms([v, u])                      # axis-0 component first
    assert d["E"] == pytest.approx(0.25, rel=1e-12)
    assert d["l2"] == pytest.approx(np.sqrt(0.5), rel=1e-12)
    assert d["grad_l2_sq"] == pytest.approx(1.0, rel=1e-12)
    assert d["enstrophy"] == pytest.approx(0.5, rel=1e-12)
    assert d["grad_l2_sq"] == pytest.approx(2 * d["enstrophy"], rel=1e-12)
    assert d["l4_pow4"] == pytest.approx(5.0 / 16.0, rel=1e-12)
    assert d["linf"] == pytest.approx(1.0, rel=1e-6)
    assert d["div_max"] < 1e-12


def test_field_norms_3d_shear_mode():
    n = 16
    x = 2 * np.pi * np.arange(n) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u0 = np.sin(Y)                               # component along axis 0 depends on axis 1
    d = field_norms([u0, np.zeros_like(u0), np.zeros_like(u0)])
    assert d["E"] == pytest.approx(0.25, rel=1e-12)
    assert d["grad_l2_sq"] == pytest.approx(0.5, rel=1e-12)
    assert d["enstrophy"] == pytest.approx(0.25, rel=1e-12)
    assert d["div_max"] < 1e-12
    assert d["l4_pow4"] == pytest.approx(3.0 / 8.0, rel=1e-12)


def test_pad_spectrum_interpolates_exactly():
    n = 16
    x = 2 * np.pi * np.arange(n) / n
    f = np.cos(3 * x) + 0.5 * np.sin(5 * x)
    big = pad_spectrum(np.fft.fft(f), 2)
    xb = 2 * np.pi * np.arange(2 * n) / (2 * n)
    np.testing.assert_allclose(big, np.cos(3 * xb) + 0.5 * np.sin(5 * xb), atol=1e-12)


def test_energy_budget_closes_for_damped_taylor_green(tmp_path):
    nu, lam = 0.005, 0.5
    s = SpectralNS2D(32, nu=nu, lam=lam, dt_max=0.005)
    s.set_taylor_green()
    log = DiagnosticsLog()
    log.record(s.time, **s.diagnostics())
    while s.time < 1.0 - 1e-14:
        s.step(min(s.cfl_dt(), 1.0 - s.time))
        log.record(s.time, **s.diagnostics())
    b = log.budget(nu=nu)
    assert b.max_abs_residual < 1e-6
    summ = b.summary()
    assert summ["E_vac_final"] > 0 and summ["D_visc_final"] > 0
    assert 0.9 < summ["vac_fraction"] < 1.0           # damping dominates viscosity here
    # export round trip
    p = log.write_csv(tmp_path / "run.csv")
    back = DiagnosticsLog.read_csv(p)
    np.testing.assert_allclose(back.series("E"), log.series("E"), rtol=1e-12)
    j = json.loads(log.write_json(tmp_path / "run.json").read_text())
    assert set(j) >= {"t", "E", "enstrophy", "l4", "vac_power"}


def test_energy_budget_rejects_mismatched_series():
    with pytest.raises(ValueError):
        EnergyBudget([0, 1], [1, 1, 1], [0, 0], [0, 0])


def test_particle_energy_invariant_with_damping():
    """E_kin + E_pot + E_vac is conserved by the symmetric integrator up to
    O(dt^2); E_vac is accumulated exactly from the damping substeps."""
    n = 32
    x = np.linspace(-1.0, 1.0, n)
    def run(dt):
        sys = LagrangianSystem(x[:, None], np.zeros((n, 1)), np.full(n, 1.0 / n), h=0.2,
                               G=1.0, c=0.05, softening=0.4, damping=True, gravity=True)
        e0 = sys.diagnostics(jacobian=False)["E_total"]
        while sys.time < 1.5 - 1e-12:
            sys.step(min(dt, 1.5 - sys.time))
        d = sys.diagnostics()
        return d, abs(d["E_total"] - e0)
    d1, err1 = run(0.02)
    d2, err2 = run(0.01)
    assert d1["E_vac"] > 0.05 * abs(d1["E_pot"])       # damping actually removed energy
    assert err1 < 5e-2 * d1["E_vac"]                   # invariant holds to O(dt^2)
    assert err2 < 0.35 * err1                          # and improves ~4x per dt halving
    assert "J_min" in d1 and np.isfinite(d1["J_min"])


def test_particle_diagnostics_without_gravity():
    sys = LagrangianSystem(np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]]), np.array([2.0]),
                           h=0.5, damping=True, gravity=False)
    sys.step(0.3)
    d = sys.diagnostics(jacobian=False)
    assert d["E_pot"] == 0.0
    assert d["E_kin"] + d["E_vac"] == pytest.approx(1.0, rel=1e-12)   # 1/2 m v0^2 = 1
