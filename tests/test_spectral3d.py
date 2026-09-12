"""3D spectral solver: exact shear-mode decay, divergence, energy identity,
cubic dealiasing, RK4 order, Taylor-Green run and the benchmark entry."""

from pathlib import Path

import numpy as np
import pytest

from moss_reg.benchmarks.tg3d import run_tg3d_benchmark
from moss_reg.diagnostics import DiagnosticsLog
from moss_reg.fluid.spectral3d import SpectralNS3D


def test_shear_mode_decays_exactly():
    nu = 0.05
    s = SpectralNS3D(16, nu=nu, lam=0.0, dt_max=0.01)
    s.set_shear_mode(amplitude=0.7)
    while s.time < 1.0 - 1e-12:
        s.step(min(s.cfl_dt(), 1.0 - s.time))
    u = s.velocity()
    assert np.abs(u[0] - 0.7 * np.exp(-nu) * np.sin(s.X[1])).max() < 1e-12
    assert np.abs(u[1]).max() < 1e-12 and np.abs(u[2]).max() < 1e-12
    d = s.diagnostics()
    assert d["E"] == pytest.approx(0.25 * 0.49 * np.exp(-2 * nu), rel=1e-10)


def test_taylor_green_initial_diagnostics_and_divergence():
    s = SpectralNS3D(16, nu=1e-3, lam=0.5, dt_max=0.01)
    s.set_taylor_green()
    d = s.diagnostics()
    assert d["E"] == pytest.approx(1.0 / 8.0, rel=1e-12)          # <|u|^2>/2 = 1/8
    assert d["enstrophy"] == pytest.approx(3.0 / 8.0, rel=1e-12)   # <|omega|^2>/2 = 3/8
    assert d["grad_l2_sq"] == pytest.approx(2 * d["enstrophy"], rel=1e-12)
    assert d["div_max"] < 1e-12
    for _ in range(20):
        s.step(s.cfl_dt())
    d = s.diagnostics()
    assert d["div_max"] < 1e-10 and np.isfinite(d["omega_max"])


def test_energy_identity_with_damping():
    nu, lam = 0.01, 0.5
    s = SpectralNS3D(16, nu=nu, lam=lam, dt_max=0.01)
    s.set_taylor_green()
    log = DiagnosticsLog()
    log.record(s.time, **s.diagnostics())
    while s.time < 1.0 - 1e-14:
        s.step(min(s.cfl_dt(), 1.0 - s.time))
        log.record(s.time, **s.diagnostics())
    b = log.budget(nu=nu)
    assert b.max_abs_residual < 1e-6
    assert b.summary()["E_vac_final"] > 0.0


def test_cubic_term_dealiased_in_3d():
    """u = cos(5 y) x-hat on n = 16 (k_max = 5): |u|^2 u contains cos(15 y),
    which aliases to |k| = 1 on the native grid but not on the padded grid."""
    n, k = 16, 5
    for cubic_dealias, spurious in ((False, 0.125), (True, 0.0)):
        s = SpectralNS3D(n, nu=0.0, lam=1.0, cubic_dealias=cubic_dealias)
        u0 = np.cos(k * s.X[1])
        s.set_field([u0, np.zeros_like(u0), np.zeros_like(u0)])
        rhs = s._rhs(s.u_hat)
        a = rhs[0] / n**3
        assert abs(a[0, k, 0]) == pytest.approx(3.0 / 8.0, rel=1e-10)
        assert abs(a[0, 1, 0]) == pytest.approx(spurious, abs=1e-10)


def test_rk4_fourth_order():
    def energy_at(dt, T=0.2):
        s = SpectralNS3D(16, nu=0.01, lam=0.5, dt_max=dt)
        s.set_taylor_green()
        while s.time < T - 1e-12:
            s.step(min(dt, T - s.time))
        return s.diagnostics()["E"]
    ref = energy_at(0.0005)
    errs = [abs(energy_at(dt) - ref) for dt in (0.02, 0.01, 0.005)]
    ratios = [errs[i] / errs[i + 1] for i in range(2)]
    assert all(10.0 <= r <= 22.0 for r in ratios), ratios


def test_tg3d_benchmark_smoke(tmp_path):
    m = run_tg3d_benchmark(tmp_path, n=8, lam=0.5, re=100.0, t_max=0.3, dt_max=0.01)
    assert (tmp_path / "05_tg3d.png").is_file()
    assert all(m["checks"].values()), m["checks"]
    assert set(m["params"]) == {"n", "lam", "re", "nu", "t_max", "dt_max"}
    assert m["E_final_moss"] < m["E_final_classical"]
    for csv in m["timeseries_csv"]:
        assert (tmp_path / csv.split("/")[-1]).is_file()


def test_tg3d_sweep_smoke(tmp_path):
    from moss_reg.benchmarks.tg3d_sweep import run_tg3d_sweep

    m = run_tg3d_sweep(tmp_path, quick=True, re=100.0, dt_max=0.02)
    assert (tmp_path / "06_tg3d_sweep.png").is_file()
    assert all(m["checks"].values()), m["checks"]
    assert len(m["classical"]) == 2 and len(m["damped"]) == 3
    assert m["threshold_lambda_4lam_over_re_eq_1"] == pytest.approx(25.0)
    assert all((tmp_path / Path(c).name).is_file() for c in m["timeseries_csv"])
