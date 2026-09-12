"""Benchmark 3: 2D Taylor-Green vortex, classical NS vs. Moss damping.

Writes 03_cfd_stability.png with
  (a) enstrophy Omega(t),
  (b) L4 velocity norm ||u||_{L4}(t),
  (c) kinetic energy E(t)
for lambda = 0 (classical NS) and lambda > 0 (Moss regularization).

``lam`` is the *dimensionless damping number* lambda_code = lambda_eff U L
of the nondimensionalized equations; it is a free parameter of this
benchmark chosen so that the effect is visible, not a physical value.

For lambda = 0 the 2D Taylor-Green vortex is an exact viscous eigenmode,
omega(t) = 2 sin x sin y exp(-2 nu t), which verifies the solver.  The
energy identity

    E(t) = E0 - 2 nu int_0^t Omega dt' - lam int_0^t ||u||_{L4}^4 dt'

is checked for the damped run with every-step trapezoid quadrature.
Note that 2D Navier-Stokes has no finite-time blow-up even without the
damping term: this benchmark tests the solver and the energy budget,
not a regularity claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..diagnostics import DiagnosticsLog
from ..fluid.spectral import SpectralNS2D
from ._style import CLASSICAL, MOSS, apply_style

__all__ = ["run_cfd_benchmark"]


def _integrate(solver: SpectralNS2D, t_end: float, log: DiagnosticsLog):
    """Every-step histories (t, E, Omega, L4), including t = 0; the full
    diagnostics (norms, damping power, ...) go to ``log``."""
    times, e_hist, o_hist, l4_hist = [0.0], [], [], []
    e, o, l4 = solver.measure()
    e_hist.append(e); o_hist.append(o); l4_hist.append(l4)
    log.record(solver.time, **solver.diagnostics())
    dt_min = np.inf
    while solver.time < t_end - 1e-14:
        dt = min(solver.cfl_dt(), t_end - solver.time)
        dt_min = min(dt_min, dt)
        solver.step(dt)
        e, o, l4 = solver.measure()
        times.append(solver.time)
        e_hist.append(e)
        o_hist.append(o)
        l4_hist.append(l4)
        log.record(solver.time, **solver.diagnostics())
    return (
        np.asarray(times),
        np.asarray(e_hist),
        np.asarray(o_hist),
        np.asarray(l4_hist),
        dt_min,
    )


def run_cfd_benchmark(
    outdir: str | Path = "assets",
    n: int = 64,
    lam: float = 0.5,
    re: float = 200.0,
    t_max: float = 10.0,
    dt_max: float = 0.005,
) -> Dict[str, object]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nu = 1.0 / re
    apply_style()

    solver0 = SpectralNS2D(n, nu=nu, lam=0.0, dt_max=dt_max)
    solver0.set_taylor_green()
    log0 = DiagnosticsLog()
    t0, e0, o0, l40, _ = _integrate(solver0, t_max, log0)

    # exact viscous eigenmode check for the classical run
    e_exact = 0.25 * np.exp(-4.0 * nu * t0)
    o_exact = 0.5 * np.exp(-4.0 * nu * t0)
    err_e = float(np.max(np.abs(e0 - e_exact) / e_exact))
    err_o = float(np.max(np.abs(o0 - o_exact) / o_exact))

    solver1 = SpectralNS2D(n, nu=nu, lam=lam, dt_max=dt_max)
    solver1.set_taylor_green()
    lim = solver1.dt_limits()
    log1 = DiagnosticsLog()
    t1, e1, o1, l41, dt_min = _integrate(solver1, t_max, log1)
    budget = log1.budget(nu=nu).summary()
    csv0 = log0.write_csv(outdir / "03_cfd_classical_timeseries.csv")
    csv1 = log1.write_csv(outdir / "03_cfd_moss_timeseries.csv")

    # energy identity residual for the damped run (every-step trapezoid)
    e_predicted = e1[0] - np.trapezoid(2.0 * nu * o1 + lam * l41**4, t1)
    residual = float(e1[-1] - e_predicted)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, y0, y1, ylab, title in (
        (axes[0], o0, o1, r"$\Omega(t)$", "Enstrophy"),
        (axes[1], l40, l41, r"$\|u\|_{L^4}(t)$", r"$L^4$ velocity norm"),
        (axes[2], e0, e1, r"$E(t)$", "Kinetic energy"),
    ):
        ax.plot(t0, y0, color=CLASSICAL, label=r"classical NS ($\lambda = 0$)")
        ax.plot(t1, y1, color=MOSS, label=rf"moss ($\lambda_\mathrm{{code}} = {lam:g}$)")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend()

    fig.suptitle(
        rf"2D Taylor-Green vortex, $N = {n}$, $Re = {re:g}$: "
        r"$\partial_t u + u\cdot\nabla u = -\nabla p + \nu\nabla^2 u - \lambda_\mathrm{code} |u|^2 u$"
        f"   (energy-identity residual {residual:+.1e})",
        fontsize=10,
    )
    fig.tight_layout()
    path = outdir / "03_cfd_stability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    checks = {
        "taylor_green_exact_decay": err_e < 1e-6 and err_o < 1e-6,
        "energy_identity": abs(residual) < 1e-5,
        "energy_budget_closed": budget["max_abs_residual"] < 1e-5,
        "energy_monotone_classical": bool(np.all(np.diff(e0) <= 0.0)),
        "energy_monotone_moss": bool(np.all(np.diff(e1) <= 0.0)),
    }
    return {
        "n": n,
        "re": re,
        "nu": nu,
        "lam": lam,
        "exact_decay_energy_err": err_e,
        "exact_decay_enstrophy_err": err_o,
        "energy_identity_residual": residual,
        "dt_limits_initial": {k: float(v) for k, v in lim.items()},
        "dt_min_used": float(dt_min),
        "E_classical_final": float(e0[-1]),
        "E_moss_final": float(e1[-1]),
        "energy_budget_moss": budget,
        "timeseries_csv": [str(csv0), str(csv1)],
        "steps_classical": solver0.steps,
        "steps_moss": solver1.steps,
        "params": {"n": n, "lam": lam, "re": re, "nu": nu, "t_max": t_max, "dt_max": dt_max},
        "checks": checks,
        "figure": str(path),
    }
