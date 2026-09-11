"""Benchmark 3: 2D Taylor-Green vortex, classical NS vs. Moss regularization.

Writes assets/03_cfd_stability.png with
  (a) enstrophy Omega(t),
  (b) L4 velocity norm ||u||_{L4}(t),
  (c) kinetic energy E(t)
for lambda = 0 (classical NS) and lambda > 0 (Moss regularization).

For lambda = 0 the 2D Taylor-Green vortex is an exact viscous eigenmode,
omega(t) = 2 sin x sin y exp(-2 nu t), which is used to verify the
solver.  The energy identity

    E(t) = E0 - 2 nu int_0^t Omega dt' - lam int_0^t ||u||_{L4}^4 dt'

is checked for the damped run and its residual is reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..fluid.spectral import SpectralNS2D

__all__ = ["run_cfd_benchmark"]


def _integrate(solver: SpectralNS2D, t_end: float, snap_dt: float):
    times, e_hist, o_hist, l4_hist = [], [], [], []
    t_next = 0.0
    while solver.time < t_end - 1e-14:
        dt = min(solver.cfl_dt(), t_end - solver.time)
        solver.step(dt)
        if solver.time >= t_next - 1e-14:
            e, o, l4 = solver.measure()
            times.append(solver.time)
            e_hist.append(e)
            o_hist.append(o)
            l4_hist.append(l4)
            t_next += snap_dt
    return (
        np.asarray(times),
        np.asarray(e_hist),
        np.asarray(o_hist),
        np.asarray(l4_hist),
    )


def run_cfd_benchmark(
    outdir: str | Path = "assets",
    n: int = 64,
    lam: float = 0.5,
    re: float = 200.0,
    t_max: float = 10.0,
    dt_max: float = 0.005,
) -> Dict[str, float]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nu = 1.0 / re
    snap_dt = max(t_max / 400.0, dt_max)

    solver0 = SpectralNS2D(n, nu=nu, lam=0.0, dt_max=dt_max)
    solver0.set_taylor_green()
    t0, e0, o0, l40 = _integrate(solver0, t_max, snap_dt)

    # exact viscous eigenmode check for the classical run
    e_exact = 0.25 * np.exp(-4.0 * nu * t0)
    o_exact = 0.5 * np.exp(-4.0 * nu * t0)
    err_e = float(np.max(np.abs(e0 - e_exact) / e_exact))
    err_o = float(np.max(np.abs(o0 - o_exact) / o_exact))

    solver1 = SpectralNS2D(n, nu=nu, lam=lam, dt_max=dt_max)
    solver1.set_taylor_green()
    t1, e1, o1, l41 = _integrate(solver1, t_max, snap_dt)

    # energy identity residual for the damped run
    l4_pow4 = l41**4
    e_predicted = 0.25 - np.trapezoid(2.0 * nu * o1 + lam * l4_pow4, t1)
    residual = float(e1[-1] - e_predicted)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, y0, y1, ylab, title in (
        (axes[0], o0, o1, r"$\Omega(t)$", "Enstrophy"),
        (axes[1], l40, l41, r"$\|u\|_{L^4}(t)$", r"$L^4$ velocity norm"),
        (axes[2], e0, e1, r"$E(t)$", "Kinetic energy"),
    ):
        ax.plot(t0, y0, "k-", lw=2, label=r"classical NS ($\lambda = 0$)")
        ax.plot(t1, y1, "r-", lw=2, label=rf"Moss ($\lambda = {lam:g}$)")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        rf"2D Taylor-Green vortex, $N = {n}$, $Re = {re:g}$, "
        r"$\partial_t u + u\cdot\nabla u = -\nabla p + \nu\nabla^2 u - \lambda |u|^2 u$"
    )
    fig.tight_layout()
    path = outdir / "03_cfd_stability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {
        "n": float(n),
        "re": re,
        "nu": nu,
        "lam": lam,
        "exact_decay_energy_err": err_e,
        "exact_decay_enstrophy_err": err_o,
        "energy_identity_residual": residual,
        "E_classical_final": float(e0[-1]),
        "E_moss_final": float(e1[-1]),
    }
