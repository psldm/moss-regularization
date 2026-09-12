"""Benchmark 4: 3D Taylor-Green vortex, classical vs. damped, with a
resolution indicator.

Writes 05_tg3d.png (energy, enstrophy, max vorticity, L4 norm) and the
per-step diagnostics of both runs as CSV.  The damping number ``lam`` is
a tuned code-unit parameter (see README); the point of the benchmark is
to show the solver's energy budget and the behaviour of the norms, and
to make the under-resolution of the classical run explicit through
k_max * eta rather than to claim anything about regularity.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..diagnostics import DiagnosticsLog
from ..fluid.spectral3d import SpectralNS3D
from ._style import CLASSICAL, MOSS, MUTED, apply_style

__all__ = ["run_tg3d_benchmark"]

TAIL_LIMIT = 1e-3       # energy fraction at |k| > 0.8 kmax above which a run is called under-resolved


def _run(n: int, nu: float, lam: float, t_max: float, dt_max: float) -> DiagnosticsLog:
    s = SpectralNS3D(n, nu=nu, lam=lam, dt_max=dt_max)
    s.set_taylor_green()
    log = DiagnosticsLog()
    log.record(s.time, **s.diagnostics())
    t0 = time.perf_counter()
    while s.time < t_max - 1e-14:
        s.step(min(s.cfl_dt(), t_max - s.time))
        log.record(s.time, **s.diagnostics())
    log.wall_s = time.perf_counter() - t0        # type: ignore[attr-defined]
    return log


def run_tg3d_benchmark(
    outdir: str | Path = "assets",
    n: int = 32,
    lam: float = 0.5,
    re: float = 800.0,
    t_max: float = 6.0,
    dt_max: float = 0.01,
) -> Dict[str, object]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nu = 1.0 / re
    apply_style()

    log0 = _run(n, nu, 0.0, t_max, dt_max)
    log1 = _run(n, nu, lam, t_max, dt_max)
    b0 = log0.budget(nu=nu).summary()
    b1 = log1.budget(nu=nu).summary()
    csv0 = log0.write_csv(outdir / "05_tg3d_classical_timeseries.csv")
    csv1 = log1.write_csv(outdir / "05_tg3d_moss_timeseries.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    panels = (
        (axes[0, 0], "E", r"$E(t)$", "Kinetic energy"),
        (axes[0, 1], "enstrophy", r"$\Omega(t)$", "Enstrophy"),
        (axes[1, 0], "omega_max", r"$\|\omega\|_{L^\infty}(t)$", "Maximum vorticity"),
        (axes[1, 1], "l4", r"$\|u\|_{L^4}(t)$", r"$L^4$ velocity norm"),
    )
    for ax, key, ylab, title in panels:
        ax.plot(log0.t, log0.series(key), color=CLASSICAL, label=r"classical NS ($\lambda = 0$)")
        ax.plot(log1.t, log1.series(key), color=MOSS, label=rf"damped ($\lambda_\mathrm{{code}} = {lam:g}$)")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend()
    # resolution indicator on the enstrophy panel: energy piling up at the
    # grid scale (tail fraction of the spectrum above 1e-3)
    ax = axes[0, 1]
    tail0 = log0.series("tail_fraction")
    under = np.where(tail0 > TAIL_LIMIT)[0]
    t_under = float(log0.t[under[0]]) if under.size else None
    if t_under is not None:
        ax.axvline(t_under, color=MUTED, lw=0.8, ls=":")
        ax.text(t_under, ax.get_ylim()[1] * 0.95,
                rf"classical: spectral tail $> 10^{{-3}}$ from $t = {t_under:.2f}$",
                fontsize=8, color=MUTED, ha="left", va="top")
    fig.suptitle(
        rf"3D Taylor-Green vortex, $N = {n}^3$, $Re = {re:g}$: classical vs. cubic damping  "
        rf"(energy-budget residual {b0['max_abs_residual']:.1e} / {b1['max_abs_residual']:.1e})",
        fontsize=10,
    )
    fig.tight_layout()
    path = outdir / "05_tg3d.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    i0 = int(np.argmax(log0.series("enstrophy")))
    i1 = int(np.argmax(log1.series("enstrophy")))
    checks = {
        "energy_budget_classical": b0["max_abs_residual"] < 1e-5,
        "energy_budget_moss": b1["max_abs_residual"] < 1e-5,
        "divergence_free": max(float(np.max(log0.series("div_max"))), float(np.max(log1.series("div_max")))) < 1e-8,
        "all_finite": bool(np.all(np.isfinite(log0.series("enstrophy"))) and np.all(np.isfinite(log1.series("enstrophy")))),
    }
    return {
        "n": n, "re": re, "nu": nu, "lam": lam,
        "enstrophy_peak_classical": float(log0.series("enstrophy")[i0]),
        "enstrophy_peak_time_classical": float(log0.t[i0]),
        "enstrophy_peak_moss": float(log1.series("enstrophy")[i1]),
        "enstrophy_peak_time_moss": float(log1.t[i1]),
        "omega_max_peak_classical": float(np.max(log0.series("omega_max"))),
        "omega_max_peak_moss": float(np.max(log1.series("omega_max"))),
        "kmax_eta_min_classical": float(np.min(log0.series("kmax_eta"))),
        "kmax_eta_min_moss": float(np.min(log1.series("kmax_eta"))),
        "tail_fraction_max_classical": float(np.max(tail0)),
        "tail_fraction_max_moss": float(np.max(log1.series("tail_fraction"))),
        "classical_under_resolved_from_t": t_under if t_under is not None else np.nan,
        "classical_under_resolved": t_under is not None,
        "moss_under_resolved": bool(np.max(log1.series("tail_fraction")) > TAIL_LIMIT),
        "E_final_classical": float(log0.series("E")[-1]),
        "E_final_moss": float(log1.series("E")[-1]),
        "energy_budget_classical": b0,
        "energy_budget_moss": b1,
        "steps_classical": int(log0.series("steps")[-1]),
        "steps_moss": int(log1.series("steps")[-1]),
        "wall_classical_s": float(getattr(log0, "wall_s", np.nan)),
        "wall_moss_s": float(getattr(log1, "wall_s", np.nan)),
        "timeseries_csv": [str(csv0), str(csv1)],
        "params": {"n": n, "lam": lam, "re": re, "nu": nu, "t_max": t_max, "dt_max": dt_max},
        "checks": checks,
        "figure": str(path),
    }
