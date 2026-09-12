"""Benchmark 4b: 3D Taylor-Green sweep over resolution and damping number.

Two questions an engineer or a referee asks about the 3D benchmark:

1. Resolution: at which N does the *classical* run stay resolved over the
   horizon (spectral tail fraction below 1e-3)?  Runs at several N.
2. Damping strength: how do the peak enstrophy, the peak vorticity and
   the energy handed to the reservoir depend on lam_code, in particular
   across the threshold 4 lam_code / Re = 1 above which global regularity
   of the critical damped system is proven (Hajduk & Robinson 2017)?

Writes 06_tg3d_sweep.png and returns every run's metrics for report.json.
Large-N runs record diagnostics every ``diag_every`` steps to keep the
padded-grid norms affordable.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..diagnostics import DiagnosticsLog
from ..fluid.spectral3d import SpectralNS3D
from ._style import CLASSICAL, INK2, MUTED, SERIES, apply_style
from .tg3d import TAIL_LIMIT

__all__ = ["run_tg3d_sweep"]


def _run(n: int, nu: float, lam: float, t_max: float, dt_max: float, diag_every: int) -> Dict[str, object]:
    s = SpectralNS3D(n, nu=nu, lam=lam, dt_max=dt_max)
    s.set_taylor_green()
    log = DiagnosticsLog()
    log.record(s.time, **s.diagnostics())
    t0 = time.perf_counter()
    dt_min = np.inf
    while s.time < t_max - 1e-14:
        dt = min(s.cfl_dt(), t_max - s.time)
        dt_min = min(dt_min, dt)
        s.step(dt)
        if s.steps % diag_every == 0 or s.time >= t_max - 1e-14:
            log.record(s.time, **s.diagnostics())
    wall = time.perf_counter() - t0
    b = log.budget(nu=nu).summary()
    tail = log.series("tail_fraction")
    over = np.where(tail > TAIL_LIMIT)[0]
    enst = log.series("enstrophy")
    i_pk = int(np.argmax(enst))
    return {
        "n": n, "lam": lam, "re": 1.0 / nu, "t_max": t_max, "dt_max": dt_max, "diag_every": diag_every,
        "steps": s.steps, "dt_min": float(dt_min), "wall_s": wall,
        "enstrophy_peak": float(enst[i_pk]), "enstrophy_peak_time": float(log.t[i_pk]),
        "omega_max_peak": float(np.max(log.series("omega_max"))),
        "l4_max": float(np.max(log.series("l4"))),
        "tail_fraction_max": float(np.max(tail)),
        "under_resolved_from_t": float(log.t[over[0]]) if over.size else np.nan,
        "resolved": not bool(over.size),
        "E_final": float(log.series("E")[-1]),
        "E_vac_final": b["E_vac_final"], "vac_fraction": b["vac_fraction"],
        "budget_max_abs_residual": b["max_abs_residual"],
        "threshold_ratio": float(4.0 * lam * nu),          # 4 lam / Re; >= 1: regularity proven
        "_log": log,
    }


def _public(r: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def run_tg3d_sweep(
    outdir: str | Path = "assets",
    re: float = 800.0,
    t_max: float = 6.0,
    dt_max: float = 0.01,
    classical_n_values: Sequence[int] = (32, 64, 96),
    damped_n_values: Sequence[int] = (32, 64),
    lam_values: Sequence[float] = (0.5, 5.0, 50.0),
    lam_values_small_n: Sequence[float] = (200.0,),
    quick: bool = False,
    diag_every_large: int = 5,
    large_n: int = 64,
) -> Dict[str, object]:
    """Resolution sweep of the classical run and lam_code sweep of the damped
    run.  ``lam_values_small_n`` are run only at the smallest damped N (the
    explicit time step shrinks like 1/lam)."""
    if quick:
        classical_n_values, damped_n_values = (8, 16), (8,)
        lam_values, lam_values_small_n, t_max = (0.5, 5.0), (50.0,), 1.0
    nu = 1.0 / re
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    apply_style()
    wall0 = time.perf_counter()

    def dev(n: int) -> int:
        return diag_every_large if n >= large_n else 1

    classical = [_run(n, nu, 0.0, t_max, dt_max, dev(n)) for n in classical_n_values]
    damped: List[Dict[str, object]] = []
    for n in damped_n_values:
        lams = list(lam_values) + (list(lam_values_small_n) if n == min(damped_n_values) else [])
        for lam in lams:
            damped.append(_run(n, nu, lam, t_max, dt_max, dev(n)))

    # -- figure --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2))
    ax = axes[0, 0]
    for i, r in enumerate(classical):
        log = r["_log"]
        ax.plot(log.t, log.series("enstrophy"), color=SERIES[i % len(SERIES)], label=rf"classical, $N = {r['n']}^3$")
        if not np.isnan(r["under_resolved_from_t"]):
            ax.axvline(r["under_resolved_from_t"], color=SERIES[i % len(SERIES)], lw=0.8, ls=":")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\Omega(t)$")
    ax.set_title(r"Classical run: enstrophy vs. resolution (dotted: spectral tail $> 10^{-3}$)")
    ax.legend()

    ax = axes[0, 1]
    n_big = max(damped_n_values)
    for i, r in enumerate([d for d in damped if d["n"] == n_big]):
        log = r["_log"]
        ax.plot(log.t, log.series("enstrophy"), color=SERIES[(i + 1) % len(SERIES)],
                label=rf"$\lambda_\mathrm{{code}} = {r['lam']:g}$")
    ref = next((c for c in classical if c["n"] == n_big), classical[-1])
    ax.plot(ref["_log"].t, ref["_log"].series("enstrophy"), color=CLASSICAL, ls="--", lw=1.0,
            label=rf"classical, $N = {ref['n']}^3$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\Omega(t)$")
    ax.set_title(rf"Damped runs at $N = {n_big}^3$: enstrophy vs. damping number")
    ax.legend()

    ax = axes[1, 0]
    for j, n in enumerate(damped_n_values):
        rs = sorted([d for d in damped if d["n"] == n], key=lambda d: d["lam"])
        ax.plot([r["lam"] for r in rs], [r["enstrophy_peak"] for r in rs], marker="o", ms=4.5,
                color=SERIES[j % len(SERIES)], label=rf"peak $\Omega$, $N = {n}^3$")
        ax.plot([r["lam"] for r in rs], [r["omega_max_peak"] for r in rs], marker="s", ms=4, ls="--",
                color=SERIES[j % len(SERIES)], label=rf"peak $\|\omega\|_\infty$, $N = {n}^3$")
    lam_thr = re / 4.0
    ax.axvline(lam_thr, color=MUTED, lw=0.8, ls=":")
    ax.text(lam_thr, ax.get_ylim()[1] * 0.95, rf"$4\lambda/Re = 1$ ($\lambda = {lam_thr:g}$)",
            rotation=90, va="top", ha="right", fontsize=8, color=INK2)
    for r in classical:
        ax.axhline(r["enstrophy_peak"], color=MUTED, lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda_\mathrm{code}$")
    ax.set_ylabel("peak value over the run")
    ax.set_title("Peak enstrophy and peak vorticity vs. damping number (dashed grey: classical peaks)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for i, r in enumerate(classical):
        log = r["_log"]
        ax.semilogy(log.t, np.maximum(log.series("tail_fraction"), 1e-16), color=SERIES[i % len(SERIES)],
                    label=rf"classical, $N = {r['n']}^3$")
    for i, r in enumerate([d for d in damped if d["n"] == n_big]):
        log = r["_log"]
        ax.semilogy(log.t, np.maximum(log.series("tail_fraction"), 1e-16), color=SERIES[(i + 1) % len(SERIES)],
                    ls="--", label=rf"$\lambda_\mathrm{{code}} = {r['lam']:g}$, $N = {n_big}^3$")
    ax.axhline(TAIL_LIMIT, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"energy fraction at $|k| > 0.8\,k_{\max}$")
    ax.set_title("Spectral tail (under-resolution indicator)")
    ax.legend(fontsize=8)

    fig.suptitle(rf"3D Taylor-Green sweep, $Re = {re:g}$, $t \leq {t_max:g}$", fontsize=11)
    fig.tight_layout()
    path = outdir / "06_tg3d_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    csvs = []
    for r in classical + damped:
        p = outdir / f"06_tg3d_sweep_n{r['n']}_lam{r['lam']:g}_timeseries.csv"
        r["_log"].write_csv(p)
        csvs.append(str(p))

    resolved_n = [c["n"] for c in classical if c["resolved"]]
    checks = {
        "all_budgets_closed": all(r["budget_max_abs_residual"] < 1e-5 for r in classical + damped),
        "all_finite": all(np.isfinite(r["enstrophy_peak"]) for r in classical + damped),
    }
    return {
        "re": re,
        "classical_resolved_n": resolved_n,
        "smallest_resolved_n": min(resolved_n) if resolved_n else None,
        "threshold_lambda_4lam_over_re_eq_1": lam_thr,
        "classical": [_public(r) for r in classical],
        "damped": [_public(r) for r in damped],
        "wall_s": time.perf_counter() - wall0,
        "timeseries_csv": csvs,
        "params": {
            "re": re, "t_max": t_max, "dt_max": dt_max, "classical_n_values": list(classical_n_values),
            "damped_n_values": list(damped_n_values), "lam_values": list(lam_values),
            "lam_values_small_n": list(lam_values_small_n), "quick": quick,
            "diag_every_large": diag_every_large, "large_n": large_n,
        },
        "checks": checks,
        "figure": str(path),
    }
