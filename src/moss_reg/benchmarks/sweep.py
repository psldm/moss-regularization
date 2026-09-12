"""Benchmark 2b: shell-crossing parameter sweep.

Answers, for the 1D cold collapse of :mod:`.particles`, *when* the
Lagrangian map crosses (min_i J_i <= 0) as a function of

  * c (speed of light in code units)  x  N (particle number), and
  * dt_max, h, softening around a baseline (N, c)  (sensitivity),

with the undamped run as reference at every point.  Code units are
G = M = L = 1, so compactness G M / (L c^2) = 1 / c^2 and the undamped
free-fall speed v_ff ~ 1.7 sets the boundary c = v_ff between the
physical regime (v_ff < c) and the superluminal-in-code-units regime.

Writes 02_shell_crossing_sweep.png (four panels) and returns a metrics
dictionary with every run, used by ``moss-reg sweep`` for report.json.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._style import CLASSICAL, INK2, MUTED, SERIES, apply_style
from .particles import _build, _run

__all__ = ["run_shell_crossing_sweep"]

FULL = dict(
    c_values=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
    n_values=(64, 128, 256),
    baseline_n=128,
    baseline_c=0.01,
    dt_values=(0.02, 0.01, 0.005, 0.0025),
    h_values=(0.06, 0.09, 0.12, 0.18, 0.24),
    softening_values=(0.2, 0.3, 0.4, 0.6, 0.8),
    t_max=8.0,
)
QUICK = dict(
    c_values=(0.005, 0.02, 0.1, 1.0, 5.0),
    n_values=(32, 64),
    baseline_n=64,
    baseline_c=0.01,
    dt_values=(0.01, 0.005),
    h_values=(0.06, 0.12, 0.24),
    softening_values=(0.2, 0.4, 0.8),
    t_max=4.0,
)


def _single(n: int, c: Optional[float], h: float, softening: float,
            t_max: float, dt_max: float) -> Dict[str, object]:
    t0 = time.perf_counter()
    sys, tracker = _build(n, c, softening, h)
    res = _run(sys, tracker, t_max, dt_max, stop_on_crossing=True)
    cc = c if c is not None else np.nan
    return {
        "n": n, "c": cc, "h": h, "softening": softening, "dt_max": dt_max, "t_max": t_max,
        "damped": c is not None,
        "t_cross": res["crossing"] if res["crossing"] is not None else np.nan,
        "crossed": res["crossing"] is not None,
        "min_j": float(np.min(res["jmin"])),
        "v_peak": res["vpeak"],
        "v_peak_over_c": res["vpeak"] / cc if c is not None else np.nan,
        "steps": res["steps"],
        "wall_s": time.perf_counter() - t0,
        "_times": res["times"],
        "_jmin": res["jmin"],
    }


def _tx(run: Dict[str, object], t_max: float) -> float:
    return t_max if not run["crossed"] else float(run["t_cross"])


def run_shell_crossing_sweep(
    outdir: str | Path = "assets",
    quick: bool = False,
    c_values: Optional[Sequence[float]] = None,
    n_values: Optional[Sequence[int]] = None,
    t_max: Optional[float] = None,
    dt_max: float = 0.005,
    h: float = 0.12,
    softening: float = 0.4,
    baseline_n: Optional[int] = None,
    baseline_c: Optional[float] = None,
    dt_values: Optional[Sequence[float]] = None,
    h_values: Optional[Sequence[float]] = None,
    softening_values: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    grid = QUICK if quick else FULL
    c_values = tuple(grid["c_values"] if c_values is None else c_values)
    n_values = tuple(grid["n_values"] if n_values is None else n_values)
    t_max = float(grid["t_max"] if t_max is None else t_max)
    baseline_n = int(grid["baseline_n"] if baseline_n is None else baseline_n)
    baseline_c = float(grid["baseline_c"] if baseline_c is None else baseline_c)
    dt_values = tuple(grid["dt_values"] if dt_values is None else dt_values)
    h_values = tuple(grid["h_values"] if h_values is None else h_values)
    softening_values = tuple(grid["softening_values"] if softening_values is None else softening_values)
    if baseline_c not in c_values:
        c_values = tuple(sorted(set(c_values) | {baseline_c}))
    if baseline_n not in n_values:
        n_values = tuple(sorted(set(n_values) | {baseline_n}))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    apply_style()
    wall0 = time.perf_counter()

    # -- c x N grid ----------------------------------------------------------
    undamped: Dict[int, Dict[str, object]] = {
        n: _single(n, None, h, softening, t_max, dt_max) for n in n_values
    }
    grid_runs: List[Dict[str, object]] = []
    for n in n_values:
        for c in c_values:
            grid_runs.append(_single(n, c, h, softening, t_max, dt_max))
    v_ff = float(np.mean([r["v_peak"] for r in undamped.values()]))

    # -- sensitivity around the baseline -------------------------------------
    sens: Dict[str, List[Dict[str, object]]] = {"dt_max": [], "h": [], "softening": []}
    sens_ref: Dict[str, List[Dict[str, object]]] = {"dt_max": [], "h": [], "softening": []}
    for dtv in dt_values:
        sens["dt_max"].append(_single(baseline_n, baseline_c, h, softening, t_max, dtv))
        sens_ref["dt_max"].append(_single(baseline_n, None, h, softening, t_max, dtv))
    for hv in h_values:
        sens["h"].append(_single(baseline_n, baseline_c, hv, softening, t_max, dt_max))
        sens_ref["h"].append(_single(baseline_n, None, hv, softening, t_max, dt_max))
    for sv in softening_values:
        sens["softening"].append(_single(baseline_n, baseline_c, h, sv, t_max, dt_max))
        sens_ref["softening"].append(_single(baseline_n, None, h, sv, t_max, dt_max))
    baseline_values = {"dt_max": dt_max, "h": h, "softening": softening}

    # -- figure --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2))
    colors = {n: SERIES[i % len(SERIES)] for i, n in enumerate(n_values)}

    ax = axes[0, 0]
    ax.axvspan(min(c_values) / 2.0, v_ff, color=MUTED, alpha=0.10, lw=0)
    ax.text(np.sqrt(min(c_values) / 2.0 * v_ff), t_max * 0.02,
            r"$v_\mathrm{ff} > c$: superluminal in code units", ha="center",
            va="bottom", color=INK2, fontsize=8.5)
    for n in n_values:
        runs = [r for r in grid_runs if r["n"] == n]
        cs = np.array([r["c"] for r in runs])
        tx = np.array([_tx(r, t_max) for r in runs])
        crossed = np.array([r["crossed"] for r in runs])
        ax.plot(cs, tx, color=colors[n], marker="o", ms=4.5, label=f"moss, N = {n}")
        if np.any(~crossed):
            ax.plot(cs[~crossed], tx[~crossed], ls="none", marker="o", ms=9,
                    mfc="none", mec=colors[n], mew=1.4)
        ax.axhline(_tx(undamped[n], t_max), color=colors[n], lw=1.0, ls="--")
    ax.plot([], [], color=MUTED, ls="--", lw=1.0, label="classical (dashed, per N)")
    ax.plot([], [], ls="none", marker="o", ms=9, mfc="none", mec=MUTED, mew=1.4,
            label=f"no crossing by t = {t_max:g}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$c$ (code units, $v_\mathrm{unit} = \sqrt{GM/L}$)")
    ax.set_ylabel(r"crossing time $t_\times$  ($\min_i J_i \leq 0$)")
    ax.set_title("Crossing time vs. speed of light and resolution")
    def _c_to_compactness(c):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.asarray(c) > 0, 1.0 / np.asarray(c) ** 2, np.inf)

    def _compactness_to_c(k):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.asarray(k) > 0, 1.0 / np.sqrt(np.asarray(k)), np.inf)

    sec = ax.secondary_xaxis("top", functions=(_c_to_compactness, _compactness_to_c))
    sec.set_xlabel(r"compactness $GM / (L c^2)$", fontsize=9)
    ax.set_ylim(0.0, t_max * 1.32)
    ax.legend(loc="upper right", ncol=2)

    ax = axes[0, 1]
    for n in n_values:
        runs = [r for r in grid_runs if r["n"] == n]
        ax.plot([r["c"] for r in runs], [r["v_peak_over_c"] for r in runs],
                color=colors[n], marker="o", ms=4.5, label=f"N = {n}")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$c$ (code units)")
    ax.set_ylabel(r"peak $|v| / c$ (moss run)")
    ax.set_title(r"Cubic damping caps the speed near $c$ (for $c < v_\mathrm{ff}$)")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    for i, key in enumerate(("dt_max", "h", "softening")):
        col = SERIES[i]
        xs = np.array([r[key] for r in sens[key]]) / baseline_values[key]
        ax.plot(xs, [_tx(r, t_max) for r in sens[key]], color=col, marker="o", ms=4.5,
                label=f"moss: {key}")
        ax.plot(xs, [_tx(r, t_max) for r in sens_ref[key]], color=col, lw=1.0, ls="--",
                marker="s", ms=3.5, label=f"classical: {key}")
    ax.set_xscale("log")
    ax.set_xlabel("parameter / baseline value")
    ax.set_ylabel(r"crossing time $t_\times$")
    ax.set_title(
        f"Sensitivity at N = {baseline_n}, c = {baseline_c:g} "
        f"(baseline dt = {dt_max:g}, h = {h:g}, softening = {softening:g})"
    )
    ax.set_ylim(0.0, t_max * 1.25)
    ax.legend(loc="upper center", ncol=3)

    ax = axes[1, 1]
    for n in n_values:
        run = next(r for r in grid_runs if r["n"] == n and r["c"] == baseline_c)
        ax.plot(run["_times"], run["_jmin"], color=colors[n], label=f"moss, N = {n}")
    ref = undamped[baseline_n]
    ax.plot(ref["_times"], ref["_jmin"], color=CLASSICAL, lw=1.0, ls="--",
            label=f"classical, N = {baseline_n}")
    ax.axhline(0.0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\min_i J_i(t)$")
    ax.set_title(rf"Jacobian histories at $c = {baseline_c:g}$: crossing moves earlier with N")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(loc="lower left")

    fig.suptitle(
        "1D cold collapse shell-crossing sweep  "
        rf"($G = M = L = 1$, $\lambda_i = G\rho_i / c^3$, $v_\mathrm{{ff}} \approx {v_ff:.2f}$)",
        fontsize=11,
    )
    fig.tight_layout()
    path = outdir / "02_shell_crossing_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # -- summary -------------------------------------------------------------
    def public(r: Dict[str, object]) -> Dict[str, object]:
        return {k: v for k, v in r.items() if not k.startswith("_")}

    small_c = [r for r in grid_runs if r["c"] <= 0.1]
    speed_cap = float(np.median([r["v_peak_over_c"] for r in small_c])) if small_c else np.nan
    no_cross = [(int(r["n"]), float(r["c"])) for r in grid_runs if not r["crossed"]]
    base = sorted((r for r in grid_runs if r["c"] == baseline_c), key=lambda r: r["n"])
    base_tx = [_tx(r, t_max) for r in base]
    monotone = bool(all(base_tx[i] >= base_tx[i + 1] for i in range(len(base_tx) - 1)))
    phys = [r for r in grid_runs if r["c"] >= v_ff]
    phys_delay = (
        float(max(_tx(r, t_max) - _tx(undamped[r["n"]], t_max) for r in phys)) if phys else np.nan
    )
    checks = {
        "all_runs_finite": bool(all(np.isfinite(r["min_j"]) for r in grid_runs)),
        "classical_reference_crosses": bool(all(r["crossed"] for r in undamped.values())),
        "speed_cap_near_c": bool(0.5 < speed_cap < 3.0) if small_c else True,
    }
    conclusion = (
        f"damped runs are speed-limited at |v| ~ {speed_cap:.2f} c; crossing is delayed by "
        f"~ (particle spacing)/c and moves earlier as N grows "
        f"({'monotone' if monotone else 'not monotone'} at c = {baseline_c:g}); "
        f"in the physical regime c >= v_ff the maximal delay is {phys_delay:.3g} time units "
        f"({len(no_cross)} of {len(grid_runs)} runs show no crossing within t = {t_max:g}, all at c < v_ff)"
        if all(cc < v_ff for _, cc in no_cross) else
        f"damped runs are speed-limited at |v| ~ {speed_cap:.2f} c; "
        f"{len(no_cross)} runs show no crossing within t = {t_max:g}: {no_cross}"
    )
    return {
        "v_ff": v_ff,
        "speed_cap_ratio_median": speed_cap,
        "no_crossing_within_horizon": no_cross,
        "crossing_time_monotone_in_n_at_baseline": monotone,
        "baseline_crossing_times_by_n": dict(zip([int(r["n"]) for r in base], base_tx)),
        "physical_regime_max_delay": phys_delay,
        "conclusion": conclusion,
        "wall_s": time.perf_counter() - wall0,
        "runs": [public(r) for r in grid_runs],
        "classical_reference": {int(n): public(r) for n, r in undamped.items()},
        "sensitivity": {k: [public(r) for r in v] for k, v in sens.items()},
        "sensitivity_classical": {k: [public(r) for r in v] for k, v in sens_ref.items()},
        "params": {
            "quick": quick, "c_values": list(c_values), "n_values": list(n_values),
            "t_max": t_max, "dt_max": dt_max, "h": h, "softening": softening,
            "baseline_n": baseline_n, "baseline_c": baseline_c,
            "dt_values": list(dt_values), "h_values": list(h_values),
            "softening_values": list(softening_values), "G": 1.0, "M": 1.0, "L": 1.0,
        },
        "checks": checks,
        "figure": str(path),
    }
