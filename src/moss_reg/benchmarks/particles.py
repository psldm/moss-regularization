"""Benchmark 2: 1D cold collapse, classical vs. vacuum damping (one comparison).

Code units: G = M = L = 1 (total mass M, initial half-length L), so the
velocity unit is v_unit = sqrt(G M / L) and the undamped free-fall speed
is v_ff ~ 1.7 v_unit.  The single physical dial of the problem is the
compactness

    G M / (L c^2) = 1 / c_code^2 ,

where c_code = c / v_unit is the speed of light in code units.  Physical
bodies have compactness <= 0.5 (black-hole limit), i.e. c_code >= 1.4 and
v_ff < c.  The default (compactness = 0.1, c_code ~ 3.2) is in that
regime.  c_code = 0.005 (compactness 4e4) reproduces the strong-coupling
figure of release 0.1.0, in which the flow is superluminal in the model's
own units and the cubic damping acts as a speed limiter near c.

Writes 02_shell_crossing_arrest.png with
  (a) min_i J_i(t) for the undamped and damped runs,
  (b) velocity profiles at the undamped crossing time and at the end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..particles.integrator import LagrangianSystem
from ..particles.jacobian import JacobianTracker
from ._style import CLASSICAL, INK2, MOSS, MUTED, apply_style

__all__ = ["run_shell_crossing_benchmark", "free_fall_reference"]


def _build(
    n: int, c: Optional[float], softening: float, h: float = 0.12
) -> Tuple[LagrangianSystem, JacobianTracker]:
    x = np.linspace(-1.0, 1.0, n)
    m = np.full(n, 1.0 / n)
    sys = LagrangianSystem(
        positions=x[:, None],
        velocities=np.zeros((n, 1)),
        masses=m,
        h=h,
        G=1.0,
        c=1.0 if c is None else c,
        softening=softening,
        damping=c is not None,
        gravity=True,
    )
    return sys, JacobianTracker(sys.initial_positions)


def _run(
    sys: LagrangianSystem,
    tracker: JacobianTracker,
    t_end: float,
    dt_max: float,
    stop_on_crossing: bool,
    snapshots: Sequence[float] = (),
) -> Dict[str, object]:
    times, jmin = [0.0], [1.0]
    crossing: Optional[float] = None
    vpeak = 0.0
    snaps: List[Tuple[float, np.ndarray, np.ndarray]] = []
    pending = sorted(t for t in snapshots if 0.0 < t <= t_end)
    while sys.time < t_end - 1e-14:
        dt = min(sys.adaptive_dt(dt_max=dt_max, damp_safety=np.inf), t_end - sys.time)
        if pending:
            dt = min(dt, pending[0] - sys.time)
        sys.step(dt)
        times.append(sys.time)
        j = tracker.min_jacobian(sys.positions)
        jmin.append(j)
        vpeak = max(vpeak, float(np.max(np.abs(sys.velocities))))
        if pending and abs(sys.time - pending[0]) < 1e-12:
            snaps.append((sys.time, sys.positions[:, 0].copy(), sys.velocities[:, 0].copy()))
            pending.pop(0)
        if crossing is None and j <= 0.0:
            crossing = sys.time
            if stop_on_crossing:
                break
    return {
        "times": np.asarray(times),
        "jmin": np.asarray(jmin),
        "crossing": crossing,
        "vpeak": vpeak,
        "vfinal": float(np.max(np.abs(sys.velocities))),
        "snapshots": snaps,
        "final": (sys.time, sys.positions[:, 0].copy(), sys.velocities[:, 0].copy()),
        "steps": sys.n_steps,
    }


def free_fall_reference(
    n: int = 200, softening: float = 0.4, h: float = 0.12, t_max: float = 6.0, dt_max: float = 0.005
) -> Dict[str, float]:
    """Undamped crossing time and peak speed v_ff (code units)."""
    sys, tr = _build(n, None, softening, h)
    res = _run(sys, tr, t_max, dt_max, stop_on_crossing=True)
    return {"crossing_time": res["crossing"], "v_ff": res["vpeak"]}


def run_shell_crossing_benchmark(
    outdir: str | Path = "assets",
    n: int = 200,
    compactness: Optional[float] = None,
    c: Optional[float] = None,
    t_max: float = 6.0,
    dt_max: float = 0.005,
    softening: float = 0.4,
    h: float = 0.12,
) -> Dict[str, object]:
    """Classical vs. moss 1D collapse.  ``c`` overrides ``compactness``
    (default compactness 0.1, i.e. c = 1/sqrt(0.1) ~ 3.16 code units)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if c is None:
        compactness = 0.1 if compactness is None else float(compactness)
        c = 1.0 / np.sqrt(compactness)
    c = float(c)
    compactness = 1.0 / c**2
    apply_style()

    sys0, tr0 = _build(n, None, softening, h)
    r0 = _run(sys0, tr0, t_max, dt_max, stop_on_crossing=True)
    t_x0 = r0["crossing"]
    v_ff = r0["vpeak"]

    snap_times = [t_x0] if t_x0 is not None else []
    sys1, tr1 = _build(n, c, softening, h)
    r1 = _run(sys1, tr1, t_max, dt_max, stop_on_crossing=False, snapshots=snap_times)
    t_x1 = r1["crossing"]
    regime = "physical (v_ff < c)" if v_ff < c else "superluminal in code units (v_ff > c)"

    # -- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.plot(r0["times"], r0["jmin"], color=CLASSICAL, label="classical (no damping)")
    ax.plot(r1["times"], r1["jmin"], color=MOSS, label=r"moss ($\lambda_i = G\rho_i/c^3$)")
    ax.axhline(0.0, color=MUTED, lw=0.8, ls="--")
    for t_x, col, dy in ((t_x0, CLASSICAL, 0.22), (t_x1, MOSS, 0.08)):
        if t_x is not None:
            ax.axvline(t_x, color=col, lw=0.8, ls=":")
            ax.annotate(f"crossing t = {t_x:.2f}", (t_x, 0.0), xytext=(t_x + 0.15, -0.55 - dy),
                        color=col, fontsize=8.5)
    if t_x1 is None:
        ax.text(0.98, 0.9, f"moss: no crossing by t = {t_max:g}", transform=ax.transAxes,
                ha="right", color=MOSS, fontsize=8.5)
    ax.set_xlabel(r"$t$ (units of $\sqrt{L^3/GM}$)")
    ax.set_ylabel(r"$\min_i J_i(t)$")
    ax.set_title("Deformation Jacobian of the Lagrangian map")
    ax.set_ylim(-1.2, 1.2)
    ax.legend(loc="lower left")

    ax = axes[1]
    if r1["snapshots"]:
        t_s, x_s, v_s = r1["snapshots"][0]
        ax.plot(x_s, v_s / c, color=MOSS, lw=1.4, label=f"moss, t = {t_s:.2f}")
    t_f, x_f, v_f = r1["final"]
    ax.plot(x_f, v_f / c, color=MOSS, lw=1.0, ls="--", label=f"moss, t = {t_f:.2f}")
    t_0, x_0, v_0 = r0["final"]
    ax.plot(x_0, v_0 / c, color=CLASSICAL, lw=1.4, ls=":", label=f"classical, t = {t_0:.2f}")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
    ax.axhline(-1.0, color=MUTED, lw=0.8, ls="--")
    ax.text(0.02, 0.97, f"peak |v|/c: classical {v_ff / c:.2g}, moss {r1['vpeak'] / c:.2g}",
            transform=ax.transAxes, va="top", color=INK2, fontsize=8.5)
    vlim = max(1.0, v_ff / c, r1["vpeak"] / c) * 1.3
    ax.set_ylim(-vlim, vlim)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v / c$")
    ax.set_title("Velocity profiles (dashed: $|v| = c$)")
    ax.legend(loc="lower right")

    fig.suptitle(
        f"1D cold collapse, N = {n}, softening = {softening:g}, h = {h:g};  "
        rf"$c = {c:.3g}$ code units, compactness $GM/Lc^2 = {compactness:.3g}$, "
        rf"$v_\mathrm{{ff}}/c = {v_ff / c:.2g}$  [{regime}]",
        fontsize=10,
    )
    fig.tight_layout()
    path = outdir / "02_shell_crossing_arrest.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    finite = bool(np.all(np.isfinite(r0["jmin"])) and np.all(np.isfinite(r1["jmin"])))
    checks = {
        "undamped_crossing_detected": t_x0 is not None,
        "jacobian_histories_finite": finite,
        "damped_speed_bounded": r1["vpeak"] <= max(v_ff, 2.0 * c) * 1.05,
    }
    return {
        "n": n,
        "c": c,
        "compactness": compactness,
        "regime": regime,
        "v_ff": v_ff,
        "v_ff_over_c": v_ff / c,
        "crossing_time_classical": t_x0 if t_x0 is not None else np.nan,
        "crossing_time_moss": t_x1 if t_x1 is not None else np.nan,
        "crossing_delay": (t_x1 - t_x0) if (t_x0 is not None and t_x1 is not None) else np.nan,
        "min_jacobian_moss": float(np.min(r1["jmin"])),
        "vpeak_moss": r1["vpeak"],
        "vpeak_moss_over_c": r1["vpeak"] / c,
        "vfinal_moss": r1["vfinal"],
        "steps_classical": r0["steps"],
        "steps_moss": r1["steps"],
        "params": {
            "n": n, "c": c, "compactness": compactness, "t_max": t_max,
            "dt_max": dt_max, "softening": softening, "h": h, "G": 1.0, "M": 1.0, "L": 1.0,
        },
        "checks": checks,
        "figure": str(path),
    }
