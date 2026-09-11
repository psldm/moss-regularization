"""Benchmark 2: 1D cold collapse, shell crossing vs. vacuum damping.

Writes assets/02_shell_crossing_arrest.png with
  (a) min_i J_i(t) for the undamped (crossing) and damped (arrested) runs,
  (b) velocity profiles showing the bounded deceleration.

Code units: G = 1, damping coupling lambda_i = G rho_i / c^3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..particles.integrator import LagrangianSystem
from ..particles.jacobian import JacobianTracker

__all__ = ["run_shell_crossing_benchmark"]


def _build(
    n: int, c: Optional[float], softening: float
) -> Tuple[LagrangianSystem, JacobianTracker]:
    x = np.linspace(-1.0, 1.0, n)
    m = np.full(n, 1.0 / n)
    sys = LagrangianSystem(
        positions=x[:, None],
        velocities=np.zeros((n, 1)),
        masses=m,
        h=0.12,
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
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    times, jmin = [0.0], [1.0]
    crossing = None
    while sys.time < t_end - 1e-14:
        dt = min(sys.adaptive_dt(dt_max=dt_max, damp_safety=np.inf), t_end - sys.time)
        sys.step(dt)
        times.append(sys.time)
        jmin.append(tracker.min_jacobian(sys.positions))
        if stop_on_crossing and jmin[-1] <= 0.0:
            crossing = sys.time
            break
    return np.array(times), np.array(jmin), crossing


def run_shell_crossing_benchmark(
    outdir: str | Path = "assets",
    n: int = 200,
    c: float = 0.005,
    t_max: float = 6.0,
    dt_max: float = 0.005,
    softening: float = 0.4,
) -> Dict[str, float]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sys0, tr0 = _build(n, None, softening)
    t0, j0, crossing = _run(sys0, tr0, t_max, dt_max, stop_on_crossing=True)
    vmax0 = float(np.max(np.abs(sys0.velocities)))

    sys1, tr1 = _build(n, c, softening)
    t1, j1, _ = _run(sys1, tr1, t_max, dt_max, stop_on_crossing=False)
    vmax1 = float(np.max(np.abs(sys1.velocities)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(t0, j0, "r-", lw=2, label="no damping")
    ax.plot(t1, j1, "b-", lw=2, label=r"vacuum damping ($\lambda_i = G\rho_i/c^3$)")
    ax.axhline(0.0, color="k", ls="--", lw=1)
    if crossing is not None:
        ax.axvline(crossing, color="r", ls=":", lw=1)
        ax.annotate(
            "shell crossing", (crossing, 0.0), xytext=(0.55, 0.32),
            textcoords="axes fraction", color="r",
            arrowprops=dict(arrowstyle="->", color="r"),
        )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\min_i J_i(t)$")
    ax.set_title(r"Deformation Jacobian (Theorem D.1)")
    ax.set_ylim(-1.2, 1.2)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    snap_times = [1.0, 2.0, 4.0]
    sys_s, tr_s = _build(n, c, softening)
    snap_idx = 0
    while sys_s.time < t_max - 1e-14 and snap_idx < len(snap_times):
        dt = min(
            sys_s.adaptive_dt(dt_max=dt_max, damp_safety=np.inf),
            snap_times[snap_idx] - sys_s.time,
        )
        sys_s.step(dt)
        if abs(sys_s.time - snap_times[snap_idx]) < 1e-12:
            ax.plot(
                sys_s.positions[:, 0], sys_s.velocities[:, 0], "o-",
                ms=3, lw=1, label=f"damped, t={snap_times[snap_idx]:g}",
            )
            snap_idx += 1
    ax.plot(
        sys0.positions[:, 0], sys0.velocities[:, 0], "r--",
        lw=1.5, label=f"undamped, t={sys0.time:.2f}",
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$")
    ax.set_title("Velocity profiles: bounded deceleration")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"1D cold collapse, N = {n}, softening = {softening}; "
        rf"$\lambda_i = G\rho_i/c^3$, $c = {c:g}$"
    )
    fig.tight_layout()
    path = outdir / "02_shell_crossing_arrest.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {
        "n": float(n),
        "crossing_time": float(crossing) if crossing is not None else np.nan,
        "min_jacobian_damped": float(np.min(j1)),
        "vmax_undamped": vmax0,
        "vmax_damped": vmax1,
    }
