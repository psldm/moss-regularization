"""Benchmark 1: analytical velocity damping vs. explicit Euler.

Writes assets/01_velocity_decay.png with
  (a) the exact decay law  v(t) = v0 / sqrt(1 + 2 lam |v0|^2 t)  against
      explicit Euler substeps for several dt,
  (b) the Euler error vs. dt, demonstrating first-order convergence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.analytical import euler_step, exact_damping
from ._style import INK, SERIES, apply_style

__all__ = ["run_decay_benchmark"]


def run_decay_benchmark(
    outdir: str | Path = "assets",
    lambda_eff: float = 1.0,
    v0: Sequence[float] = (1.0, 0.25, -0.5),
    t_max: float = 4.0,
    dt_max: float = 0.02,
) -> Dict[str, float]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    apply_style()
    v0 = np.asarray(v0, dtype=np.float64)
    speed0 = float(np.linalg.norm(v0))
    lam = float(lambda_eff)
    T = float(t_max)

    t_fine = np.linspace(0.0, T, 800)
    v_exact = exact_damping(v0, t_fine, lam)
    speed_exact = np.linalg.norm(v_exact, axis=-1)

    # convergence: Euler error at t = T for halved dt
    n_list = [16, 32, 64, 128, 256, 512]
    errors = []
    for n in n_list:
        dt = T / n
        v = v0.copy()
        for _ in range(n):
            v = euler_step(v, dt, lam)
        errors.append(np.linalg.norm(v - exact_damping(v0, T, lam)))
    errors = np.asarray(errors)
    dts = T / np.asarray(n_list, dtype=np.float64)
    slope = np.polyfit(np.log(dts), np.log(errors), 1)[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(t_fine, speed_exact, color=INK, lw=2, label="exact")
    for series, n in enumerate((16, 64, 256)):
        dt = T / n
        t = np.linspace(0.0, T, n + 1)
        v = np.tile(v0, (n + 1, 1))
        for i in range(n):
            v[i + 1] = euler_step(v[i], dt, lam)
        ax.plot(t, np.linalg.norm(v, axis=-1), "o--", ms=3, color=SERIES[series],
                label=f"euler dt={dt:g}")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$|\mathbf{v}|$")
    ax.set_title(
        r"$\dot{\mathbf{v}} = -\lambda |\mathbf{v}|^2 \mathbf{v}$, "
        rf"$|\mathbf{{v}}_0| = {speed0:.2f}$, $\lambda = {lam:g}$"
    )
    ax.legend()

    ax = axes[1]
    ax.loglog(dts, errors, "o-", color=SERIES[0], label=f"Euler error (slope {slope:.2f})")
    ax.loglog(dts, errors[0] * (dts / dts[0]) ** 1.0, color=INK, ls="--", lw=1, label=r"$\propto \Delta t$")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$|\mathbf{v}_{\mathrm{euler}} - \mathbf{v}_{\mathrm{exact}}|$")
    ax.set_title("Convergence at fixed horizon")
    ax.legend()
    ax.grid(True, which="both")

    fig.suptitle("Analytical vs. numerical integration of the damping ODE")
    fig.tight_layout()
    path = outdir / "01_velocity_decay.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {
        "speed0": speed0,
        "lambda": lam,
        "euler_slope": float(slope),
        "params": {"lambda_eff": lam, "v0": v0.tolist(), "t_max": T, "dt_max": dt_max},
        "checks": {"euler_first_order": 0.9 <= float(slope) <= 1.1},
        "figure": str(path),
    }
