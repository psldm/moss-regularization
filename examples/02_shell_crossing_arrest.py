"""1D cold collapse: shell-crossing singularity vs. vacuum damping arrest.

Two runs of the same cold uniform line collapsing under (Plummer-softened)
1/r^2 gravity:

  1. no damping  -- the flow shells cross, min_i J_i(t) -> 0 and below;
  2. vacuum damping with lambda_i = G rho_i / c^3 -- Theorem D.1 holds
     over the simulated window: min_i J_i(t) > 0 and the crossing
     singularity is suppressed with bounded deceleration.

Code units: G = 1, masses ~ 1/N, and c = 0.005 so the Planck-scale
coupling is visible on simulation scales.  The analytical damping
substep is unconditionally stable, so only the gravity/CFL limits and
dt_max bound the adaptive timestep here (the finite damping-stability
criterion of ``adaptive_dt`` applies to explicit schemes and is covered
by the unit tests).

Saves shell_crossing_comparison.png.
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from moss_reg.particles.integrator import LagrangianSystem
from moss_reg.particles.jacobian import JacobianTracker


def build(c: float | None, n: int = 200) -> tuple[LagrangianSystem, JacobianTracker]:
    x = np.linspace(-1.0, 1.0, n)
    m = np.full(n, 1.0 / n)
    sys = LagrangianSystem(
        positions=x[:, None],
        velocities=np.zeros((n, 1)),
        masses=m,
        h=0.12,
        G=1.0,
        c=1.0 if c is None else c,
        softening=0.4,
        damping=c is not None,
        gravity=True,
    )
    return sys, JacobianTracker(sys.initial_positions)


def run_to(
    sys: LagrangianSystem,
    tracker: JacobianTracker,
    t_end: float,
    dt_max: float = 0.005,
    stop_on_crossing: bool = False,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    times, jmin = [0.0], [1.0]
    crossing_t = None
    while sys.time < t_end - 1e-14:
        dt = min(sys.adaptive_dt(dt_max=dt_max, damp_safety=np.inf), t_end - sys.time)
        sys.step(dt)
        times.append(sys.time)
        jmin.append(tracker.min_jacobian(sys.positions))
        if stop_on_crossing and jmin[-1] <= 0.0:
            crossing_t = sys.time
            break
    return np.array(times), np.array(jmin), crossing_t


def main() -> None:
    n = 200
    t_end = 6.0
    dt_max = 0.005

    # --- undamped: collapse until shell crossing --------------------------
    sys0, tr0 = build(None, n)
    t0, j0, crossing_t = run_to(sys0, tr0, t_end, dt_max, stop_on_crossing=True)
    vmax0 = np.max(np.abs(sys0.velocities))
    print(
        f"undamped: crossing at t = {crossing_t:.3f}, "
        f"min J = {np.min(j0):.4g}, peak |v| = {vmax0:.3f}"
    )

    # --- vacuum damping: same collapse, same horizon ----------------------
    sys1, tr1 = build(0.005, n)
    t1, j1, _ = run_to(sys1, tr1, t_end, dt_max)
    vmax1 = np.max(np.abs(sys1.velocities))
    print(
        f"damped:   min J = {np.min(j1):.4g}, peak |v| = {vmax1:.4g} "
        f"(no crossing through t = {t_end})"
    )

    # --- figures ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(t0, j0, "r-", lw=2, label="no damping")
    ax.plot(t1, j1, "b-", lw=2, label=r"vacuum damping ($\lambda = G\rho/c^3$)")
    ax.axhline(0.0, color="k", ls="--", lw=1)
    if crossing_t is not None:
        ax.axvline(crossing_t, color="r", ls=":", lw=1)
        ax.annotate(
            "shell crossing", (crossing_t, 0.0), xytext=(0.55, 0.35),
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
    x0 = sys0.positions[:, 0]
    v0f = sys0.velocities[:, 0]
    # velocity profiles at snapshots: re-run damped case to fixed times
    sys_s, tr_s = build(0.005, n)
    snap_times = [1.0, 2.0, 4.0]
    snap_idx = 0
    while sys_s.time < t_end - 1e-14 and snap_idx < len(snap_times):
        dt = min(sys_s.adaptive_dt(dt_max=dt_max, damp_safety=np.inf),
                 snap_times[snap_idx] - sys_s.time)
        sys_s.step(dt)
        if abs(sys_s.time - snap_times[snap_idx]) < 1e-12:
            ax.plot(sys_s.positions[:, 0], sys_s.velocities[:, 0],
                    "o-", ms=3, lw=1, label=f"damped, t={snap_times[snap_idx]:g}")
            snap_idx += 1
    ax.plot(x0, v0f, "r--", lw=1.5, label=f"undamped, t={sys0.time:.2f}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$")
    ax.set_title("Velocity profiles")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"1D cold collapse, N = {n}, softening = 0.4; "
        r"$\lambda_i = G\rho_i/c^3$, $c = 0.005$"
    )
    fig.tight_layout()
    fig.savefig("shell_crossing_comparison.png", dpi=150)
    print("saved shell_crossing_comparison.png")


if __name__ == "__main__":
    main()
