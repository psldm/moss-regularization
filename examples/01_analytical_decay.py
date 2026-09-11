"""Analytical decay law vs. explicit Euler integration.

Plots the exact damping solution

    v(t) = v0 / sqrt(1 + 2 * lambda_eff * |v0|^2 * t)

alongside explicit Euler substeps for a few dt choices, demonstrating
convergence of the discrete scheme toward the closed-form curve.
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from moss_reg.core.analytical import euler_step, exact_damping


def main() -> None:
    v0 = np.array([1.0, 0.25, -0.5])
    speed0 = np.linalg.norm(v0)
    lam = 1.0  # natural units: c = G = 1  =>  lambda = 1
    T = 4.0

    t_fine = np.linspace(0.0, T, 400)
    v_exact = exact_damping(v0, t_fine, lam)
    speed_exact = np.linalg.norm(v_exact, axis=-1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_fine, speed_exact, "k-", lw=2, label="exact")
    for n in (4, 8, 16, 64):
        dt = T / n
        t = np.linspace(0.0, T, n + 1)
        v = np.tile(v0, (n + 1, 1))
        for i in range(n):
            v[i + 1] = euler_step(v[i], dt, lam)
        ax.plot(t, np.linalg.norm(v, axis=-1), "o--", ms=3, label=f"euler dt={dt:g}")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$|\mathbf{v}|$")
    ax.set_title(
        r"$\dot{\mathbf{v}} = -\lambda |\mathbf{v}|^2 \mathbf{v}$,"
        rf" $v_0 = {speed0:.2f}$, $\lambda = {lam}$"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("decay_comparison.png", dpi=150)
    print(f"saved decay_comparison.png (v0={speed0:.3f}, lambda={lam})")


if __name__ == "__main__":
    main()
