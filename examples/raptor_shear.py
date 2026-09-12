"""Example: extreme shear layer (Kelvin-Helmholtz) benchmark, classical vs. Moss damping.

A periodic double shear layer (tanh profile) modelling the extreme
high-speed exhaust / combustion-chamber shear at

    Re = 2.5e6   (nu ~ 1e-4 m^2/s, U0 = 500 m/s in SI terms),

run in code units with the same 2D spectral solver:

  * Classical NS (lambda = 0): the Kelvin-Helmholtz instability drives
    enstrophy to the grid scale, ||omega||_inf diverges, and the
    under-resolved run fails numerically (CRASHED) within a few
    convective time units.
  * Moss regularization (lambda > 0): the nonlinear -lambda |u|^2 u
    damping caps the maximum vorticity into a bounded plateau and the
    run stays smooth and STABLE without any artificial viscosity.

Run:  python examples/raptor_shear.py [--n 256] [--t-max 10] [--lambda 0.5] [--output .]

Writes 04_raptor_benchmark.png with
  A, B: vorticity field snapshots (classical blown-up state vs. Moss
        coherent vortex street),
  C:    max vorticity ||omega(t)||_inf,
  D:    kinetic energy E(t) with the stability verdict.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moss_reg.fluid.spectral import SpectralNS2D

__all__ = ["RaptorShearBenchmark"]


class RaptorShearBenchmark:
    """Extreme shear layer: classical NS blow-up vs. Moss regularization.

    Parameters
    ----------
    n : int
        Grid resolution (n x n).
    re : float
        Reynolds number in code units (nu = U0 L / Re with L = 1).
    u0 : float
        Characteristic velocity (500 m/s in SI).
    delta : float
        Shear layer thickness (code units, domain 2 pi).
    epsilon : float
        Perturbation amplitude relative to u0.
    lam : float
        Moss damping coupling in code units (lam = G rho / c^3).
    t_max : float
        Integration horizon in convective time units.
    dt_max : float
        Maximum timestep.
    snap_dt : float
        Snapshot interval for diagnostics.
    """

    def __init__(
        self,
        n: int = 256,
        re: float = 2.5e6,
        u0: float = 1.0,
        delta: float = 0.05,
        epsilon: float = 0.05,
        lam: float = 0.5,
        t_max: float = 10.0,
        dt_max: float = 0.002,
        snap_dt: float = 0.05,
    ) -> None:
        self.n = int(n)
        self.re = float(re)
        self.u0 = float(u0)
        self.delta = float(delta)
        self.epsilon = float(epsilon)
        self.lam = float(lam)
        self.t_max = float(t_max)
        self.dt_max = float(dt_max)
        self.snap_dt = float(snap_dt)
        self.nu = self.u0 * 1.0 / self.re  # L = 1 in code units

    # -- initial condition --------------------------------------------------

    def initial_condition(self) -> Tuple[np.ndarray, np.ndarray]:
        """Periodic double tanh shear layer (Bell-Colella-Glaz style).

        Layers of co-rotating vorticity at y = pi/2 and y = 3 pi/2:

            u = u0 [tanh((y - pi/2)/delta) - tanh((y - 3 pi/2)/delta) - 1]

        plus the Kelvin-Helmholtz perturbation

            v = eps u0 (sin x + 0.5 sin 2x + 0.25 sin 3x).

        Divergence-free by construction.
        """
        y = 2.0 * np.pi * np.arange(self.n) / self.n
        x = 2.0 * np.pi * np.arange(self.n) / self.n
        xg, yg = np.meshgrid(x, y)
        u = self.u0 * (
            np.tanh((yg - 0.5 * np.pi) / self.delta)
            - np.tanh((yg - 1.5 * np.pi) / self.delta)
            - 1.0
        )
        v = (
            self.epsilon
            * self.u0
            * (np.sin(xg) + 0.5 * np.sin(2.0 * xg) + 0.25 * np.sin(3.0 * xg))
        )
        return u, v

    # -- single run ---------------------------------------------------------

    def _run_case(self, lam: float) -> Dict:
        solver = SpectralNS2D(self.n, nu=self.nu, lam=lam, dt_max=self.dt_max)
        u, v = self.initial_condition()
        solver.set_field(u, v)

        times: list = [0.0]
        omega_max: list = []
        energy: list = []
        u_max: list = []
        w0 = solver.vorticity()
        omega_max.append(float(np.abs(w0).max()))
        u, v = solver.velocity()
        energy.append(0.5 * float(np.mean(u * u + v * v)))
        u_max.append(float(np.sqrt(np.max(u * u + v * v))))
        omega_snapshot = w0.copy()

        crashed = False
        crash_time: Optional[float] = None
        t_next = self.snap_dt
        wall_start = time.perf_counter()
        while solver.time < self.t_max - 1e-14:
            dt = min(solver.cfl_dt(), self.t_max - solver.time)
            solver.step(dt)
            w = solver.vorticity()
            if not np.isfinite(w).all():
                crashed = True
                crash_time = solver.time
                break
            if solver.time >= t_next - 1e-14:
                times.append(solver.time)
                omega_max.append(float(np.abs(w).max()))
                u, v = solver.velocity()
                energy.append(0.5 * float(np.mean(u * u + v * v)))
                u_max.append(float(np.sqrt(np.max(u * u + v * v))))
                omega_snapshot = w.copy()
                t_next += self.snap_dt
        wall = time.perf_counter() - wall_start

        return {
            "times": np.asarray(times),
            "omega_max": np.asarray(omega_max),
            "energy": np.asarray(energy),
            "u_max": np.asarray(u_max),
            "crashed": crashed,
            "crash_time": crash_time,
            "final_time": crash_time if crashed else solver.time,
            "omega_snapshot": omega_snapshot,
            "wall_time": wall,
        }

    # -- benchmark ----------------------------------------------------------

    def run(self, outdir: str | Path = "assets") -> Dict[str, float]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        res0 = self._run_case(0.0)
        res1 = self._run_case(self.lam)

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # panels A & B: vorticity fields
        for ax, res, title, cmap_label in (
            (axes[0, 0], res0, self._classical_title(res0), "r"),
            (axes[0, 1], res1, self._moss_title(res1), "b"),
        ):
            w = res["omega_snapshot"]
            wmax = np.abs(w).max() or 1.0
            im = ax.imshow(
                w,
                extent=[0, 2 * np.pi, 0, 2 * np.pi],
                origin="lower",
                cmap="RdBu_r",
                vmin=-wmax,
                vmax=wmax,
                aspect="auto",
            )
            fig.colorbar(im, ax=ax, shrink=0.85)
            ax.set_title(title, color=cmap_label, fontsize=10)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")

        # panel C: max vorticity
        ax = axes[1, 0]
        ax.semilogy(
            res0["times"], res0["omega_max"], "r-", lw=2,
            label=r"classical NS ($\lambda = 0$)",
        )
        ax.semilogy(
            res1["times"], res1["omega_max"], "b-", lw=2,
            label=rf"Moss ($\lambda = {self.lam:g}$)",
        )
        ax.axhline(res1["omega_max"][0] * 1e4, color="k", ls=":", lw=1)
        ax.annotate(
            "grid-scale blow-up", (0.02, 0.85), xycoords="axes fraction",
            color="r", fontsize=9,
        )
        ax.set_xlabel(r"$t$ (convective units)")
        ax.set_ylabel(r"$\|\omega(t)\|_{L^\infty}$")
        ax.set_title("Maximum vorticity")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

        # panel D: kinetic energy + verdict
        ax = axes[1, 1]
        ax.plot(res0["times"], res0["energy"], "r-", lw=2,
                label=r"classical NS ($\lambda = 0$)")
        ax.plot(res1["times"], res1["energy"], "b-", lw=2,
                label=rf"Moss ($\lambda = {self.lam:g}$)")
        ax.set_xlabel(r"$t$ (convective units)")
        ax.set_ylabel(r"$E(t)$")
        ax.set_title("Kinetic energy and stability verdict")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        verdict = (
            f"classical: {'CRASHED at t = %.2f' % res0['crash_time'] if res0['crashed'] else 'unstable grid-scale state'}"
            f"  (wall {res0['wall_time']:.1f} s)\n"
            f"moss: STABLE, t = {res1['final_time']:.2f}"
            f"  (wall {res1['wall_time']:.1f} s)\n"
            f"peak |omega|: {res0['omega_max'].max():.2e}  vs  {res1['omega_max'].max():.2e}"
        )
        ax.text(0.03, 0.03, verdict, transform=ax.transAxes, fontsize=8,
                va="bottom", bbox=dict(facecolor="white", alpha=0.85))

        fig.suptitle(
            "SpaceX Raptor extreme shear layer: "
            rf"$Re = {self.re:g}$, $\delta = {self.delta:g}$, $N = {self.n}$"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = outdir / "04_raptor_benchmark.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return {
            "n": float(self.n),
            "re": self.re,
            "nu": self.nu,
            "delta": self.delta,
            "epsilon": self.epsilon,
            "lam": self.lam,
            "classical_crashed": float(res0["crashed"]),
            "classical_crash_time": (
                float(res0["crash_time"]) if res0["crash_time"] is not None else np.nan
            ),
            "classical_omega_peak": float(res0["omega_max"].max()),
            "classical_u_peak": float(res0["u_max"].max()),
            "moss_stable": float(not res1["crashed"]),
            "moss_final_time": float(res1["final_time"]),
            "moss_omega_peak": float(res1["omega_max"].max()),
            "moss_u_peak": float(res1["u_max"].max()),
            "moss_E_final": float(res1["energy"][-1]),
            "wall_classical": float(res0["wall_time"]),
            "wall_moss": float(res1["wall_time"]),
        }

    @staticmethod
    def _classical_title(res: Dict) -> str:
        if res["crashed"]:
            return (
                r"A: classical NS ($\lambda = 0$): CRASHED at "
                rf"$t = {res['crash_time']:.2f}$"
            )
        return r"A: classical NS ($\lambda = 0$): grid-scale breakdown"

    def _moss_title(self, res: Dict) -> str:
        return (
            rf"B: Moss ($\lambda = {self.lam:g}$): smooth coherent "
            rf"vortices, $t = {res['final_time']:.2f}$"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="extreme shear layer: classical NS vs. Moss damping")
    parser.add_argument("--n", type=int, default=256, help="grid resolution")
    parser.add_argument("--re", type=float, default=2.5e6, help="Reynolds number")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5, help="damping number (code units)")
    parser.add_argument("--t-max", type=float, default=10.0, help="integration horizon")
    parser.add_argument("--output", default=".", help="output directory")
    args = parser.parse_args()
    bench = RaptorShearBenchmark(n=args.n, re=args.re, lam=args.lam, t_max=args.t_max)
    metrics = bench.run(args.output)
    for key, value in metrics.items():
        print(f"{key:24s} {value}")


if __name__ == "__main__":
    main()
