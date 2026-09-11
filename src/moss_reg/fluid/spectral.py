"""2D spectral (Fourier) incompressible Navier-Stokes solver with the
Moss vacuum-damping regularization.

Solves the momentum form on [0, 2 pi]^2,

    d u / dt + u . grad u = -grad p + nu Lap u - lam |u|^2 u,
    div u = 0,

with the pressure handled by a divergence-free projection in spectral
space.  The damping term -lam |u|^2 u (lam = G rho / c^3 in code units)
is purely dissipative:

    d/dt (1/2 ||u||^2) = -nu ||grad u||^2 - lam ||u||_{L4}^4.

Dealiasing uses the 2/3 rule; time stepping is explicit fourth-order
Runge-Kutta with projection.  For lam = 0 the 2D Taylor-Green vortex

    u = (sin x cos y, -cos x sin y)

is an exact viscous eigenmode, u(t) = u(0) exp(-2 nu t), which is used
to verify the solver.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["SpectralNS2D"]


class SpectralNS2D:
    """2D incompressible NS with Moss damping, momentum formulation.

    Parameters
    ----------
    n : int
        Grid resolution (n x n collocation points).
    nu : float
        Kinematic viscosity.
    lam : float
        Damping coupling in code units (lam = 0 recovers classical NS).
    dt_max : float
        Maximum timestep.
    dealias : bool
        Apply the 2/3 dealiasing rule.
    """

    def __init__(
        self,
        n: int,
        nu: float = 1e-3,
        lam: float = 0.0,
        dt_max: float = 0.005,
        dealias: bool = True,
    ) -> None:
        self.n = int(n)
        self.nu = float(nu)
        self.lam = float(lam)
        self.dt_max = float(dt_max)
        self.time = 0.0
        self.steps = 0

        k = np.fft.fftfreq(self.n) * self.n
        self.kx, self.ky = np.meshgrid(k, k)
        self.k2 = self.kx**2 + self.ky**2
        self.mask = (
            (self.k2 <= (self.n / 3.0) ** 2) if dealias else np.ones_like(self.k2, bool)
        )

        x = 2.0 * np.pi * np.arange(self.n) / self.n
        self.x, self.y = np.meshgrid(x, x)
        self.u_hat = np.zeros((self.n, self.n), dtype=np.complex128)
        self.v_hat = np.zeros((self.n, self.n), dtype=np.complex128)

    # -- initial conditions ------------------------------------------------

    def set_taylor_green(self) -> None:
        """Classic 2D Taylor-Green vortex, u = (sin x cos y, -cos x sin y)."""
        u = np.sin(self.x) * np.cos(self.y)
        v = -np.cos(self.x) * np.sin(self.y)
        self.u_hat = np.fft.fft2(u)
        self.v_hat = np.fft.fft2(v)
        self._enforce_mask()

    def set_random(self, n_modes: int = 8, amplitude: float = 1.0, seed: int = 0) -> None:
        """Random smooth solenoidal initial condition."""
        rng = np.random.default_rng(seed)
        u = np.zeros_like(self.x)
        v = np.zeros_like(self.y)
        for _ in range(n_modes):
            kx = rng.integers(1, self.n // 4)
            ky = rng.integers(1, self.n // 4)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            u += amplitude / np.sqrt(n_modes) * np.sin(kx * self.x) * np.cos(ky * self.y + phase)
            v += -amplitude / np.sqrt(n_modes) * kx / max(ky, 1) * np.cos(kx * self.x) * np.sin(ky * self.y + phase)
        self.u_hat = np.fft.fft2(u)
        self.v_hat = np.fft.fft2(v)
        self.u_hat, self.v_hat = self._project(self.u_hat, self.v_hat)
        self._enforce_mask()

    # -- spectral operations ------------------------------------------------

    def _enforce_mask(self) -> None:
        self.u_hat *= self.mask
        self.v_hat *= self.mask
        self.u_hat[0, 0] = 0.0
        self.v_hat[0, 0] = 0.0

    def _project(
        self, u_hat: np.ndarray, v_hat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove the divergent part: u <- u - k (k . u) / k^2."""
        u_hat = u_hat.copy()
        v_hat = v_hat.copy()
        nz = self.k2 > 0.0
        div = (self.kx * u_hat + self.ky * v_hat) / np.where(nz, self.k2, 1.0)
        u_hat[nz] -= self.kx[nz] * div[nz]
        v_hat[nz] -= self.ky[nz] * div[nz]
        u_hat[0, 0] = 0.0
        v_hat[0, 0] = 0.0
        return u_hat, v_hat

    def velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.fft.ifft2(self.u_hat).real, np.fft.ifft2(self.v_hat).real

    def vorticity(self) -> np.ndarray:
        """omega = d_x v - d_y u."""
        w_hat = 1j * self.kx * self.v_hat - 1j * self.ky * self.u_hat
        return np.fft.ifft2(w_hat).real

    # -- right-hand side ---------------------------------------------------

    def _rhs(
        self, u_hat: np.ndarray, v_hat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """RHS of the momentum equation for a (projected) spectral state."""
        u_hat, v_hat = self._project(u_hat, v_hat)
        u = np.fft.ifft2(u_hat).real
        v = np.fft.ifft2(v_hat).real
        ux = np.fft.ifft2(1j * self.kx * u_hat).real
        uy = np.fft.ifft2(1j * self.ky * u_hat).real
        vx = np.fft.ifft2(1j * self.kx * v_hat).real
        vy = np.fft.ifft2(1j * self.ky * v_hat).real

        adv_u = -(u * ux + v * uy)
        adv_v = -(u * vx + v * vy)
        rhs_u = -self.nu * self.k2 * u_hat + np.fft.fft2(adv_u)
        rhs_v = -self.nu * self.k2 * v_hat + np.fft.fft2(adv_v)
        if self.lam:
            f = u * u + v * v
            rhs_u -= self.lam * np.fft.fft2(f * u)
            rhs_v -= self.lam * np.fft.fft2(f * v)
        rhs_u *= self.mask
        rhs_v *= self.mask
        rhs_u[0, 0] = 0.0
        rhs_v[0, 0] = 0.0
        return rhs_u, rhs_v

    # -- integration -------------------------------------------------------

    def step(self, dt: Optional[float] = None) -> None:
        """One explicit RK4 step with projection after each stage input."""
        dt = self.dt_max if dt is None else float(dt)
        u0, v0 = self.u_hat, self.v_hat
        k1u, k1v = self._rhs(u0, v0)
        k2u, k2v = self._rhs(u0 + 0.5 * dt * k1u, v0 + 0.5 * dt * k1v)
        k3u, k3v = self._rhs(u0 + 0.5 * dt * k2u, v0 + 0.5 * dt * k2v)
        k4u, k4v = self._rhs(u0 + dt * k3u, v0 + dt * k3v)
        self.u_hat = u0 + dt / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        self.v_hat = v0 + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        self.u_hat, self.v_hat = self._project(self.u_hat, self.v_hat)
        self._enforce_mask()
        self.time += dt
        self.steps += 1

    def cfl_dt(self, cfl: float = 0.5) -> float:
        """Stability-limited dt for the explicit scheme."""
        u, v = self.velocity()
        umax = float(np.sqrt(np.max(u * u + v * v)))
        kmax = float(np.sqrt(np.max(self.k2 * self.mask)))
        dt_adv = np.inf if umax == 0.0 else cfl / (kmax * umax)
        dt_visc = np.inf if self.nu == 0.0 else cfl * 2.0 / (self.nu * kmax**2)
        return float(min(self.dt_max, dt_adv, dt_visc))

    def advance(self, t_end: float, callback: Optional[callable] = None) -> None:
        while self.time < t_end - 1e-14:
            dt = min(self.cfl_dt(), t_end - self.time)
            self.step(dt)
            if callback is not None:
                callback(self)

    # -- diagnostics -------------------------------------------------------

    def measure(self) -> Tuple[float, float, float]:
        """(E, Omega, L4): kinetic energy, enstrophy, L4 velocity norm.

        E     = 1/2 <|u|^2>,         Omega = 1/2 <omega^2>,
        L4    = <|u|^4>^(1/4),       < . > = spatial mean.
        """
        u, v = self.velocity()
        w = self.vorticity()
        e = 0.5 * float(np.mean(u * u + v * v))
        omega = 0.5 * float(np.mean(w * w))
        l4 = float(np.mean((u * u + v * v) ** 2)) ** 0.25
        return e, omega, l4
