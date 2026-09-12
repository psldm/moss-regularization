"""3D spectral (Fourier) incompressible Navier-Stokes solver with the cubic
damping term, on the periodic box [0, 2 pi]^3.

    d u / dt + u . grad u = -grad p + nu Lap u - lam |u|^2 u,   div u = 0.

Same construction as :class:`~moss_reg.fluid.spectral.SpectralNS2D`:
divergence-free projection, explicit RK4, 2/3 rule for the quadratic term
(written in rotational form, -(omega x u), whose gradient part is absorbed
by the projection), and the cubic term evaluated on a zero-padded 2n grid
(exact).  Velocity component ``i`` is along array axis ``i``
(``indexing='ij'``), matching :func:`moss_reg.diagnostics.field_norms`.

Verification cases (see tests): the shear mode u = (sin y, 0, 0) is an
exact solution decaying as exp(-nu t); the 3D Taylor-Green vortex
u = (sin x cos y cos z, -cos x sin y cos z, 0) is the standard transition
benchmark.  At the resolutions used here (N = 32..64) the classical
Taylor-Green run becomes under-resolved once the enstrophy grows; the
benchmark reports the resolution indicator k_max * eta (Kolmogorov
scale eta = (nu^3 / epsilon)^(1/4)) so that this is visible.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import fft as sfft

from ..diagnostics.norms import field_norms

__all__ = ["SpectralNS3D"]


class SpectralNS3D:
    PAD_FACTOR = 2

    def __init__(
        self,
        n: int,
        nu: float = 1e-3,
        lam: float = 0.0,
        dt_max: float = 0.01,
        dealias: bool = True,
        cubic_dealias: bool = True,
        workers: int = -1,
    ) -> None:
        self.n = int(n)
        self.nu = float(nu)
        self.dt_max = float(dt_max)
        self.cubic_dealias = bool(cubic_dealias)
        self.workers = int(workers)
        self.time = 0.0
        self.steps = 0

        k = np.fft.fftfreq(self.n) * self.n
        self.K = list(np.meshgrid(k, k, k, indexing="ij"))
        self.k2 = sum(Ki**2 for Ki in self.K)
        self.mask = (self.k2 <= (self.n / 3.0) ** 2) if dealias else np.ones_like(self.k2, bool)
        self.kmax = float(np.sqrt(np.max(self.k2 * self.mask)))

        self.n_pad = self.PAD_FACTOR * self.n
        kidx = np.rint(k).astype(int) % self.n_pad
        self._pad_index = np.ix_(kidx, kidx, kidx)
        self._pad_scale = (self.n_pad / self.n) ** 3

        x = 2.0 * np.pi * np.arange(self.n) / self.n
        self.X = list(np.meshgrid(x, x, x, indexing="ij"))
        self.u_hat: List[np.ndarray] = [np.zeros((self.n,) * 3, dtype=np.complex128) for _ in range(3)]
        self.set_lambda(lam)

    # -- coupling ----------------------------------------------------------

    def set_lambda(self, lam) -> None:
        """Scalar coupling or a (n, n, n) field lam(x) >= 0 (band-limited)."""
        arr = np.asarray(lam, dtype=np.float64)
        if arr.ndim == 0:
            self.lam = float(arr)
            self.lam_field = None
            self._lam_pad = None
            self.lam_max = self.lam
        else:
            if arr.shape != (self.n,) * 3:
                raise ValueError(f"lam field must have shape {(self.n,) * 3}")
            lam_hat = self._fft(arr) * self.mask
            self.lam_field = self._ifft(lam_hat)
            self._lam_pad = self._pad_to_real(lam_hat)
            self.lam = float(np.mean(self.lam_field))
            self.lam_max = float(np.max(self.lam_field))

    @property
    def has_damping(self) -> bool:
        return self.lam_field is not None or self.lam != 0.0

    # -- transforms --------------------------------------------------------

    def _fft(self, f: np.ndarray) -> np.ndarray:
        return sfft.fftn(f, workers=self.workers)

    def _ifft(self, f_hat: np.ndarray) -> np.ndarray:
        return sfft.ifftn(f_hat, workers=self.workers).real

    def _pad_to_real(self, f_hat: np.ndarray) -> np.ndarray:
        big = np.zeros((self.n_pad,) * 3, dtype=np.complex128)
        big[self._pad_index] = f_hat
        return sfft.ifftn(big, workers=self.workers).real * self._pad_scale

    def _real_to_unpadded(self, f: np.ndarray) -> np.ndarray:
        return sfft.fftn(f, workers=self.workers)[self._pad_index] / self._pad_scale

    # -- initial conditions ------------------------------------------------

    def set_field(self, components: Sequence[np.ndarray]) -> None:
        hats = [self._fft(np.asarray(c, dtype=np.float64)) for c in components]
        self.u_hat = self._project(hats)
        self._enforce_mask()

    def set_taylor_green(self, amplitude: float = 1.0) -> None:
        X, Y, Z = self.X
        u0 = amplitude * np.sin(X) * np.cos(Y) * np.cos(Z)
        u1 = -amplitude * np.cos(X) * np.sin(Y) * np.cos(Z)
        self.set_field([u0, u1, np.zeros_like(u0)])

    def set_shear_mode(self, amplitude: float = 1.0) -> None:
        """u = (A sin y, 0, 0): exact NS solution, decays as exp(-nu t)."""
        u0 = amplitude * np.sin(self.X[1])
        self.set_field([u0, np.zeros_like(u0), np.zeros_like(u0)])

    # -- spectral operations ----------------------------------------------

    def _enforce_mask(self) -> None:
        for h in self.u_hat:
            h *= self.mask
            h[0, 0, 0] = 0.0

    def _project(self, hats: Sequence[np.ndarray]) -> List[np.ndarray]:
        nz = self.k2 > 0.0
        div = sum(Ki * h for Ki, h in zip(self.K, hats)) / np.where(nz, self.k2, 1.0)
        out = []
        for Ki, h in zip(self.K, hats):
            g = h.copy()
            g[nz] -= Ki[nz] * div[nz]
            g[0, 0, 0] = 0.0
            out.append(g)
        return out

    def velocity(self) -> List[np.ndarray]:
        return [self._ifft(h) for h in self.u_hat]

    def vorticity_hat(self, hats: Optional[Sequence[np.ndarray]] = None) -> List[np.ndarray]:
        h = self.u_hat if hats is None else hats
        K0, K1, K2 = self.K
        return [
            1j * (K1 * h[2] - K2 * h[1]),
            1j * (K2 * h[0] - K0 * h[2]),
            1j * (K0 * h[1] - K1 * h[0]),
        ]

    def vorticity(self) -> List[np.ndarray]:
        return [self._ifft(w) for w in self.vorticity_hat()]

    def cubic_term(self, hats: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Spectra of |u|^2 u_i (unmasked), padded evaluation when enabled."""
        if self.cubic_dealias:
            u = [self._pad_to_real(h) for h in hats]
            f = sum(ui * ui for ui in u)
            return [self._real_to_unpadded(f * ui) for ui in u]
        u = [self._ifft(h) for h in hats]
        f = sum(ui * ui for ui in u)
        return [self._fft(f * ui) for ui in u]

    def damping_term(self, hats: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Spectra of lam |u|^2 u_i for scalar or field lam."""
        if self.lam_field is None:
            return [self.lam * c for c in self.cubic_term(hats)]
        if self.cubic_dealias:
            u = [self._pad_to_real(h) for h in hats]
            f = self._lam_pad * sum(ui * ui for ui in u)
            return [self._real_to_unpadded(f * ui) for ui in u]
        u = [self._ifft(h) for h in hats]
        f = self.lam_field * sum(ui * ui for ui in u)
        return [self._fft(f * ui) for ui in u]

    # -- right-hand side ---------------------------------------------------

    def _rhs(self, hats: Sequence[np.ndarray]) -> List[np.ndarray]:
        hats = self._project(hats)
        u = [self._ifft(h) for h in hats]
        w = [self._ifft(wh) for wh in self.vorticity_hat(hats)]
        # rotational form: -(omega x u); the gradient part of u.grad u is a
        # pure gradient and is removed by the projection
        adv = [
            -(w[1] * u[2] - w[2] * u[1]),
            -(w[2] * u[0] - w[0] * u[2]),
            -(w[0] * u[1] - w[1] * u[0]),
        ]
        rhs = [-self.nu * self.k2 * h + self._fft(a) for h, a in zip(hats, adv)]
        if self.has_damping:
            damp = self.damping_term(hats)
            rhs = [r - d for r, d in zip(rhs, damp)]
        out = []
        for r in rhs:
            r = r * self.mask
            r[0, 0, 0] = 0.0
            out.append(r)
        return out

    # -- integration -------------------------------------------------------

    def step(self, dt: Optional[float] = None) -> None:
        dt = self.dt_max if dt is None else float(dt)
        u0 = self.u_hat
        k1 = self._rhs(u0)
        k2 = self._rhs([a + 0.5 * dt * b for a, b in zip(u0, k1)])
        k3 = self._rhs([a + 0.5 * dt * b for a, b in zip(u0, k2)])
        k4 = self._rhs([a + dt * b for a, b in zip(u0, k3)])
        new = [a + dt / 6.0 * (b1 + 2.0 * b2 + 2.0 * b3 + b4)
               for a, b1, b2, b3, b4 in zip(u0, k1, k2, k3, k4)]
        self.u_hat = self._project(new)
        self._enforce_mask()
        self.time += dt
        self.steps += 1

    def dt_limits(self, cfl: float = 0.5) -> Dict[str, float]:
        u = self.velocity()
        umax2 = float(np.max(sum(ui * ui for ui in u)))
        umax = float(np.sqrt(umax2))
        dt_adv = np.inf if umax == 0.0 else cfl / (self.kmax * umax)
        dt_visc = np.inf if self.nu == 0.0 else cfl * 2.0 / (self.nu * self.kmax**2)
        dt_damp = (np.inf if (not self.has_damping or self.lam_max == 0.0 or umax2 == 0.0)
                   else cfl * 2.0 / (3.0 * self.lam_max * umax2))
        return {"advection": dt_adv, "viscosity": dt_visc, "damping": dt_damp}

    def cfl_dt(self, cfl: float = 0.5) -> float:
        lim = self.dt_limits(cfl)
        return float(min(self.dt_max, lim["advection"], lim["viscosity"], lim["damping"]))

    def advance(self, t_end: float, callback=None) -> None:
        while self.time < t_end - 1e-14:
            dt = min(self.cfl_dt(), t_end - self.time)
            self.step(dt)
            if callback is not None:
                callback(self)

    # -- diagnostics -------------------------------------------------------

    def diagnostics(self) -> dict:
        """E, enstrophy, grad_l2_sq, l4, linf, div_max, omega_max, vac_power,
        epsilon (viscous dissipation rate), eta (Kolmogorov scale) and
        kmax_eta (resolution indicator; >= 1 is resolved)."""
        u = self.velocity()
        d = field_norms(u, box=2.0 * np.pi, pad=self.PAD_FACTOR)
        w = self.vorticity()
        d["omega_max"] = float(np.sqrt(np.max(sum(wi * wi for wi in w))))
        if self.lam_field is None:
            d["vac_power"] = self.lam * d["l4_pow4"]
        else:
            up = [self._pad_to_real(h) for h in self.u_hat]
            d["vac_power"] = float(np.mean(self._lam_pad * sum(ui * ui for ui in up) ** 2))
        eps = self.nu * d["grad_l2_sq"]
        d["epsilon"] = eps
        eta = (self.nu**3 / eps) ** 0.25 if eps > 0 else np.inf
        d["eta"] = float(eta)
        d["kmax_eta"] = float(self.kmax * eta)
        d["steps"] = self.steps
        return d

    def measure(self):
        d = self.diagnostics()
        return d["E"], d["enstrophy"], d["l4"]
