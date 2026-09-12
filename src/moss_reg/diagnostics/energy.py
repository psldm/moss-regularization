"""Time-integrated energy budget of the damped Navier-Stokes system.

    d/dt E = -nu <|grad u|^2> - <lambda |u|^4>          (E = 1/2 <|u|^2>)

integrated with the trapezoid rule on the recorded samples:

    D_visc(t) = nu  int_0^t <|grad u|^2> dt'     viscous dissipation
    E_vac(t)  =     int_0^t <lambda |u|^4> dt'   energy handed to the vacuum reservoir
    r(t)      = E(t) - E(0) + D_visc(t) + E_vac(t)   residual (0 up to quadrature/time-stepping error)

For particle systems the analogous budget is E_kin + E_pot + E_vac = const,
with E_vac accumulated exactly by the integrator (see
``LagrangianSystem.energy_to_vacuum``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.integrate import cumulative_trapezoid

__all__ = ["EnergyBudget"]


@dataclass
class EnergyBudget:
    t: np.ndarray
    E: np.ndarray
    visc_power: np.ndarray     # nu <|grad u|^2>
    vac_power: np.ndarray      # <lambda |u|^4>

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=np.float64)
        self.E = np.asarray(self.E, dtype=np.float64)
        self.visc_power = np.asarray(self.visc_power, dtype=np.float64)
        self.vac_power = np.asarray(self.vac_power, dtype=np.float64)
        n = self.t.size
        if not (self.E.size == self.visc_power.size == self.vac_power.size == n):
            raise ValueError("all series must have the same length")

    @property
    def d_visc(self) -> np.ndarray:
        return cumulative_trapezoid(self.visc_power, self.t, initial=0.0)

    @property
    def e_vac(self) -> np.ndarray:
        return cumulative_trapezoid(self.vac_power, self.t, initial=0.0)

    @property
    def residual(self) -> np.ndarray:
        return self.E - self.E[0] + self.d_visc + self.e_vac

    @property
    def max_abs_residual(self) -> float:
        return float(np.max(np.abs(self.residual))) if self.t.size else 0.0

    def summary(self) -> Dict[str, float]:
        return {
            "E0": float(self.E[0]),
            "E_final": float(self.E[-1]),
            "D_visc_final": float(self.d_visc[-1]),
            "E_vac_final": float(self.e_vac[-1]),
            "residual_final": float(self.residual[-1]),
            "max_abs_residual": self.max_abs_residual,
            "vac_fraction": float(self.e_vac[-1] / max(self.E[0] - self.E[-1], 1e-300)),
        }
