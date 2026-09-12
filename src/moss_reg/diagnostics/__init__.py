"""Runtime diagnostics and invariants.

norms    : Sobolev-type norms of periodic grid fields (L2, H1 seminorm,
           enstrophy, L4 on a padded grid, L-infinity, divergence)
energy   : time-integrated energy budget  E(t) = E0 - D_visc(t) - E_vac(t)
monitor  : DiagnosticsLog, a per-step recorder with CSV / JSON export

The fluid solver exposes ``SpectralNS2D.diagnostics()`` and the particle
system ``LagrangianSystem.diagnostics()``; both return plain dictionaries
that ``DiagnosticsLog.record`` accepts directly.
"""

from .energy import EnergyBudget
from .monitor import DiagnosticsLog
from .norms import field_norms, pad_spectrum

__all__ = ["EnergyBudget", "DiagnosticsLog", "field_norms", "pad_spectrum"]
