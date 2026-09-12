"""Adapters for existing solvers.

numpy_ops : ``damp(v, dt, lam, alpha=2)`` for NumPy arrays (any shape, last
            axis = components), per-item ``lam`` allowed; returns the damped
            velocities and, optionally, the kinetic energy removed.
torch_ops : the same for PyTorch tensors (CPU/GPU, differentiable); torch is
            imported lazily and is not a dependency of moss-reg.

Non-Python adapters live in the repository directory ``integrations/``:
``c/moss_damp.h`` (header-only C99/C++), ``csharp/MossDamp.cs``,
``dedalus/`` and ``openfoam/`` (source-term examples).
"""

from .numpy_ops import damp, damp_with_energy

__all__ = ["damp", "damp_with_energy"]
