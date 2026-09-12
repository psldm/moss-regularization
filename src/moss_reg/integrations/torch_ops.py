"""PyTorch adapter: exact damping substep for tensors (CPU/GPU, autograd-safe).

``torch`` is imported lazily; moss-reg does not depend on it.
"""

from __future__ import annotations

from typing import Any

__all__ = ["damp"]


def damp(v: Any, dt: float, lam: Any, alpha: float = 2.0) -> Any:
    """v <- v (1 + alpha lam |v|^alpha dt)^(-1/alpha) along the last axis.

    ``lam`` may be a float or a tensor broadcastable against ``v[..., 0]``.
    """
    import torch  # noqa: WPS433 (lazy optional dependency)

    v = torch.as_tensor(v)
    speed2 = torch.sum(v * v, dim=-1, keepdim=True)
    lam_t = torch.as_tensor(lam, dtype=v.dtype, device=v.device)
    if lam_t.dim() > 0:
        lam_t = lam_t.unsqueeze(-1)
    if alpha == 2.0:
        factor = torch.rsqrt(1.0 + 2.0 * lam_t * speed2 * dt)
    else:
        factor = torch.pow(1.0 + alpha * lam_t * torch.pow(speed2, 0.5 * alpha) * dt, -1.0 / alpha)
    return v * factor
