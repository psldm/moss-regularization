"""Norms of periodic velocity fields on uniform grids (any dimension).

``components[i]`` is the velocity component along array axis ``i``; every
component has the same shape ``(n_0, ..., n_{d-1})`` and the box has
length ``box`` (scalar or per-axis) with ``x_j = box_j * m / n_j``.
Derivatives are spectral.  The L4 norm and the maximum are evaluated on a
grid zero-padded to twice the resolution, which is exact for fields
band-limited to the 2/3 dealiasing band.
"""

from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np

__all__ = ["field_norms", "pad_spectrum", "wavenumbers"]

Box = Union[float, Sequence[float]]


def wavenumbers(shape: Sequence[int], box: Box = 2.0 * np.pi) -> list:
    """Wavenumber grids ``K[j]`` (``indexing='ij'``) for a periodic box."""
    boxes = np.broadcast_to(np.asarray(box, dtype=np.float64), (len(shape),))
    ks = [np.fft.fftfreq(n) * n * (2.0 * np.pi / b) for n, b in zip(shape, boxes)]
    return list(np.meshgrid(*ks, indexing="ij"))


def pad_spectrum(f_hat: np.ndarray, factor: int = 2) -> np.ndarray:
    """Zero-pad a numpy-convention spectrum to ``factor`` times the resolution
    and return the real-space field there (exact trigonometric interpolation)."""
    shape = f_hat.shape
    big_shape = tuple(factor * n for n in shape)
    big = np.zeros(big_shape, dtype=np.complex128)
    idx = []
    for n, m in zip(shape, big_shape):
        k = np.rint(np.fft.fftfreq(n) * n).astype(int)
        idx.append(k % m)
    big[np.ix_(*idx)] = f_hat
    scale = np.prod(big_shape) / np.prod(shape)
    return np.fft.ifftn(big).real * scale


def field_norms(components: Sequence[np.ndarray], box: Box = 2.0 * np.pi,
                pad: int = 2) -> Dict[str, float]:
    """Norms of a periodic velocity field.

    Returns a dictionary with
      ``E``            kinetic energy per unit volume, 1/2 <|u|^2>
      ``l2``           sqrt<|u|^2>
      ``grad_l2_sq``   <|grad u|^2>  (H^1 seminorm squared; = 2 * enstrophy
                       for divergence-free periodic fields)
      ``enstrophy``    1/2 <|omega|^2>  (2D scalar or 3D vector vorticity)
      ``l4``           <|u|^4>^(1/4)   (padded quadrature)
      ``l4_pow4``      <|u|^4>
      ``linf``         max |u|        (padded grid)
      ``div_max``      max |div u|
    """
    comps = [np.asarray(c, dtype=np.float64) for c in components]
    d = len(comps)
    shape = comps[0].shape
    if any(c.shape != shape for c in comps) or len(shape) != d:
        raise ValueError("need d components of identical d-dimensional shape")
    npts = float(np.prod(shape))
    K = wavenumbers(shape, box)
    hats = [np.fft.fftn(c) for c in comps]

    e = 0.5 * float(sum(np.mean(c * c) for c in comps))
    # Parseval: <|d_j u_i|^2> = sum_k k_j^2 |u_i_hat|^2 / N^2
    grad_sq = float(sum(np.sum((K[j] ** 2) * np.abs(h) ** 2) for h in hats for j in range(d)) / npts**2)
    # vorticity
    if d == 2:
        w_hat = 1j * K[0] * hats[1] - 1j * K[1] * hats[0]
        enstrophy = 0.5 * float(np.sum(np.abs(w_hat) ** 2) / npts**2)
    elif d == 3:
        w0 = 1j * K[1] * hats[2] - 1j * K[2] * hats[1]
        w1 = 1j * K[2] * hats[0] - 1j * K[0] * hats[2]
        w2 = 1j * K[0] * hats[1] - 1j * K[1] * hats[0]
        enstrophy = 0.5 * float(sum(np.sum(np.abs(w) ** 2) for w in (w0, w1, w2)) / npts**2)
    else:
        enstrophy = 0.5 * grad_sq
    div_hat = sum(1j * K[i] * hats[i] for i in range(d))
    div_max = float(np.max(np.abs(np.fft.ifftn(div_hat).real)))
    # padded quadrature for quartic and maximum
    if pad and pad > 1:
        big = [pad_spectrum(h, pad) for h in hats]
    else:
        big = comps
    speed2 = sum(b * b for b in big)
    l4_pow4 = float(np.mean(speed2 * speed2))
    linf = float(np.sqrt(np.max(speed2)))
    return {
        "E": e,
        "l2": float(np.sqrt(2.0 * e)),
        "grad_l2_sq": grad_sq,
        "enstrophy": enstrophy,
        "l4": l4_pow4 ** 0.25,
        "l4_pow4": l4_pow4,
        "linf": linf,
        "div_max": div_max,
    }
