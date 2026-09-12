"""``moss-reg validate``: PASS / FAIL self-checks.

Groups
------
formulas     dimensional and numerical claims of the README
             (:mod:`moss_reg.dimensions`; known open issues are XFAIL)
damping      exact ODE solution, flow composition, particle-integrator
             horizon regression (the 0.1.0 double-damping bug), Euler order
convergence  KDK second order (two-body), RK4 fourth order (damped TG)
particles    kernel normalization, dense vs tree density, Jacobian monitor,
             undamped collapse crosses, damped speed cap ~ c
fluid        Taylor-Green exact decay, divergence-free, energy identity,
             cubic-term dealiasing, damping stiffness limit

Every check returns a :class:`Result`; ``validate`` exits 1 if any hard
FAIL is present (``--strict`` also fails on XFAIL known issues).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from . import dimensions as D
from .core.analytical import euler_step, exact_damping
from .fluid.spectral import SpectralNS2D
from .particles import integrator as _imod
from .particles import kernel as _kmod
from .particles.integrator import LagrangianSystem
from .particles.jacobian import JacobianTracker, one_d_jacobians, sph_deformation_gradients
from .particles.kernel import compute_density, cubic_spline

__all__ = ["Result", "run_checks", "format_table", "has_failures"]

CheckFn = Callable[[], Tuple[bool, str]]


@dataclass(frozen=True)
class Result:
    group: str
    name: str
    status: str      # PASS | FAIL | XFAIL | XPASS | ERROR
    detail: str
    seconds: float

    @property
    def ok(self) -> bool:
        return self.status in ("PASS", "XFAIL", "XPASS")


# ---------------------------------------------------------------------------
# damping
# ---------------------------------------------------------------------------

def _exact_closed_form() -> Tuple[bool, str]:
    rng = np.random.default_rng(0)
    worst = 0.0
    for lam in (1.0, 3.7, 100.0):
        for t in (0.01, 1.0, 100.0):
            v0 = rng.normal(size=(20, 3))
            expect = v0 / np.sqrt(1.0 + 2.0 * lam * np.sum(v0 * v0, axis=1, keepdims=True) * t)
            got = exact_damping(v0, t, lam)
            worst = max(worst, float(np.max(np.abs(got - expect) / np.abs(expect))))
    return worst < 1e-13, f"max rel err {worst:.1e}"


def _flow_composition() -> Tuple[bool, str]:
    rng = np.random.default_rng(1)
    v0 = rng.normal(size=(20, 3))
    lam = rng.uniform(0.5, 5.0, size=20)
    worst = 0.0
    for a, b in ((0.1, 0.7), (2.0, 0.05), (1e-3, 1e3)):
        two = exact_damping(exact_damping(v0, a, lam), b, lam)
        one = exact_damping(v0, a + b, lam)
        worst = max(worst, float(np.max(np.abs(two - one) / np.abs(one))))
    return worst < 1e-12, f"phi_a o phi_b vs phi_(a+b): max rel err {worst:.1e}"


def _integrator_horizon() -> Tuple[bool, str]:
    rho = 4.0 / (3.0 * 0.5)
    lam = rho
    T = 1.0
    worst = 0.0
    for dt in (0.5, 0.05, 0.01):
        sys = LagrangianSystem(
            positions=np.array([[0.0]]), velocities=np.array([[1.0]]),
            masses=np.array([1.0]), h=0.5, G=1.0, c=1.0, damping=True, gravity=False,
        )
        while sys.time < T - 1e-12:
            sys.step(min(dt, T - sys.time))
        expected = 1.0 / np.sqrt(1.0 + 2.0 * lam * T)
        worst = max(worst, abs(sys.velocities[0, 0] - expected) / expected)
    wrong = 1.0 / np.sqrt(1.0 + 4.0 * lam * T)
    ok = worst < 1e-10 and abs(wrong - expected) > 0.1 * expected
    return ok, f"v(T) vs 1/sqrt(1+2 lam v0^2 T): max rel err {worst:.1e} (0.1.0 value would be {wrong:.4f} vs {expected:.4f})"


def _euler_order() -> Tuple[bool, str]:
    v0 = np.array([1.0, 0.5, -0.25])
    T, lam = 1.0, 1.0
    ref = exact_damping(v0, T, lam)
    errs, dts = [], []
    for n in (100, 200, 400, 800):
        dt = T / n
        v = v0.copy()
        for _ in range(n):
            v = euler_step(v, dt, lam)
        errs.append(np.linalg.norm(v - ref))
        dts.append(dt)
    slope = float(np.polyfit(np.log(dts), np.log(errs), 1)[0])
    return 0.9 <= slope <= 1.1, f"error slope {slope:.3f} (expect 1)"


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------

def _two_body(dt: float, T: float) -> np.ndarray:
    x = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    v = np.array([[0.0, -0.6, 0.0], [0.0, 0.6, 0.0]])
    sys = LagrangianSystem(x, v, np.ones(2), h=0.3, G=1.0, softening=0.1, damping=False, gravity=True)
    while sys.time < T - 1e-12:
        sys.step(min(dt, T - sys.time))
    return sys.positions.copy()


def _kdk_second_order() -> Tuple[bool, str]:
    T = 2.0
    ref = _two_body(0.0005, T)
    errs = [float(np.max(np.abs(_two_body(dt, T) - ref))) for dt in (0.04, 0.02, 0.01)]
    ratios = [errs[i] / errs[i + 1] for i in range(2)]
    ok = all(3.3 <= r <= 4.7 for r in ratios)
    return ok, f"error ratios per dt halving {[round(r, 2) for r in ratios]} (expect 4)"


def _rk4_fourth_order() -> Tuple[bool, str]:
    def energy_at(dt: float, T: float = 0.5) -> float:
        s = SpectralNS2D(32, nu=0.01, lam=0.5, dt_max=dt)
        s.set_taylor_green()
        while s.time < T - 1e-12:
            s.step(min(dt, T - s.time))
        return s.measure()[0]

    ref = energy_at(0.00125)
    errs = [abs(energy_at(dt) - ref) for dt in (0.05, 0.025, 0.0125)]
    ratios = [errs[i] / errs[i + 1] for i in range(2)]
    ok = all(10.0 <= r <= 22.0 for r in ratios)
    return ok, f"E(T) error ratios per dt halving {[round(r, 1) for r in ratios]} (expect 16)"


# ---------------------------------------------------------------------------
# particles
# ---------------------------------------------------------------------------

def _kernel_normalization() -> Tuple[bool, str]:
    r = np.linspace(-1.5, 1.5, 60001)
    i1 = float(np.trapezoid(cubic_spline(np.abs(r), 1.0, dim=1), r))
    r3 = np.linspace(0.0, 1.0, 20001)
    i3 = float(np.trapezoid(4.0 * np.pi * r3**2 * cubic_spline(r3, 1.0, dim=3), r3))
    ok = abs(i1 - 1.0) < 1e-4 and abs(i3 - 1.0) < 1e-4
    return ok, f"int W dV: 1D {i1:.6f}, 3D {i3:.6f}"


def _density_paths() -> Tuple[bool, str]:
    rng = np.random.default_rng(12)
    x1 = np.sort(rng.uniform(-1.0, 1.0, 120))[:, None]
    m1 = rng.uniform(0.5, 1.5, 120)
    h1 = rng.uniform(0.05, 0.2, 120)
    x3 = rng.uniform(0.0, 1.0, (150, 3))
    m3 = rng.uniform(0.5, 1.5, 150)
    saved = _kmod._DENSE_MAX_ELEMENTS
    try:
        d1 = compute_density(x1, m1, h1)
        d3 = compute_density(x3, m3, 0.3, boxsize=1.0)
        _kmod._DENSE_MAX_ELEMENTS = 0
        t1 = compute_density(x1, m1, h1)
        t3 = compute_density(x3, m3, 0.3, boxsize=1.0)
    finally:
        _kmod._DENSE_MAX_ELEMENTS = saved
    err = max(float(np.max(np.abs(d1 - t1))), float(np.max(np.abs(d3 - t3))))
    return err < 1e-12, f"dense vs cKDTree density: max abs diff {err:.1e}"


def _gravity_paths() -> Tuple[bool, str]:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(90, 3))
    m = rng.uniform(0.1, 2.0, 90)
    saved = _imod._DENSE_MAX_ELEMENTS
    try:
        dense = _imod._direct_gravity(x, m, 1.7, 0.05)
        _imod._DENSE_MAX_ELEMENTS = 0
        loop = _imod._direct_gravity(x, m, 1.7, 0.05)
    finally:
        _imod._DENSE_MAX_ELEMENTS = saved
    err = float(np.max(np.abs(dense - loop)))
    return err < 1e-12, f"dense vs loop gravity: max abs diff {err:.1e}"


def _jacobian_monitor() -> Tuple[bool, str]:
    q = np.linspace(0.0, 1.0, 21)
    j1 = one_d_jacobians(0.4 * q, q)
    x = q.copy()
    x[5], x[6] = x[6], x[5]
    crossed = JacobianTracker(q).min_jacobian(x) < 0.0
    g = np.linspace(0.0, 1.0, 6)
    q3 = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    grads = sph_deformation_gradients(2.0 * q3, q3, h=0.9)
    det = float(np.min(np.linalg.det(grads)))
    ok = np.allclose(j1, 0.4, rtol=1e-12) and crossed and abs(det - 8.0) < 1e-8
    return ok, f"1D scaling J = {j1[0]:.3f}, swap detected {crossed}, 3D det {det:.6f} (expect 8)"


def _collapse(n: int, c, T: float, dt: float = 0.005):
    x = np.linspace(-1.0, 1.0, n)
    sys = LagrangianSystem(
        x[:, None], np.zeros((n, 1)), np.full(n, 1.0 / n), h=0.12, G=1.0,
        c=1.0 if c is None else c, softening=0.4, damping=c is not None, gravity=True,
    )
    tracker = JacobianTracker(sys.initial_positions)
    jmin, vpeak, t_cross = 1.0, 0.0, None
    while sys.time < T - 1e-14:
        sys.step(min(dt, T - sys.time))
        j = tracker.min_jacobian(sys.positions)
        jmin = min(jmin, j)
        vpeak = max(vpeak, float(np.max(np.abs(sys.velocities))))
        if t_cross is None and j <= 0.0:
            t_cross = sys.time
    return jmin, vpeak, t_cross


def _undamped_crosses() -> Tuple[bool, str]:
    jmin, vpeak, t_cross = _collapse(64, None, 2.0)
    return jmin <= 0.0, f"min J {jmin:.3f}, crossing at t = {t_cross}, v_ff {vpeak:.3f}"


def _damped_speed_cap() -> Tuple[bool, str]:
    c = 0.01
    jmin, vpeak, t_cross = _collapse(64, c, 3.0)
    ratio = vpeak / c
    return 0.5 < ratio < 3.0, f"c = {c}: peak |v| / c = {ratio:.2f} (cubic damping caps speed near c)"


def _physical_regime_crosses() -> Tuple[bool, str]:
    jmin, vpeak, t_cross = _collapse(64, 2.0, 2.0)
    return jmin <= 0.0, f"c = 2 (v_ff/c = {vpeak/2.0:.2f} < 1): crossing at t = {t_cross}, min J {jmin:.3f}"


# ---------------------------------------------------------------------------
# fluid
# ---------------------------------------------------------------------------

def _tg_exact_decay() -> Tuple[bool, str]:
    nu, T = 0.01, 2.0
    s = SpectralNS2D(32, nu=nu, lam=0.0, dt_max=0.005)
    s.set_taylor_green()
    worst = 0.0
    while s.time < T - 1e-14:
        s.step(min(s.cfl_dt(), T - s.time))
        e, o, _ = s.measure()
        worst = max(worst, abs(e - 0.25 * np.exp(-4 * nu * s.time)) / 0.25,
                    abs(o - 0.5 * np.exp(-4 * nu * s.time)) / 0.5)
    return worst < 1e-6, f"max rel err vs exp(-4 nu t): {worst:.1e}"


def _divergence_free() -> Tuple[bool, str]:
    s = SpectralNS2D(32, nu=0.005, lam=0.5)
    s.set_taylor_green()
    for _ in range(50):
        s.step(0.005)
    u, v = s.velocity()
    div = np.fft.ifft2(1j * s.kx * np.fft.fft2(u) + 1j * s.ky * np.fft.fft2(v)).real
    m = float(np.abs(div).max())
    return m < 1e-10, f"max |div u| = {m:.1e}"


def _energy_identity() -> Tuple[bool, str]:
    nu, lam, T = 0.005, 0.5, 2.0
    s = SpectralNS2D(32, nu=nu, lam=lam, dt_max=0.005)
    s.set_taylor_green()
    t, e, o, l4 = [0.0], [], [], []
    ei, oi, li = s.measure()
    e.append(ei); o.append(oi); l4.append(li)
    while s.time < T - 1e-14:
        s.step(min(s.cfl_dt(), T - s.time))
        ei, oi, li = s.measure()
        t.append(s.time); e.append(ei); o.append(oi); l4.append(li)
    t, e, o, l4 = map(np.asarray, (t, e, o, l4))
    predicted = e[0] - np.trapezoid(2.0 * nu * o + lam * l4**4, t)
    res = float(e[-1] - predicted)
    return abs(res) < 1e-5, f"E(T) - [E0 - int(2 nu Omega + lam ||u||_4^4)] = {res:+.1e} (every-step trapezoid)"


def _cubic_dealiasing() -> Tuple[bool, str]:
    n, k = 48, 15
    amps = {}
    for cd in (False, True):
        s = SpectralNS2D(n, nu=0.0, lam=1.0, cubic_dealias=cd)
        s.set_field(np.cos(k * s.y), 0.0)
        rhs_u, _ = s._rhs(s.u_hat, s.v_hat)
        amps[cd] = abs(rhs_u[3, 0]) / n**2
    ok = amps[True] < 1e-10 and abs(amps[False] - 0.125) < 1e-6
    return ok, f"aliased |k|=3 amplitude: 2/3 rule {amps[False]:.4f}, padded {amps[True]:.1e}"


def _damping_dt_limit() -> Tuple[bool, str]:
    s = SpectralNS2D(32, nu=1e-3, lam=200.0, dt_max=1.0)
    s.set_taylor_green()
    lim = s.dt_limits(0.5)
    expect = 0.5 * 2.0 / (3.0 * 200.0 * 1.0)
    ok = abs(lim["damping"] - expect) / expect < 1e-6 and s.cfl_dt(0.5) <= lim["damping"]
    return ok, f"dt_damp = {lim['damping']:.2e} (2 cfl / 3 lam |u|^2 = {expect:.2e}), cfl_dt = {s.cfl_dt(0.5):.2e}"


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

_CHECKS: List[Tuple[str, str, CheckFn]] = [
    ("damping", "exact solution matches closed form", _exact_closed_form),
    ("damping", "exact flow composes: phi_a o phi_b = phi_(a+b)", _flow_composition),
    ("damping", "particle integrator: v(T) = v0/sqrt(1+2 lam v0^2 T) for any dt", _integrator_horizon),
    ("damping", "explicit Euler is first order", _euler_order),
    ("convergence", "KDK gravity integrator is second order (two-body)", _kdk_second_order),
    ("convergence", "spectral RK4 is fourth order (damped Taylor-Green)", _rk4_fourth_order),
    ("particles", "cubic spline kernels are normalized (1D, 3D)", _kernel_normalization),
    ("particles", "dense and cKDTree density estimates agree", _density_paths),
    ("particles", "dense and loop gravity agree", _gravity_paths),
    ("particles", "Jacobian monitor: linear maps, crossing detection", _jacobian_monitor),
    ("particles", "undamped 1D cold collapse shell-crosses (J <= 0)", _undamped_crosses),
    ("particles", "damped collapse (c = 0.01): speed capped near c", _damped_speed_cap),
    ("particles", "physical regime (v_ff < c): damping does not prevent crossing", _physical_regime_crosses),
    ("fluid", "Taylor-Green viscous eigenmode decays exactly", _tg_exact_decay),
    ("fluid", "flow stays divergence-free with damping", _divergence_free),
    ("fluid", "energy identity dE/dt = -2 nu Omega - lam ||u||_4^4", _energy_identity),
    ("fluid", "cubic damping term dealiased (2/3 rule alone is not enough)", _cubic_dealiasing),
    ("fluid", "cfl_dt includes the cubic damping stiffness limit", _damping_dt_limit),
]


def run_checks() -> List[Result]:
    results: List[Result] = []
    for c in D.check_formulas():
        results.append(Result("formulas", c.name, c.status, c.detail, 0.0))
    for group, name, fn in _CHECKS:
        t1 = time.perf_counter()
        try:
            passed, detail = fn()
            status = "PASS" if passed else "FAIL"
        except Exception as exc:  # pragma: no cover - defensive
            status, detail = "ERROR", f"{type(exc).__name__}: {exc}"
        results.append(Result(group, name, status, detail, time.perf_counter() - t1))
    return results


def has_failures(results: List[Result], strict: bool = False) -> bool:
    for r in results:
        if r.status in ("FAIL", "ERROR"):
            return True
        if strict and r.status in ("XFAIL", "XPASS"):
            return True
    return False


def format_table(results: List[Result]) -> str:
    lines = []
    width = max(len(r.name) for r in results)
    group = None
    for r in results:
        if r.group != group:
            group = r.group
            lines.append(f"[{group}]")
        secs = f"{r.seconds:5.1f}s" if r.seconds > 0.05 else "      "
        lines.append(f"  {r.status:5s} {secs} {r.name:<{width}}  {r.detail}")
    n_pass = sum(r.status == "PASS" for r in results)
    n_fail = sum(r.status in ("FAIL", "ERROR") for r in results)
    n_x = sum(r.status in ("XFAIL", "XPASS") for r in results)
    lines.append("")
    lines.append(f"{n_pass} passed, {n_fail} failed, {n_x} known issue(s) (XFAIL)")
    return "\n".join(lines)
