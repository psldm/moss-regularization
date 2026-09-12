"""Tests for the Lagrangian particle module: kernels, density, integrator
and the shell-crossing monitor (J > 0 while the Lagrangian map is invertible)."""

import numpy as np
import pytest

from moss_reg.particles.integrator import LagrangianSystem
from moss_reg.particles.jacobian import (
    JacobianTracker,
    one_d_jacobians,
    sph_deformation_gradients,
)
from moss_reg.particles.kernel import (
    adaptive_smoothing_length,
    compute_density,
    cubic_spline,
    cubic_spline_gradient,
    cubic_spline_grad_vector,
)

trapz = getattr(np, "trapezoid", None) or np.trapz

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def test_kernel_support_and_known_values():
    assert cubic_spline(0.0, 1.0, dim=1) == pytest.approx(4.0 / 3.0)
    assert cubic_spline(0.0, 1.0, dim=3) == pytest.approx(8.0 / np.pi)
    assert cubic_spline(0.5, 1.0, dim=1) == pytest.approx(1.0 / 3.0)
    assert cubic_spline(1.0, 1.0, dim=1) == 0.0
    assert cubic_spline(1.0, 1.0, dim=3) == 0.0
    assert cubic_spline(2.0, 0.5, dim=1) == 0.0
    # continuity at the u = 0.5 knot
    assert cubic_spline(0.5, 1.0, dim=3) == pytest.approx(8.0 / np.pi * 0.25)


def test_kernel_normalization_1d():
    h = 1.0
    r = np.linspace(-1.5 * h, 1.5 * h, 60001)
    integral = trapz(cubic_spline(np.abs(r), h, dim=1), r)
    assert integral == pytest.approx(1.0, rel=1e-4)


def test_kernel_normalization_3d():
    h = 1.0
    r = np.linspace(0.0, h, 20001)
    integral = trapz(4.0 * np.pi * r**2 * cubic_spline(r, h, dim=3), r)
    assert integral == pytest.approx(1.0, rel=1e-4)


@pytest.mark.parametrize("dim", [1, 3])
@pytest.mark.parametrize("r", [0.1, 0.3, 0.7, 0.9])
def test_kernel_gradient_matches_finite_difference(dim, r):
    h = 1.0
    delta = 1e-6
    fd = (cubic_spline(r + delta, h, dim) - cubic_spline(r - delta, h, dim)) / (2 * delta)
    assert cubic_spline_gradient(r, h, dim) == pytest.approx(fd, rel=1e-5)


def test_gradient_vector_3d_points_radially_inward():
    h = 1.0
    rng = np.random.default_rng(0)
    dr = rng.normal(size=(5, 3))
    g = cubic_spline_grad_vector(dr, h, dim=3)
    r = np.linalg.norm(dr, axis=1)
    for i in range(5):
        if r[i] < h and r[i] > 0:
            assert np.dot(g[i], dr[i]) < 0  # W decreases with r
            assert np.linalg.norm(g[i]) == pytest.approx(
                abs(cubic_spline_gradient(r[i], h, 3)), rel=1e-12
            )


def test_kernel_rejects_bad_dimension():
    with pytest.raises(ValueError):
        cubic_spline(0.1, 1.0, dim=2)


# ---------------------------------------------------------------------------
# Density estimation
# ---------------------------------------------------------------------------


def test_compute_density_uniform_1d():
    n = 101
    x = np.linspace(-1.0, 1.0, n)
    m = np.full(n, 1.0 / n)
    h = 0.08
    rho = compute_density(x[:, None], m, h)
    spacing = x[1] - x[0]
    interior = rho[n // 4 : 3 * n // 4]
    assert np.allclose(interior, m[0] / spacing, rtol=0.02)
    total_mass = trapz(rho, x)
    assert total_mass == pytest.approx(1.0, rel=0.05)


def test_compute_density_uniform_3d():
    g = np.linspace(0.0, 1.0, 16, endpoint=False)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    n = len(grid)
    m = np.full(n, 1.0 / n)
    rho = compute_density(grid, m, h=0.35, boxsize=1.0)
    spacing = 1.0 / 16
    assert np.allclose(rho, m[0] / spacing**3, rtol=1e-3)
    dV = spacing**3
    assert np.sum(rho * dV) == pytest.approx(1.0, rel=1e-3)  # mass conservation


def test_compute_density_accepts_per_particle_h():
    n = 50
    x = np.linspace(-1.0, 1.0, n)
    m = np.full(n, 1.0 / n)
    h = np.linspace(0.05, 0.12, n)
    rho = compute_density(x[:, None], m, h)
    assert rho.shape == (n,)
    assert np.all(rho > 0.0)


def test_adaptive_smoothing_length_hits_target_count():
    n = 64
    x = np.linspace(-1.0, 1.0, n)
    target = 8
    h = adaptive_smoothing_length(x[:, None], h0=0.05, n_ngb=target, n_iter=4)
    from scipy.spatial import cKDTree

    tree = cKDTree(x[:, None])
    counts = np.array([len(tree.query_ball_point(x[i], h[i])) - 1 for i in range(n)])
    assert np.all((counts >= 0.5 * target) & (counts <= 2.0 * target))


# ---------------------------------------------------------------------------
# Jacobian tracker
# ---------------------------------------------------------------------------


def test_one_d_jacobian_of_uniform_scaling():
    q = np.linspace(0.0, 1.0, 21)
    x = 0.4 * q
    J = one_d_jacobians(x, q)
    assert np.allclose(J, 0.4, rtol=1e-12)


def test_one_d_jacobian_detects_shell_crossing():
    q = np.linspace(0.0, 1.0, 11)
    x = q.copy()
    tracker = JacobianTracker(q)
    assert tracker.min_jacobian(x) == pytest.approx(1.0)
    x[5], x[6] = x[6], x[5]  # swap two neighbours -> ordering violated
    assert tracker.min_jacobian(x) < 0.0


def test_3d_deformation_gradient_of_linear_map():
    g = np.linspace(0.0, 1.0, 6)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    x = 2.0 * q  # spacing doubles to 0.4 -> h must exceed neighbour spacing
    grads = sph_deformation_gradients(x, q, h=0.9)
    assert np.allclose(grads, 2.0 * np.eye(3), atol=1e-9)
    tracker = JacobianTracker(q, h=0.9)
    assert tracker.min_jacobian(x) == pytest.approx(8.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Integrator: damping-only dynamics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", [1.0, 0.5, 0.1, 0.01])
def test_single_particle_damping_matches_closed_form(dt):
    """Regression for the 0.1.0 double-damping bug.

    lambda = G rho / c^3 with rho = m W(0, h).  The integrator applies two
    exact substeps of dt/2 per step, which compose to exactly dt of
    physical damping, so after a horizon T the closed form

        v(T) = v0 / sqrt(1 + 2 lambda |v0|^2 T)

    must hold to round-off for *any* dt (0.1.0 produced the
    1 / sqrt(1 + 4 lambda T) of a doubly-damped run instead).
    """
    sys = LagrangianSystem(
        positions=np.array([[0.0]]),
        velocities=np.array([[1.0]]),
        masses=np.array([1.0]),
        h=0.5,
        G=1.0,
        c=1.0,
        damping=True,
        gravity=False,
    )
    rho = 4.0 / (3.0 * 0.5)  # m * W(0, h) in 1D
    lam = rho  # G = c = 1
    assert sys.densities[0] == pytest.approx(rho, rel=1e-12)
    T = 1.0
    while sys.time < T - 1e-12:
        sys.step(min(dt, T - sys.time))
    assert sys.time == pytest.approx(T, rel=1e-12)
    expected = 1.0 / np.sqrt(1.0 + 2.0 * lam * T)
    wrong_0_1_0 = 1.0 / np.sqrt(1.0 + 4.0 * lam * T)
    assert sys.velocities[0, 0] == pytest.approx(expected, rel=1e-12)
    assert abs(sys.velocities[0, 0] - wrong_0_1_0) > 0.1 * expected


def test_single_step_damps_over_exactly_dt():
    """One step of length dt equals the exact damping flow over dt."""
    from moss_reg.core.analytical import exact_damping

    v0 = np.array([[0.3, -1.2, 0.7]])
    sys = LagrangianSystem(
        positions=np.zeros((1, 3)),
        velocities=v0.copy(),
        masses=np.array([2.0]),
        h=0.7,
        G=1.0,
        c=1.3,
        damping=True,
        gravity=False,
    )
    lam = sys.damping_coefficients()
    dt = 0.37
    sys.step(dt)
    np.testing.assert_allclose(sys.velocities, exact_damping(v0, dt, lam), rtol=1e-13)


def test_density_dense_path_matches_tree_path(monkeypatch):
    from moss_reg.particles import kernel as kmod

    rng = np.random.default_rng(12)
    x1 = np.sort(rng.uniform(-1.0, 1.0, 120))[:, None]
    m1 = rng.uniform(0.5, 1.5, 120)
    h1 = rng.uniform(0.05, 0.2, 120)
    x3 = rng.uniform(0.0, 1.0, (150, 3))
    m3 = rng.uniform(0.5, 1.5, 150)

    dense_1d = compute_density(x1, m1, h1)
    dense_3d = compute_density(x3, m3, 0.3, boxsize=1.0)
    monkeypatch.setattr(kmod, "_DENSE_MAX_ELEMENTS", 0)  # force the cKDTree loop
    tree_1d = compute_density(x1, m1, h1)
    tree_3d = compute_density(x3, m3, 0.3, boxsize=1.0)
    np.testing.assert_allclose(dense_1d, tree_1d, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(dense_3d, tree_3d, rtol=1e-12, atol=1e-14)


def test_gravity_dense_path_matches_loop(monkeypatch):
    from moss_reg.particles import integrator as imod

    rng = np.random.default_rng(3)
    x = rng.normal(size=(90, 3))
    m = rng.uniform(0.1, 2.0, 90)
    dense = imod._direct_gravity(x, m, 1.7, 0.05)
    monkeypatch.setattr(imod, "_DENSE_MAX_ELEMENTS", 0)
    loop = imod._direct_gravity(x, m, 1.7, 0.05)
    np.testing.assert_allclose(dense, loop, rtol=1e-12, atol=1e-14)


def test_damping_disabled_leaves_velocity_unchanged():
    sys = LagrangianSystem(
        positions=np.array([[0.0]]),
        velocities=np.array([[3.0]]),
        masses=np.array([1.0]),
        h=0.5,
        damping=False,
        gravity=False,
    )
    for _ in range(10):
        sys.step(0.1)
    assert sys.velocities[0, 0] == pytest.approx(3.0, rel=1e-14)
    assert sys.time == pytest.approx(1.0, rel=1e-14)
    assert sys.n_steps == 10


def test_adaptive_dt_obeys_limits():
    sys = LagrangianSystem(
        positions=np.linspace(-1.0, 1.0, 8)[:, None],
        velocities=np.full((8, 1), 100.0),
        masses=np.full(8, 1.0),
        h=0.4,
        c=1.0,
        damping=True,
        gravity=True,
    )
    dt_max = 1e-3
    dt = sys.adaptive_dt(dt_max=dt_max, eta=0.2, cfl=0.5, damp_safety=0.5)
    assert 0.0 < dt <= dt_max
    lam = sys.damping_coefficients()
    speed2 = 100.0**2
    dt_damp = 0.5 * 2.0 / np.max(lam * speed2)
    assert dt <= dt_damp


# ---------------------------------------------------------------------------
# 1D cold collapse: shell crossing without damping, J > 0 with damping
# ---------------------------------------------------------------------------


def build_collapse(c=None, n=64, h=0.12, softening=0.4):
    x = np.linspace(-1.0, 1.0, n)
    m = np.full(n, 1.0 / n)
    sys = LagrangianSystem(
        positions=x[:, None],
        velocities=np.zeros((n, 1)),
        masses=m,
        h=h,
        G=1.0,
        c=1.0 if c is None else c,
        softening=softening,
        damping=c is not None,
        gravity=True,
    )
    tracker = JacobianTracker(sys.initial_positions)
    return sys, tracker


def run_collapse(sys, tracker, T, dt=0.005):
    """min J history and crossing time (None if no crossing by T)."""
    history, t_cross = [], None
    while sys.time < T - 1e-14:
        sys.step(min(dt, T - sys.time))
        history.append(tracker.min_jacobian(sys.positions))
        if t_cross is None and history[-1] <= 0.0:
            t_cross = sys.time
    return np.array(history), t_cross


def test_cold_collapse_crosses_without_damping():
    sys, tracker = build_collapse(c=None)
    assert tracker.min_jacobian(sys.positions) == pytest.approx(1.0, rel=1e-12)
    jmin, t_cross = run_collapse(sys, tracker, T=2.0)
    assert np.all(np.isfinite(jmin))
    assert np.min(jmin) <= 0.0 and t_cross is not None  # shell crossing: J -> 0 and below


def test_cold_collapse_damping_caps_speed_and_delays_crossing():
    """Strong-coupling regime, c = 0.01 (free-fall speed ~170 c).

    The cubic damping acts as a speed limiter near c: the damped run
    crosses later than the classical one and its peak speed is O(c).
    It does *not* prevent the crossing at this resolution (N = 128).
    """
    c, T = 0.01, 6.0
    sys0, tr0 = build_collapse(c=None, n=128)
    _, t_cross0 = run_collapse(sys0, tr0, T=T)
    sys1, tr1 = build_collapse(c=c, n=128)
    jmin, t_cross1 = run_collapse(sys1, tr1, T=T)
    assert np.all(np.isfinite(jmin)) and np.all(sys1.densities > 0.0)
    assert t_cross0 is not None and t_cross1 is not None
    assert t_cross1 > 2.0 * t_cross0
    vmax = np.max(np.abs(sys1.velocities))
    assert vmax < 2.0 * c


def test_cold_collapse_arrest_is_not_resolution_robust():
    """At c = 0.01 the 0.1.0 'arrest' (no crossing by t = 6 at N = 64) is a
    resolution artifact: the crossing time decreases monotonically with N
    because the delay scales like (particle spacing) / c."""
    c, T = 0.01, 6.0
    t_cross = []
    for n in (64, 128, 200):
        sys, tr = build_collapse(c=c, n=n)
        _, tx = run_collapse(sys, tr, T=T)
        t_cross.append(np.inf if tx is None else tx)
    assert t_cross[0] > t_cross[1] > t_cross[2]
    assert np.isfinite(t_cross[2])


def test_cold_collapse_crosses_when_v_below_c():
    """Physical regime (v_ff < c): the damping does NOT arrest shell crossing.

    With c = 2 code units the free-fall speed (~1.7) stays below c, the
    damping rate G rho v^2 / c^3 is small compared with the dynamical
    rate, and the map crosses essentially as in the undamped run.
    """
    sys0, tr0 = build_collapse(c=None)
    _, t_cross0 = run_collapse(sys0, tr0, T=2.0)
    sys, tracker = build_collapse(c=2.0)
    jmin, t_cross = run_collapse(sys, tracker, T=2.0)
    assert np.all(np.isfinite(jmin))
    assert np.min(jmin) <= 0.0 and t_cross is not None
    assert abs(t_cross - t_cross0) < 0.1
