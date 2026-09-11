"""Tests for the Lagrangian particle module: kernels, density, integrator
and the Theorem D.1 shell-crossing monitor (J > 0)."""

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


def test_single_particle_damping_matches_closed_form():
    # lambda = G rho / c^3 with rho = m W(0, h); two substeps of dt each
    # per step give v(t) = v0 / sqrt(1 + 4 lambda |v0|^2 t).
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
    dt = 0.01
    while sys.time < T - 1e-12:
        sys.step(dt)
    expected = 1.0 / np.sqrt(1.0 + 4.0 * lam * T)
    assert sys.velocities[0, 0] == pytest.approx(expected, rel=1e-9)


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
    history = []
    while sys.time < T - 1e-14:
        sys.step(dt)
        history.append(tracker.min_jacobian(sys.positions))
    return np.array(history)


def test_cold_collapse_crosses_without_damping():
    sys, tracker = build_collapse(c=None)
    assert tracker.min_jacobian(sys.positions) == pytest.approx(1.0, rel=1e-12)
    jmin = run_collapse(sys, tracker, T=2.0)
    assert np.all(np.isfinite(jmin))
    assert np.min(jmin) <= 0.0  # shell crossing: J -> 0 and below


def test_cold_collapse_arrested_with_vacuum_damping():
    sys, tracker = build_collapse(c=0.01)
    assert tracker.min_jacobian(sys.positions) == pytest.approx(1.0, rel=1e-12)
    jmin = run_collapse(sys, tracker, T=6.0)
    assert np.all(np.isfinite(jmin))
    # Theorem D.1: the damping keeps the flow invertible throughout the
    # run -- the crossing singularity of the undamped collapse is
    # suppressed and the compression stays bounded.
    assert np.min(jmin) > 0.05
    assert np.all(sys.densities > 0.0)
    # bounded deceleration: residual speeds are orders of magnitude
    # below the undamped crossing speed (~1)
    vmax = np.max(np.abs(sys.velocities))
    assert vmax < 0.05
