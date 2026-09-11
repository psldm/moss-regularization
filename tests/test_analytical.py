"""Analytical damping solution vs. discrete Euler integration."""

import numpy as np
import pytest

from moss_reg import constants as K
from moss_reg.core.analytical import (
    damped_step,
    damping_timescale,
    euler_step,
    exact_damping,
)


@pytest.mark.parametrize("dims", [1, 2, 3])
@pytest.mark.parametrize("lambda_eff", [1.0, 3.7, 100.0])
def test_exact_formula_against_closed_form(dims, lambda_eff):
    rng = np.random.default_rng(42 + dims)
    v0 = rng.normal(size=dims)
    t = 0.7
    expected = v0 / np.sqrt(1.0 + 2.0 * lambda_eff * np.dot(v0, v0) * t)
    np.testing.assert_allclose(
        exact_damping(v0, t, lambda_eff), expected, rtol=1e-14, atol=1e-15
    )


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_direction_preserved(dims):
    rng = np.random.default_rng(7)
    v0 = rng.normal(size=dims)
    if np.allclose(v0, 0.0):
        return
    for t in (1e-3, 1.0, 1e3):
        v = exact_damping(v0, t, lambda_eff=2.0)
        np.testing.assert_allclose(
            v / np.linalg.norm(v), v0 / np.linalg.norm(v0), rtol=1e-12, atol=1e-14
        )


def test_damped_step_matches_exact_solution():
    v0 = np.array([1.0, -2.0, 0.5])
    dt = 0.25
    np.testing.assert_allclose(
        damped_step(v0, dt, lambda_eff=5.0),
        exact_damping(v0, dt, lambda_eff=5.0),
        rtol=1e-15,
    )


def test_damped_step_default_lambda_is_si_constant():
    v0 = np.array([1.0, 0.0])
    v1 = damped_step(v0, dt=1.0)
    assert K.LAMBDA > 0
    np.testing.assert_allclose(
        v1, exact_damping(v0, 1.0, K.LAMBDA), rtol=1e-15
    )


def test_zero_velocity_stays_zero():
    for v0 in (np.zeros(1), np.zeros(2), np.zeros(3)):
        np.testing.assert_allclose(exact_damping(v0, 10.0), v0)
        assert not np.any(np.isnan(exact_damping(v0, 10.0)))


def test_speed_monotonically_decreases():
    rng = np.random.default_rng(11)
    v0 = rng.normal(size=3)
    s0 = np.linalg.norm(v0)
    for t in (0.1, 1.0, 10.0):
        assert np.linalg.norm(exact_damping(v0, t, lambda_eff=1.0)) < s0


def test_euler_converges_to_exact_as_dt_vanishes():
    """Explicit Euler error scales ~ O(dt) toward the exact solution."""
    v0 = np.array([1.0, 0.5, -0.25])
    T = 1.0
    lam = 1.0
    v_exact = exact_damping(v0, T, lam)
    errs = []
    for n in (100, 200, 400, 800):
        dt = T / n
        v = v0.copy()
        for _ in range(n):
            v = euler_step(v, dt, lam)
        errs.append(np.linalg.norm(v - v_exact))
    # halving dt must roughly halve the error (order >= ~0.8)
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    for r in ratios:
        assert 1.7 <= r <= 2.3


def test_vectorized_over_ensemble():
    rng = np.random.default_rng(3)
    v0 = rng.normal(size=(50, 3))
    t = np.linspace(0.0, 2.0, 50)
    v = exact_damping(v0, t, lambda_eff=0.5)
    assert v.shape == (50, 3)
    for i in (0, 17, 49):
        np.testing.assert_allclose(
            v[i],
            exact_damping(v0[i], t[i], lambda_eff=0.5),
            rtol=1e-14,
        )


def test_batch_scalar_time_broadcast():
    rng = np.random.default_rng(5)
    v0 = rng.normal(size=(10, 2))
    v = exact_damping(v0, 1.5, lambda_eff=0.25)
    assert v.shape == (10, 2)


def test_damping_timescale():
    v = np.array([[3.0, 4.0], [0.0, 0.0]])  # |v| = 5 and 0
    tau = damping_timescale(v, lambda_eff=2.0)
    assert tau[0, 0] == pytest.approx(1.0 / (2.0 * 25.0), rel=1e-15)
    assert np.isinf(tau[1, 0])


def test_high_precision_single_step():
    # exact step must beat Euler at finite dt by construction
    v0 = np.array([2.0, 0.0])
    dt, lam = 0.5, 1.0
    exact = exact_damping(v0, dt, lam)
    euler = euler_step(v0, dt, lam)
    assert np.linalg.norm(exact) < np.linalg.norm(v0)
    assert np.linalg.norm(euler) == pytest.approx(np.linalg.norm(v0), rel=1e-15)
    assert not np.allclose(exact, euler)
