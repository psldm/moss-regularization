"""Dimensional and numerical consistency of the constants module."""

import math

import numpy as np
import pytest

from moss_reg import constants as K


def test_lambda_definition():
    assert K.LAMBDA == pytest.approx(K.G / K.C**3, rel=1e-15, abs=0.0)


def test_vacuum_shear_modulus():
    assert K.XI == pytest.approx(K.C**4 / (8.0 * math.pi * K.G), rel=1e-15)


def test_lambda_times_planck_force_is_unity():
    # spec identity: lambda = 1 / F_P  =>  lambda * F_P == 1
    assert K.LAMBDA * K.F_P == pytest.approx(1.0, rel=1e-15)
    assert 1.0 / K.F_P == pytest.approx(K.LAMBDA, rel=1e-15, abs=0.0)


def test_lambda_c5_relation():
    # 1 / (8 pi Xi c) == G / c^5 == lambda / c^2  (NOT lambda, NOT lambda c^2).
    # abs=0: pytest.approx's default absolute tolerance (1e-12) would
    # accept *any* pair of these tiny numbers -- the 0.1.0 test asserted
    # lambda * c^2 and passed for exactly that reason.
    assert K.lambda_c5 == pytest.approx(1.0 / (8.0 * math.pi * K.XI * K.C), rel=1e-15, abs=0.0)
    assert K.lambda_c5 == pytest.approx(K.G / K.C**5, rel=1e-15, abs=0.0)
    assert K.lambda_c5 == pytest.approx(K.LAMBDA / K.C**2, rel=1e-15, abs=0.0)
    assert K.lambda_c5 != pytest.approx(K.LAMBDA * K.C**2, rel=1e-6, abs=0.0)
    assert K.lambda_c5 != pytest.approx(K.LAMBDA, rel=1e-6, abs=0.0)


def test_planck_length_time_mass():
    assert K.L_P == pytest.approx(math.sqrt(K.HBAR * K.G / K.C**3), rel=1e-15, abs=0.0)
    assert K.T_P == pytest.approx(K.L_P / K.C, rel=1e-15, abs=0.0)
    assert K.M_P == pytest.approx(math.sqrt(K.HBAR * K.C / K.G), rel=1e-15, abs=0.0)
    assert K.RHO_P == pytest.approx(K.M_P / K.L_P**3, rel=1e-15)


def test_planck_scale_reference_values():
    # CODATA-based reference magnitudes
    assert K.L_P == pytest.approx(1.616255e-35, rel=1e-4, abs=0.0)
    assert K.T_P == pytest.approx(5.391247e-44, rel=1e-4, abs=0.0)
    assert K.M_P == pytest.approx(2.176434e-8, rel=1e-4, abs=0.0)
    assert K.RHO_P == pytest.approx(5.1550e96, rel=1e-4)


def test_standard_planck_force_relation():
    # textbook F_P = c^4 / G = m_P c^2 / l_P
    assert K.F_P_STANDARD == pytest.approx(K.C**4 / K.G, rel=1e-15)
    assert K.F_P_STANDARD == pytest.approx(K.M_P * K.C**2 / K.L_P, rel=1e-15)


def test_dimensions_of_hbar():
    # [hbar] = J s = kg m^2 / s
    assert K.HBAR == pytest.approx(K.M_P * K.C * K.L_P, rel=1e-12, abs=0.0)


def test_unit_system_natural_normalization():
    nat = K.UnitSystem.natural()
    assert nat.c == 1.0
    assert nat.G == 1.0
    assert nat.lambda_ == pytest.approx(1.0, rel=1e-15)
    assert nat.f_p == pytest.approx(1.0, rel=1e-15)
    assert nat.xi == pytest.approx(1.0 / (8.0 * math.pi), rel=1e-15)


def test_si_unit_system_matches_module_constants():
    si = K.UnitSystem()
    assert si.xi == pytest.approx(K.XI, rel=1e-15)
    assert si.lambda_ == pytest.approx(K.LAMBDA, rel=1e-15, abs=0.0)
    assert si.l_p == pytest.approx(K.L_P, rel=1e-15, abs=0.0)
    assert si.rho_p == pytest.approx(K.RHO_P, rel=1e-15)


def test_planck_units_instance():
    assert K.planck_units.lambda_ == pytest.approx(1.0, rel=1e-15)
    assert K.natural_units.lambda_ == pytest.approx(1.0, rel=1e-15)


def test_numpy_scalar_consistency():
    assert np.float64(K.LAMBDA) * np.float64(K.F_P) == pytest.approx(1.0, rel=1e-15)
