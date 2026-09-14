"""Homogeneity checks: the tool that rejects the version-1 inequality."""

from fractions import Fraction

import pytest

from moss_reg.scaling import (
    check_inequality,
    damping_exponent_scaling,
    interpolation_exponent,
    parse_inequality,
)


def test_version1_inequality_is_rejected():
    r = check_inequality("|D1 u|_2^2 <= |u|_4^{4/3} |D2 u|_2^{2/3}", d=3)
    assert r["homogeneous"] is False
    assert r["lhs_dilation"] == "-1" and r["rhs_dilation"] == "-2/3"
    assert "mu -> 0" in r["reason"]


def test_corrected_gagliardo_nirenberg_is_admissible():
    r = check_inequality("|D1 u|_2 <= |u|_4^{4/5} |D2 u|_2^{1/5}", d=3)
    assert r["homogeneous"] is True
    assert interpolation_exponent(1, 2, 2, 2, 4, d=3) == Fraction(1, 5)


def test_classical_inequalities():
    assert check_inequality("|u|_6 <= |D1 u|_2", d=3)["homogeneous"]                 # Sobolev, 3D
    assert check_inequality("|u|_4^4 <= |u|_2^2 |D1 u|_2^2", d=2)["homogeneous"]     # Ladyzhenskaya, 2D
    assert not check_inequality("|u|_4^4 <= |u|_2^2 |D1 u|_2^2", d=3)["homogeneous"] # false in 3D
    assert check_inequality("|u|_4^4 <= |u|_2 |D1 u|_2^3", d=3)["homogeneous"]       # 3D version
    assert check_inequality("|D1 u|_3 <= |D1 u|_2^{1/2} |D2 u|_2^{1/2}", d=3)["homogeneous"]
    assert not check_inequality("|u|_inf <= |u|_2", d=3)["homogeneous"]              # sup by L2: false


def test_amplitude_mismatch_detected():
    r = check_inequality("|u|_2^2 <= |u|_2", d=3)
    assert r["homogeneous"] is False and "amplitude" in r["reason"]


def test_parser_variants_and_errors():
    i = parse_inequality("|u|_4^{4/3} * |D2 u|_2^{2/3} <= C |D1 u|_2^2", d=3)
    assert len(i.lhs) == 2 and len(i.rhs) == 1
    assert str(i.rhs[0]) == "|D1 u|_2^{2}"
    with pytest.raises(ValueError):
        parse_inequality("|u|_2 = |u|_2")
    with pytest.raises(ValueError):
        parse_inequality("|u|_2 <= junk")


def test_damping_exponent_scaling():
    assert damping_exponent_scaling(2)["class"].startswith("critical")
    assert damping_exponent_scaling(3)["class"].startswith("supercritical")
    assert damping_exponent_scaling(1)["class"].startswith("subcritical")
    assert damping_exponent_scaling(2)["exponent"] == "3"
