"""Adapters: NumPy, PyTorch (optional), C header (if a compiler is present),
C# (if dotnet is present).  All must reproduce the Python reference."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from moss_reg.core.analytical import exact_damping, exact_damping_alpha
from moss_reg.integrations import damp, damp_with_energy

ROOT = Path(__file__).resolve().parents[1]
V0 = np.array([0.3, -1.2, 0.7])
LAM, DT = 3.7, 0.7


def test_numpy_adapter_matches_reference():
    np.testing.assert_allclose(damp(V0, DT, LAM), exact_damping(V0, DT, LAM), rtol=1e-15)
    rng = np.random.default_rng(0)
    v = rng.normal(size=(7, 5, 3))
    lam = rng.uniform(0.1, 2.0, size=(7, 5))
    np.testing.assert_allclose(damp(v, 0.3, lam), exact_damping(v, 0.3, lam), rtol=1e-14)
    out, removed = damp_with_energy(v, 0.3, lam, mass=np.full((7, 5), 2.0))
    e0 = 0.5 * np.sum(2.0 * np.sum(v * v, axis=-1))
    e1 = 0.5 * np.sum(2.0 * np.sum(out * out, axis=-1))
    assert removed == pytest.approx(e0 - e1, rel=1e-12)


@pytest.mark.parametrize("alpha", [1.0, 2.0, 3.0, 4.5])
def test_general_exponent_solves_the_ode(alpha):
    def rhs(t, v):
        return -LAM * np.linalg.norm(v) ** alpha * v
    ref = solve_ivp(rhs, (0.0, DT), V0, rtol=1e-12, atol=1e-14).y[:, -1]
    np.testing.assert_allclose(exact_damping_alpha(V0, DT, LAM, alpha), ref, rtol=1e-9)
    if alpha == 2.0:
        np.testing.assert_allclose(exact_damping_alpha(V0, DT, LAM, 2.0), exact_damping(V0, DT, LAM), rtol=1e-15)


def test_torch_adapter_if_available():
    torch = pytest.importorskip("torch")
    from moss_reg.integrations.torch_ops import damp as tdamp

    v = torch.tensor(np.tile(V0, (4, 1)), dtype=torch.float64, requires_grad=True)
    out = tdamp(v, DT, LAM)
    np.testing.assert_allclose(out.detach().numpy(), np.tile(exact_damping(V0, DT, LAM), (4, 1)), rtol=1e-14)
    out.sum().backward()
    assert torch.isfinite(v.grad).all()
    out3 = tdamp(v.detach(), DT, LAM, alpha=3.0)
    np.testing.assert_allclose(out3.numpy(), np.tile(exact_damping_alpha(V0, DT, LAM, 3.0), (4, 1)), rtol=1e-12)


def _parse_vec(line: str) -> np.ndarray:
    return np.array([float(x) for x in line.split("=", 1)[1].split()])


def test_c_header_matches_reference(tmp_path):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler")
    exe = tmp_path / "test_moss_damp"
    src = ROOT / "integrations" / "c" / "test_moss_damp.c"
    subprocess.run([cc, "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2", "-o", str(exe), str(src), "-lm"], check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    lines = {l.split("=", 1)[0]: l for l in out.strip().splitlines() if "=" in l}
    assert out.strip().endswith("OK")
    np.testing.assert_allclose(_parse_vec(lines["v_out"]), exact_damping(V0, DT, LAM), rtol=1e-15)
    np.testing.assert_allclose(_parse_vec(lines["v_out_alpha3"]), exact_damping_alpha(V0, DT, LAM, 3.0), rtol=1e-14)
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if cxx is not None:
        subprocess.run([cxx, "-std=c++11", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", "-x", "c++",
                        str(ROOT / "integrations" / "c" / "moss_damp.h")], check=True)


def test_csharp_matches_reference_if_dotnet_available():
    if shutil.which("dotnet") is None:
        pytest.skip("no dotnet SDK")
    proj = ROOT / "integrations" / "csharp" / "MossDamp.Test"
    out = subprocess.run(["dotnet", "run", "--project", str(proj), "-c", "Release", "--nologo"],
                         capture_output=True, text=True, check=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("v_out="))
    np.testing.assert_allclose(_parse_vec(line), exact_damping(V0, DT, LAM), rtol=1e-15)
    assert out.strip().endswith("OK")
