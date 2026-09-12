# Changelog

## 0.4.0 (2026-09-12)

### Added
- Operator-split exact damping for both spectral solvers
  (`damping_mode="split"`): Strang splitting with the exact substep on the
  padded grid, no timestep limit from the damping, exactly accumulated
  `energy_to_vacuum` and a separately reported truncation/projection loss;
  first order in dt (the pointwise substep does not commute with the
  projection), converges to the `"rhs"` mode as dt (tested).
- `moss-reg sweep3d`: 3D Taylor-Green sweep over resolution (classical at
  32^3, 64^3, 96^3) and damping number (damped, split mode, lambda_code =
  0.05 ... 50 at 64^3 and up to 200 at 32^3), with the regularity threshold
  4 lambda / Re = 1 marked; figure 06 and per-run CSV. Result: at Re = 800
  no resolution up to 96^3 stays spectrally resolved to t = 6; the damping
  changes the classical curves only for lambda_code >= 0.05 and extinguishes
  the flow for lambda_code >= 5, so the theorem's threshold (200) lies in
  the extinguished regime.
- `EnergyBudget` accepts exact E_vac and extra-loss series.
- PyPI publication workflow (Trusted Publishing on GitHub release), PyTorch
  adapter CI job, `CONTRIBUTING.md`, issue templates.

## 0.3.0 (2026-09-12)

### Added
- `moss_reg.diagnostics`: `field_norms` (L2, H1 seminorm, enstrophy, L4 on a
  padded grid, L-infinity, divergence, any dimension), `EnergyBudget`
  (E(t) = E0 - D_visc - E_vac with every-step quadrature), `DiagnosticsLog`
  (per-step recorder, CSV/JSON). Solvers expose `diagnostics()`; the particle
  integrator accumulates the energy handed to the damping exactly
  (`energy_to_vacuum`) so that E_kin + E_pot + E_vac is a checked invariant.
- 3D spectral solver `SpectralNS3D` (rotational form, projection, RK4, 2/3
  rule, padded cubic term) with the 3D Taylor-Green benchmark
  `moss-reg compare --type fluid3d` (figure 05, CSV time series, Kolmogorov
  resolution indicator k_max eta). `run-all` includes it (`--no-3d` to skip).
- Spatially varying coupling `lam(x)` for both spectral solvers
  (band-limited field, evaluated on the padded grid) and
  `moss_reg.fluid.coupling`: `damping_number`, `lambda_eff_si`,
  `physical_damping_number`, `lambda_field_from_density`.
- General exponent: `exact_damping_alpha` (alpha >= 0; alpha = 2 is cubic).
- Adapters (`integrations/`): header-only C99/C++ `moss_damp.h` with a
  self-test compiled in CI; C# `MossDamp.cs` with a .NET 8 self-test in CI;
  `moss_reg.integrations.damp` (NumPy) and `torch_ops.damp` (PyTorch,
  CPU/GPU, autograd); Dedalus v3 and OpenFOAM `fvOptions` source-term
  examples. All are cross-checked against the Python reference.
- `moss-reg validate` grew to 29 checks (+ 1 XFAIL): 3D shear-mode decay,
  3D energy budget and divergence, 3D dealiasing, particle energy invariant.
- Benchmarks write every-step diagnostics as CSV next to the figures and
  report the energy budget / invariant drift as checks.

### Changed
- `examples/raptor_shear.py` reworded as an under-resolved numerical test
  with a tuned damping number (no "SpaceX", no "CRASHED vs STABLE").
- Package author and project URLs set in `pyproject.toml`.

## 0.2.1 (2026-09-11)

### Added
- `moss-reg supplement`: LaTeX supplementary material generated entirely from
  `report.json` (provenance, full validation table, every parameter and
  metric, complete sweep grid and sensitivity tables, figures), so the
  paper's numerical appendix cannot drift from the code.
- `run-all` stores the full validation results (group, name, status, detail,
  timing) in the report, not only the statuses.

### Fixed
- Removed references to a "Theorem D.1" from docstrings, tests and README:
  no such theorem exists in the accompanying paper or its technical
  appendix; the label was introduced by the 0.1.0 documentation.

## 0.2.0 (2026-09-11)

### Fixed
- **Particle integrator damped twice as strongly as the stated equation.**
  `LagrangianSystem.step` applied two exact damping substeps of length `dt`
  per step of physical length `dt` (equivalent to `1/sqrt(1 + 4 lambda v0^2 T)`
  instead of `1/sqrt(1 + 2 lambda v0^2 T)`), and the unit test codified the
  wrong value. The scheme is now the symmetric kick/2 - damp/2 - drift -
  damp/2 - kick/2 splitting; a damping-only run reproduces the closed form to
  round-off for any `dt` (regression test + `moss-reg validate`).
- README identity `lambda = 1/(8 pi Xi c)` was false: `1/(8 pi Xi c) = G/c^5 =
  lambda / c^2`. `Xi = c^4/(8 pi G)` has the dimension of a force, not a
  pressure (comment in `constants.py` corrected).
- `tests/test_constants.py` asserted `lambda_c5 == lambda * c^2` (wrong by
  `c^4`) and passed because `pytest.approx` accepts any two numbers below its
  default absolute tolerance; the comparisons now use `abs=0`.
- The 2/3 dealiasing rule does not dealias the cubic damping term `|u|^2 u`;
  it is now evaluated on a zero-padded `2n` grid (exact). The `L^4` norm is
  computed on the padded grid as well. `cfl_dt` now includes the cubic
  damping stiffness limit `dt <= cfl * 2 / (3 lambda |u|_max^2)`.
- Energy-identity checks recorded no `t = 0` sample and used coarse time
  sampling; they now use every-step trapezoid quadrature (residual ~1e-7).

### Changed
- **Shell-crossing claims.** The 0.1.0 figure used `c = 0.005` code units, a
  collapse that is superluminal in the model's own units (free-fall speed
  ~340 c). The sweep (`moss-reg sweep`) shows that the cubic damping caps the
  speed near `c`, delays the crossing by about (particle spacing)/`c`, and
  that the delay shrinks with resolution; in the physical regime `v_ff < c`
  the crossing is unchanged. The comparison benchmark now defaults to
  compactness `GM/(Lc^2) = 0.1` (`c ~ 3.2`) and reports the regime.
- Dense `(N, N)` vectorized gravity and SPH density for small systems
  (same results as the loop / cKDTree paths, ~10x faster).
- Figures use a fixed colour-vision-safe palette (classical = blue,
  moss = orange).

### Added
- `moss_reg.dimensions`: exact (M, L, T) dimensional bookkeeping and
  `check_formulas()`; documents the open issue that `G rho / c^3` has
  dimension `s m^-3` while the ODE requires `s m^-2`.
- `moss-reg validate`: 25 PASS/FAIL checks (formulas, exact damping,
  integrator regression, KDK and RK4 convergence orders, kernels, Jacobian
  monitor, Taylor-Green decay, divergence, energy identity, dealiasing).
- `moss-reg compare --type {particles,fluid}` and `moss-reg sweep`: one
  command each, PNG plus `report.json` with commit SHA, environment,
  parameters, metrics and checks.
- `--compactness` / `--c` parametrization of the collapse benchmark.
- `examples/raptor_shear.py`: periodic double shear layer (Kelvin-Helmholtz)
  at high Reynolds number, classical NS vs. Moss damping.
- `report.json` records the git state captured before any output is written.

## 0.1.0

Initial release.
