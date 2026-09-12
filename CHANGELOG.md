# Changelog

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
