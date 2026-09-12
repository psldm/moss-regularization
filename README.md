[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22716096.svg)](https://doi.org/10.5281/zenodo.22716096)

# moss-regularization

**A research library and validation harness for the cubic "vacuum damping"
term in CFD and N-body simulations.**

`moss-reg` implements the nonlinear damping

$$
\frac{d\mathbf{v}}{dt} = -\lambda_{\mathrm{eff}}\,|\mathbf{v}|^2\,\mathbf{v},
\qquad
\lambda_{\mathrm{eff}} = \lambda\,\rho, \qquad \lambda = \frac{G}{c^3},
$$

inside an exact operator-splitting particle integrator and a 2D spectral
Navier–Stokes solver, together with the tooling needed to check what the
term actually does: `moss-reg validate` (PASS/FAIL self-checks),
`moss-reg compare` (classical vs. moss, one command, PNG + `report.json`)
and `moss-reg sweep` (resolution / timestep / kernel / softening /
speed-of-light sweep of the shell-crossing benchmark).

## What this repository shows, and what it does not

**Shown (and checked on every CI run):**

* the exact solution of the damping ODE and its use as an unconditionally
  stable substep; the exact flow composes, $\phi_a\circ\phi_b=\phi_{a+b}$;
* a symmetric kick/2–damp/2–drift–damp/2–kick/2 particle integrator that
  damps over exactly $dt$ per step (second order in the gravity part);
* a 2D spectral solver for the damped Navier–Stokes equations with the
  cubic term properly dealiased, verified against the exact Taylor–Green
  decay and the energy identity to $\sim 10^{-7}$;
* the phenomenology of the cubic term in a 1D cold collapse: it acts as a
  **speed limiter near $c$**, delays shell crossing by roughly (particle
  spacing)/$c$, and this delay **shrinks with resolution**; in the
  physically admissible regime $v_{\mathrm{ff}} < c$ the collapse is
  unchanged (figure `02_shell_crossing_sweep.png`).

**Not shown:**

* prevention of finite-time singularities of Navier–Stokes. The fluid
  benchmark is two-dimensional, and 2D Navier–Stokes has no finite-time
  blow-up with or without the damping term; the benchmark tests the
  solver and the energy budget, not regularity;
* arrest of shell crossing at physical parameters. The 0.1.0 figure used
  $c = 0.005$ code units, a collapse that is superluminal in the model's
  own units (free-fall speed $\approx 300\,c$); its "arrest" was a
  finite-resolution, finite-horizon effect (see the sweep) and, in
  addition, release 0.1.0 damped twice as strongly as the stated equation
  (see [CHANGELOG.md](CHANGELOG.md)).

## The damping term

The substep ODE has the closed-form solution

$$
\mathbf{v}(t) = \frac{\mathbf{v}_0}{\sqrt{1 + 2\,\lambda_{\mathrm{eff}}\,|\mathbf{v}_0|^2\,t}},
\qquad
\hat{\mathbf{v}}(t) = \hat{\mathbf{v}}_0 ,
$$

so the direction is preserved and $|\mathbf{v}| \sim t^{-1/2}$. The
substep is integrated exactly in the particle integrator. In the spectral
fluid solver the term is part of the explicit RK4 right-hand side, so
there it *does* carry a stability limit,
$dt \le \mathrm{CFL}\cdot 2/(3\lambda_{\mathrm{eff}}|u|_{\max}^2)$, which
`cfl_dt()` enforces.

### Constants and dimensions

In SI, $\lambda = G/c^3$ has dimension $\mathrm{s\,kg^{-1}}$ and
$\lambda = 1/F_P$ with $F_P = c^3/G$ (kg/s). Two statements of earlier
drafts are wrong and are now tested against (`tests/test_dimensions.py`,
`moss-reg validate`):

* $\Xi = c^4/(8\pi G)$ is a **force** ($\mathrm{kg\,m\,s^{-2}}$, the Planck
  force over $8\pi$), not a pressure or energy density. A "vacuum shear
  modulus" in Pa needs an additional area, which the theory does not yet
  specify.
* $1/(8\pi\,\Xi\,c) = G/c^5 = \lambda/c^2 \neq \lambda$.

**Open issue.** The ODE requires $[\lambda_{\mathrm{eff}}] =
\mathrm{s\,m^{-2}}$, whereas $G\rho/c^3$ has dimension
$\mathrm{s\,m^{-3}}$: the SI form of $\lambda_{\mathrm{eff}} = \lambda\rho$
is missing one power of length. All numerical work in this repository is
done in code units ($G = 1$, mass and length units $= 1$), where this is
invisible, but the physical derivation of the coefficient has to supply
that length (a kernel scale $h$, a column density $\rho\ell$, or a
different coupling) before any SI value can be quoted. This is reported as
`XFAIL` by `moss-reg validate` and by the test-suite.

### Parameters

The fundamental coefficient is not fitted. The *numerical models* do have
parameters, and they are stated explicitly rather than hidden:

| parameter | where | meaning |
| --- | --- | --- |
| $N$, $dt_{\max}$, $h$, softening | particles | resolution, timestep, SPH kernel scale, Plummer softening |
| $c$ (code units) or compactness $GM/(Lc^2) = 1/c^2$ | particles | the one physical dial of the collapse; physical bodies have compactness $\le 0.5$ |
| $N$, $dt_{\max}$, $Re$ | fluid | resolution, timestep, Reynolds number |
| $\lambda_{\mathrm{code}} = \lambda_{\mathrm{eff}} U L$ | fluid | dimensionless damping number; the benchmark uses $0.5$ to make the effect visible, which is many orders of magnitude above any physical value |

Every `compare` / `sweep` / `run-all` run records all of them, together
with the commit SHA and environment, in `report.json`.

## Fluid: damped Navier–Stokes

For an incompressible flow the regularized system reads

$$
\partial_t \mathbf{u} + \mathbf{u}\cdot\nabla\mathbf{u}
= -\nabla p + \nu\nabla^2\mathbf{u} - \lambda_{\mathrm{eff}}\,|\mathbf{u}|^2\mathbf{u},
\qquad
\nabla\cdot\mathbf{u} = 0 ,
$$

with the energy identity

$$
\frac{1}{2}\frac{d}{dt}\|\mathbf{u}\|_{L^2}^2
= -\nu\,\|\nabla\mathbf{u}\|_{L^2}^2
- \lambda_{\mathrm{eff}}\,\|\mathbf{u}\|_{L^4}^4 .
$$

This equation is known in the PDE literature as the (critical, exponent
$r = 3$) **convective Brinkman–Forchheimer** or **damped Navier–Stokes**
equation. In three dimensions, global weak solutions exist for all
coefficients and global strong solutions under conditions on the
coefficients (for $r = 3$ when $4\nu\lambda_{\mathrm{eff}} \ge 1$); see
e.g. X. Cai & Q. Jiu, *J. Math. Anal. Appl.* 343 (2008) 799–809, and
K. W. Hajduk & J. C. Robinson, *J. Differential Equations* 263 (2017)
7141–7161 ([arXiv:1612.02020](https://arxiv.org/abs/1612.02020)). The
mere presence of a cubic damping term is therefore not new; what would
be new is a physical derivation of *this* coefficient. The
Gagliardo–Nirenberg bound
$\|\mathbf{u}\|_{L^4}^4 \le C\,\|\mathbf{u}\|_{L^2}^2\|\nabla\mathbf{u}\|_{L^2}^2$
(in $d = 2$) is an upper bound on the damping dissipation and by itself
excludes nothing.

Implementation notes (`src/moss_reg/fluid/spectral.py`): Fourier
collocation on $[0,2\pi]^2$, divergence-free projection, explicit RK4,
2/3 rule for the quadratic advection term. The 2/3 rule does **not**
dealias the cubic term ($3 \cdot n/3 = n$ aliases back into the retained
band), so $|\mathbf{u}|^2\mathbf{u}$ and the $L^4$ norm are evaluated on a
zero-padded $2n$ grid, which is exact; `cubic_dealias=False` restores the
aliased behaviour and `moss-reg validate` shows the spurious mode it
produces.

## Particles: 1D cold collapse

`LagrangianSystem` advances a self-gravitating 1D/3D particle set with
$\lambda_i = G\rho_i/c^3$ from the local SPH density, and
`JacobianTracker` monitors the deformation Jacobian of the Lagrangian map,
$J = \det(\partial x_i/\partial q_j)$; shell crossing is $\min_i J_i \le 0$.

Code units: $G = M = L = 1$ for a line of total mass $M$ and initial
half-length $L$, so $v_{\mathrm{unit}} = \sqrt{GM/L}$ and the undamped
free-fall speed is $v_{\mathrm{ff}} \approx 1.57$. The speed of light in
code units is $c = 1/\sqrt{\text{compactness}}$.

### Sweep results (`moss-reg sweep`, $t_{\max} = 8$, $dt = 0.005$, $h = 0.12$, softening $0.4$)

Crossing time $t_\times$ of the damped run (classical: $t_\times \approx
1.21$ at every $N$); in parentheses the peak speed in units of $c$:

| $c$ | compactness | $N = 64$ | $N = 128$ | $N = 256$ |
| --- | --- | --- | --- | --- |
| 0.005 | $4\times10^{4}$ | none by 8 (1.55) | none by 8 (1.61) | 5.96 (1.65) |
| 0.01 | $10^{4}$ | 7.59 (1.47) | 4.73 (1.53) | 3.03 (1.57) |
| 0.02 | 2500 | 3.79 (1.44) | 2.50 (1.51) | 1.92 (1.55) |
| 0.05 | 400 | 1.90 (1.44) | 1.67 (1.50) | 1.61 (1.54) |
| 0.1 | 100 | 1.58 (1.43) | 1.53 (1.49) | 1.52 (1.52) |
| 0.5 | 4 | 1.41 (1.38) | 1.28 (1.40) | 1.19 (1.41) |
| 1 | 1 | 1.35 (1.22) | 1.31 (1.24) | 1.30 (1.24) |
| 2 | 0.25 | 1.25 (0.77) | 1.24 (0.78) | 1.24 (0.78) |
| 5 | 0.04 | 1.22 (0.31) | 1.21 (0.31) | 1.21 (0.31) |

* For $c < v_{\mathrm{ff}}$ the cubic term caps the speed at
  $|v| \approx 1.4$–$1.6\,c$ and the collapse proceeds as a crawl; the
  delay of the crossing scales like (particle spacing)/$c$ and decreases
  monotonically with $N$ at fixed $c$.
* The delay is independent of $dt$ (0.02 … 0.0025), weakly dependent on
  $h$, and shifts together with the classical crossing time when the
  softening is changed.
* For $c \ge 2$ (compactness $\le 0.25$, $v_{\mathrm{ff}} < c$) the
  damped and classical runs cross at the same time to within $0.03$.

**Consequently, $J > 0$ throughout a run (labelled "Theorem D.1" in the
0.1.0 documentation, although no such theorem exists in the paper) is not
established by these simulations in any physically admissible regime.** The 0.1.0 figure and the accompanying statement have
been withdrawn; the physical-regime comparison
(`02_shell_crossing_arrest.png`, compactness 0.1) shows the two runs on
top of each other.

## Open physics questions

1. **Missing length.** $G\rho/c^3$ is dimensionally one length short of
   the ODE coupling (see above).
2. **Rest frame.** $-\lambda\rho|\mathbf{v}|^2\mathbf{v}$ damps the velocity
   relative to the coordinate frame. For a fluid in a porous medium the
   medium provides that frame; for "vacuum damping" the theory has to say
   what $\mathbf{v}$ is measured against, or Galilean invariance of the
   Newtonian model is lost and the momentum removed from matter has no
   stated carrier.
3. **$\Xi$ as a modulus.** $c^4/(8\pi G)$ is a force; a modulus requires
   an area.
4. **Why this coefficient.** Cubic damping is a known regularization; the
   claim specific to this work is the value of the coefficient, and that
   derivation is not part of this repository.

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e .            # library + moss-reg CLI
pip install -e ".[dev]"     # + pytest for development
```

## Quickstart

```bash
moss-reg validate                       # 25 PASS/FAIL checks, ~3 s
moss-reg compare --type particles       # classical vs moss, physical regime
moss-reg compare --type particles --c 0.005   # the 0.1.0 strong-coupling regime
moss-reg compare --type fluid --lambda 0.5 --re 200
moss-reg sweep                          # c x N, dt, h, softening (~30 s)
moss-reg run-all                        # everything into assets/ + report.json
moss-reg supplement --copy-figures      # supplement.tex from report.json
```

## CLI reference

```
moss-reg validate [--json PATH] [--strict]
    PASS/FAIL self-checks: dimensional formulas, exact damping solution and
    flow composition, particle-integrator horizon regression, Euler / KDK /
    RK4 convergence orders, kernel normalization, density and gravity paths,
    Jacobian monitor, collapse phenomenology, Taylor-Green decay,
    divergence, energy identity, cubic dealiasing, damping dt limit.
    Exit 1 on any FAIL; --strict also fails on known issues (XFAIL).

moss-reg compare --type {particles,fluid} [--output DIR] [options]
    particles: --n N --compactness K | --c C --t-max T --dt DT --softening E --h H
    fluid:     --n N --lambda X --re RE --t-max T --dt DT
    Writes the figure and merges an entry into DIR/report.json.
    Exit 1 if a sanity check fails (e.g. the classical run did not cross
    within --t-max).

moss-reg sweep [--output DIR] [--quick] [--t-max T] [--dt DT] [--h H] [--softening E]
    Shell-crossing sweep over c x N plus dt / h / softening sensitivity.

moss-reg run-all [--output DIR] [--quick]
    decay, compare:particles, sweep, compare:fluid, validate -> DIR/report.json

moss-reg supplement [--report PATH] [--output PATH] [--figures-dir DIR] [--copy-figures]
    LaTeX supplementary material generated from report.json: provenance,
    full validation table, all parameters and metrics, sweep grid and
    sensitivity tables, figures. Compile with pdflatex/latexmk.

moss-reg benchmark --type {decay,particles,fluid} [options]      (legacy)
```

`report.json` holds one entry per command with: package version, git
commit and dirty flag, UTC timestamp, Python/NumPy/SciPy/Matplotlib
versions, platform, the complete parameter set, all metrics, the
PASS/FAIL checks and the figure paths. Re-running a command into the same
directory replaces only its own entry.

Generated figures:

| figure | content |
| --- | --- |
| `assets/01_velocity_decay.png` | exact decay law vs. explicit Euler, first-order convergence |
| `assets/02_shell_crossing_arrest.png` | 1D collapse, classical vs. moss at compactness 0.1 (physical regime): $\min_i J_i(t)$ and velocity profiles |
| `assets/02_shell_crossing_sweep.png` | crossing time vs. $c$ and $N$; peak speed in units of $c$; sensitivity to $dt$, $h$, softening; Jacobian histories |
| `assets/03_cfd_stability.png` | 2D Taylor–Green: enstrophy, $L^4$ norm, energy for $\lambda = 0$ vs. $\lambda_{\mathrm{code}} = 0.5$ |

## Layout

```
src/moss_reg/
├── constants.py       # c, G, hbar, Xi, lambda, Planck scales (with dimensional notes)
├── dimensions.py      # exact (M, L, T) bookkeeping; check_formulas()
├── validate.py        # moss-reg validate: PASS/FAIL self-checks
├── report.py          # report.json: commit SHA, environment, params, metrics
├── core/
│   ├── analytical.py  # exact ODE damping solution & operator-splitting step
│   └── timestep.py    # stability criteria and adaptive dt
├── fluid/
│   └── spectral.py    # 2D spectral damped NS (padded cubic term, damping dt limit)
├── particles/
│   ├── kernel.py      # cubic spline SPH kernels, density (dense / cKDTree), adaptive h
│   ├── integrator.py  # LagrangianSystem: symmetric KDK + exact damping substeps
│   └── jacobian.py    # deformation gradient tracking, J > 0 monitor
├── benchmarks/        # decay / compare (particles, fluid) / sweep
└── cli.py             # moss-reg command line interface
```

## Testing

```bash
pytest              # ~35 s
moss-reg validate   # the same physics/numerics checks as a PASS/FAIL table
```

The suite covers the dimensional relations (including the rejected
identities and the open $\mathrm{s\,m^{-3}}$ vs. $\mathrm{s\,m^{-2}}$
issue as a strict `xfail`), the exact damping solution, the integrator
horizon regression for the 0.1.0 double-damping bug, dense-vs-tree
density and gravity, the Jacobian monitor, the collapse phenomenology
(speed cap, resolution dependence, unchanged physical regime), the
spectral solver (exact Taylor–Green decay, energy identity, divergence,
cubic dealiasing, damping dt limit) and the CLI including `report.json`.

## License

MIT — see [LICENSE](LICENSE).
