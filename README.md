[![CI](https://github.com/psldm/moss-regularization/actions/workflows/ci.yml/badge.svg)](https://github.com/psldm/moss-regularization/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22716096.svg)](https://doi.org/10.5281/zenodo.22716096)

# moss-regularization

**Cubic vacuum-damping regularization for CFD and N-body codes: exact
substep, verified 2D/3D spectral and particle solvers, runtime
diagnostics, drop-in adapters (C/C++, C#, NumPy, PyTorch, Dedalus,
OpenFOAM), and a one-command validation harness.**

`moss-reg` implements the nonlinear damping

$$
\frac{d\mathbf{v}}{dt} = -\lambda_{\mathrm{eff}}\,|\mathbf{v}|^{\alpha}\,\mathbf{v},
\qquad \alpha = 2 \ \text{(cubic)},\qquad
\lambda_{\mathrm{eff}} = \frac{G\rho}{c^3}\ \text{(code units)},
$$

proposed as the response of the physical vacuum to extreme fluid motion,
and everything needed to check what the term actually does in a solver.
The accompanying paper (version 3.14) and its machine-generated
supplement are attached to the [releases](https://github.com/psldm/moss-regularization/releases).

## Thirty-second quickstart

```bash
pip install -e .
moss-reg validate                 # 32 PASS/FAIL checks + 1 documented open issue, ~5 s
moss-reg compare --type fluid     # 2D Taylor-Green, classical vs damped: PNG + CSV + report.json
moss-reg compare --type particles # 1D collapse, classical vs damped (physical regime)
moss-reg compare --type fluid3d   # 3D Taylor-Green at 32^3, ~2 min
moss-reg sweep                    # shell-crossing sweep over c x N, dt, h, softening, ~30 s
moss-reg sweep3d                  # 3D resolution x damping-number sweep, ~1 h at 64^3
moss-reg run-all                  # all of the above into assets/ + report.json (+ --no-3d)
moss-reg supplement --copy-figures  # LaTeX supplement generated from report.json
```

Every command writes its figure, the per-step diagnostics as CSV, and an
entry in `assets/report.json` with the git commit, environment, parameters,
metrics and PASS/FAIL checks. `validate` exits non-zero on any failure, so
it can gate a CI pipeline.

## What is shown, and what is not

**Shown, and checked on every CI run:**

* the exact solution of the damping ODE, its use as an unconditionally
  stable substep, and the composition property $\phi_a\circ\phi_b=\phi_{a+b}$;
* a symmetric kick/2–damp/2–drift–damp/2–kick/2 particle integrator
  (second order) whose energy budget $E_{\mathrm{kin}}+E_{\mathrm{pot}}+E_{\mathrm{vac}}$
  is a checked invariant;
* 2D and 3D spectral solvers for the damped Navier–Stokes equations with the
  cubic term properly dealiased, verified against exact solutions to
  $10^{-15}$ and against the energy identity to $\sim 10^{-7}$;
* the phenomenology of the term in a 1D cold collapse: a **speed limiter
  near $c$** with terminal velocity $v_{\mathrm{term}} = c\,(g/G\rho)^{1/3}$,
  a crossing delay of about one particle spacing over $c$ that **shrinks
  with resolution**, and no change at all when the free-fall speed is
  below $c$ (figure `02_shell_crossing_sweep.png`).

**Not shown:**

* prevention of finite-time singularities of Navier–Stokes: 2D has none,
  and the 3D benchmark at $32^3$–$64^3$ is a solver and budget test whose
  classical run becomes under-resolved (reported through $k_{\max}\eta$);
* arrest of shell crossing at physical parameters (the opposite is shown);
* a parameter-free stabilizer for engineering CFD: with the physical
  coefficient the term is inert wherever classical fluid dynamics is valid,
  and a coefficient large enough to act is a tuned regularization
  (see *Choosing the coefficient*).

## For engineers: embedding the substep

The damping substep is three lines of arithmetic, exact for any timestep:

```
alpha = 2 :  v <- v / sqrt(1 + 2 lam |v|^2 dt)
general   :  v <- v (1 + alpha lam |v|^alpha dt)^(-1/alpha)
```

Use it as the damping stage of an operator splitting (your predictor for
convection / viscosity / gravity, then the substep on the velocity), or as a
semi-implicit diagonal source `Sp = -lam |u|^2` in a segregated solver.

| adapter | where | tested |
| --- | --- | --- |
| C / C++ header-only | [`integrations/c/moss_damp.h`](integrations/c/moss_damp.h) | compiled and run in CI (C99, C++), cross-checked with Python |
| C# / .NET | [`integrations/csharp/MossDamp.cs`](integrations/csharp/MossDamp.cs) | `dotnet run` self-test in CI |
| NumPy | `moss_reg.integrations.damp(v, dt, lam, alpha=2)` | yes |
| PyTorch (CPU/GPU, autograd) | `moss_reg.integrations.torch_ops.damp` | yes when torch is installed |
| Dedalus v3 | [`integrations/dedalus/moss_damping_dedalus.py`](integrations/dedalus/moss_damping_dedalus.py) | example |
| OpenFOAM `fvOptions` | [`integrations/openfoam/`](integrations/openfoam/) | example |

```c
#include "moss_damp.h"
/* after your predictor step, per particle / cell: */
moss_damp(v, 3, lam, dt);                       /* v[3] in place, exact */
e_vac += moss_damp_array(v_all, n, 3, lam_i, 0.0, mass, dt);  /* energy to the reservoir */
```

```python
from moss_reg.integrations import damp, damp_with_energy
v_new = damp(v, dt, lam)                         # any shape (..., D), per-item lam allowed
v_new, removed = damp_with_energy(v, dt, lam, mass=m)
```

If the term is evaluated inside an explicit right-hand side instead
(Runge–Kutta stages), it carries the stiffness limit
$\Delta t \le \mathrm{CFL}\cdot 2/(3\lambda_{\mathrm{eff}}|u|_{\max}^2)$;
`cfl_dt()` and `moss_damp_dt_limit()` provide it.

### Choosing the coefficient

The number a solver needs is the **damping number**
$\lambda_{\mathrm{code}} = \lambda_{\mathrm{eff}}\,U^{\alpha} L$ of your
nondimensionalization (`moss_reg.fluid.damping_number`). Three ways to get it:

* code units with $G = 1$ and unit mass and length: $\lambda_i = G\rho_i/c^3$
  from the local density, as the particle integrator does;
  `lambda_field_from_density` does the same for a density field on a grid;
* SI: `lambda_eff_si(rho, ell)` $= G\rho\ell/c^3$ in $\mathrm{s\,m^{-2}}$,
  where the length $\ell$ is **not yet supplied by the theory** (open
  problem: $G\rho/c^3$ alone has dimension $\mathrm{s\,m^{-3}}$);
* a value chosen by hand to make the effect visible, as the benchmarks do
  with $\lambda_{\mathrm{code}} = 0.5$.

Be aware what the choice means. For water at $10^3\ \mathrm{m/s}$ with
$\ell = L = 1\ \mathrm{m}$, `physical_damping_number` gives
$\lambda_{\mathrm{code}} \approx 2.5\times10^{-27}$: the physical term does
nothing in any engineering flow. A $\lambda_{\mathrm{code}}$ of order unity
is a tuning parameter, comparable to an artificial viscosity, and should be
reported with the results; `moss-reg sweep` exists to document how results
depend on it, on resolution and on the timestep.

## Diagnostics API

```python
from moss_reg.diagnostics import field_norms, DiagnosticsLog
from moss_reg.fluid import SpectralNS2D

s = SpectralNS2D(64, nu=1/200, lam=0.5); s.set_taylor_green()
log = DiagnosticsLog()
log.record(s.time, **s.diagnostics())            # E, enstrophy, grad_l2_sq, l4, linf, div_max, vac_power
while s.time < 5.0:
    s.step(s.cfl_dt()); log.record(s.time, **s.diagnostics())
budget = log.budget(nu=s.nu)                     # E(t) = E0 - D_visc(t) - E_vac(t)
print(budget.summary())                          # residual ~1e-7, E_vac_final, vac_fraction
log.write_csv("run.csv"); log.write_json("run.json")

norms = field_norms([u_axis0, u_axis1], box=2*np.pi)   # any periodic grid field, any dimension
```

For particle systems `LagrangianSystem.diagnostics()` returns
$E_{\mathrm{kin}}$, $E_{\mathrm{pot}}$ (Plummer-consistent), the exactly
accumulated $E_{\mathrm{vac}}$, $E_{\mathrm{total}}$, $v_{\max}$,
$\rho_{\max}$ and $J_{\min}$ of the Lagrangian map; the benchmarks write
these per step as CSV.

## Equations and constants

Along a trajectory the damping alone has the closed-form solution

$$
\mathbf{v}(t) = \frac{\mathbf{v}_0}{\sqrt{1 + 2\,\lambda_{\mathrm{eff}}\,|\mathbf{v}_0|^2\,t}},
\qquad \hat{\mathbf{v}}(t) = \hat{\mathbf{v}}_0 ,
$$

and for a general exponent $\mathbf{v}(t) = \mathbf{v}_0\,(1 + \alpha\lambda_{\mathrm{eff}}|\mathbf{v}_0|^{\alpha}t)^{-1/\alpha}$
(`exact_damping_alpha`). For an incompressible flow the damped system is

$$
\partial_t \mathbf{u} + \mathbf{u}\cdot\nabla\mathbf{u}
= -\nabla p + \nu\nabla^2\mathbf{u} - \lambda_{\mathrm{eff}}\,|\mathbf{u}|^2\mathbf{u},
\qquad \nabla\cdot\mathbf{u} = 0,
\qquad
\frac{1}{2}\frac{d}{dt}\|\mathbf{u}\|_{L^2}^2
= -\nu\,\|\nabla\mathbf{u}\|_{L^2}^2 - \lambda_{\mathrm{eff}}\,\|\mathbf{u}\|_{L^4}^4 .
$$

This is the critical ($r = 3$) convective Brinkman–Forchheimer equation of
the PDE literature: global strong solutions are known for $\alpha > 2$ and
every coefficient, and for $\alpha = 2$ when $4\nu\lambda_{\mathrm{eff}} \ge 1$
(Cai & Jiu 2008; Zhou 2012; Hajduk & Robinson 2017). The physically
motivated case, $\alpha = 2$ with a tiny coefficient, is open. At $\alpha = 2$
the term scales exactly like the viscous term under the Navier–Stokes
scaling, so a small coefficient does not change the scaling class.

**Constants and dimensions** (`moss_reg.constants`, `moss_reg.dimensions`,
tested): $\lambda = G/c^3$ is $\mathrm{s\,kg^{-1}}$; $\Xi = c^4/(8\pi G)$ is
a **force**, not a pressure; $1/(8\pi\Xi c) = G/c^5 \neq \lambda$; and the
ODE requires $[\lambda_{\mathrm{eff}}] = \mathrm{s\,m^{-2}}$ while
$G\rho/c^3$ has $\mathrm{s\,m^{-3}}$ (reported as `XFAIL` by `validate`).

**Parameters.** The fundamental coefficient is not fitted; the numerical
models have parameters, all recorded in `report.json`:

| parameter | where | meaning |
| --- | --- | --- |
| $N$, $\Delta t_{\max}$, $h$, softening | particles | resolution, timestep, SPH kernel scale, Plummer softening |
| $c$ (code units) or compactness $GM/(Lc^2) = 1/c^2$ | particles | the one physical dial of the collapse; physical bodies have compactness $\le 0.5$ |
| $N$, $\Delta t_{\max}$, $Re$ | fluid, fluid3d | resolution, timestep, Reynolds number |
| $\lambda_{\mathrm{code}}$ | fluid, fluid3d | damping number (0.5 in the benchmarks, chosen to be visible) |

## Fluid solvers and benchmarks

`SpectralNS2D` / `SpectralNS3D` (`src/moss_reg/fluid/`): Fourier collocation
on $[0,2\pi]^d$, divergence-free projection, explicit RK4, 2/3 rule for the
quadratic term (rotational form in 3D). The 2/3 rule does **not** dealias
the cubic term, so $|\mathbf{u}|^2\mathbf{u}$ and the $L^4$ norm are evaluated on
a zero-padded $2n$ grid (exact; `validate` shows the spurious mode the
2/3 rule alone produces). `lam` may be a scalar or a band-limited field
$\lambda(\mathbf{x})$. `dt_limits()` reports the advection, viscous and
damping limits.

Two ways to apply the damping (`damping_mode`):

* `"rhs"` (default): the term is part of the RK4 right-hand side; it carries
  the stiffness limit $\Delta t \le \mathrm{CFL}\cdot 2/(3\lambda|u|_{\max}^2)$
  and the energy handed to the reservoir is obtained by quadrature of
  $\lambda\|u\|_4^4$.
* `"split"`: Strang splitting, half exact substep on the padded grid,
  RK4 for the Navier–Stokes part, half exact substep, each followed by
  truncation and projection. No timestep limit from the damping, the
  reservoir energy `energy_to_vacuum` is accumulated exactly, and the
  small truncation/projection loss of the substep is reported separately
  (`energy_split_loss`). Because the pointwise substep does not commute
  with the divergence-free projection this mode is **first order** in
  $\Delta t$ (the two modes converge to each other as $\Delta t$, tested);
  `moss-reg sweep3d` uses `"split"` so that large damping numbers cost
  nothing extra, at $\Delta t = 0.01$ the two modes differ by $\sim 10^{-4}$
  relative in the energy.

* `compare --type fluid` (2D Taylor–Green, $Re = 200$, $n = 64$): classical
  run reproduces the exact viscous decay to $5\times10^{-15}$; the damped
  run satisfies the energy identity with residual $4\times10^{-7}$. 2D
  Navier–Stokes has no finite-time blow-up: this verifies the solver.
* `compare --type fluid3d` (3D Taylor–Green, $Re = 800$, $32^3$): energy,
  enstrophy, $\|\omega\|_\infty$, $\|u\|_4$ for both runs, energy budget
  residual as a check, and two resolution indicators: the spectral tail
  fraction (energy at $|k| > 0.8\,k_{\max}$; above $10^{-3}$ the run is
  called under-resolved, and the figure marks from when) and the turbulence
  criterion $k_{\max}\eta$ with $\eta = (\nu^3/\varepsilon)^{1/4}$.
* `sweep3d`: the classical run at several resolutions (which $N$ stays
  resolved over the horizon) and the damped run over a range of damping
  numbers, with the threshold $4\lambda_{\mathrm{code}}/Re = 1$ of the
  regularity theorem marked. Long (about an hour at $64^3$); the numbers it
  produces are the honest 3D input to the paper, not a result about the
  open case.

## Particles: 1D cold collapse

`LagrangianSystem` advances a self-gravitating particle set with
$\lambda_i = G\rho_i/c^3$ from the local SPH density; `JacobianTracker`
monitors $J = \det(\partial x_i/\partial q_j)$, and shell crossing is
$\min_i J_i \le 0$. Code units $G = M = L = 1$, free-fall speed
$v_{\mathrm{ff}} \approx 1.57$, $c = 1/\sqrt{\text{compactness}}$.

Crossing time of the damped run (`moss-reg sweep`, $t_{\max} = 8$; classical
$t_\times \approx 1.21$ at every $N$; peak $|v|/c$ in parentheses):

| $c$ | compactness | $N = 64$ | $N = 128$ | $N = 256$ |
| --- | --- | --- | --- | --- |
| 0.005 | $4\times10^{4}$ | none by 8 (1.55) | none by 8 (1.61) | 5.96 (1.65) |
| 0.01 | $10^{4}$ | 7.59 (1.47) | 4.73 (1.53) | 3.03 (1.57) |
| 0.05 | 400 | 1.90 (1.44) | 1.67 (1.50) | 1.61 (1.54) |
| 0.5 | 4 | 1.41 (1.38) | 1.28 (1.40) | 1.19 (1.41) |
| 2 | 0.25 | 1.25 (0.77) | 1.24 (0.78) | 1.24 (0.78) |
| 5 | 0.04 | 1.22 (0.31) | 1.21 (0.31) | 1.21 (0.31) |

The delay is independent of $\Delta t$, weakly dependent on $h$, shifts with
the classical crossing when the softening changes, and vanishes with
resolution; for $c \ge 2$ the two runs cross within $0.03$ of each other.
$J > 0$ throughout a run is therefore not established in any physically
admissible regime.

## Open physics questions

1. **Missing length.** $G\rho/c^3$ is one length short of the ODE coupling.
2. **Rest frame.** The term damps velocity relative to the coordinate
   frame; the hypothesis must say what $\mathbf{v}$ is measured against and
   what carries the removed momentum.
3. **$\Xi$ as a modulus.** $c^4/(8\pi G)$ is a force; a modulus needs an area.
4. **Exponent.** $\alpha$ is not derived; $\alpha = 2$ is exactly critical.

## CLI reference

```
moss-reg validate [--json PATH] [--strict]
    32 PASS/FAIL checks + 1 XFAIL: dimensional formulas, exact damping and
    composition, integrator horizon regression, Euler / KDK / RK4 orders,
    kernels, density and gravity paths, Jacobian monitor, collapse
    phenomenology, 2D and 3D exact decays, divergence, energy budgets,
    cubic dealiasing (2D, 3D), damping dt limit, particle energy invariant,
    scaling admissibility of the paper's inequalities and the criticality
    of alpha = 2.

moss-reg compare --type {particles,fluid,fluid3d} [--output DIR] [options]
    particles: --n N --compactness K | --c C --t-max T --dt DT --softening E --h H
    fluid, fluid3d: --n N --lambda X --re RE --t-max T --dt DT
    Writes the figure, per-step CSV time series and an entry in DIR/report.json.
    Exit 1 if a sanity check fails.

moss-reg sweep [--output DIR] [--quick] [--t-max T] [--dt DT] [--h H] [--softening E]
moss-reg sweep3d [--output DIR] [--quick] [--re RE] [--t-max T] [--dt DT]
                 [--classical-n N ...] [--damped-n N ...] [--lambdas X ...] [--lambdas-small-n X ...]
moss-reg run-all [--output DIR] [--quick] [--no-3d]
moss-reg supplement [--report PATH] [--output PATH] [--figures-dir DIR] [--copy-figures]
moss-reg scaling "<inequality>" [--dim d]
    Homogeneity check of a norm inequality, e.g.
    moss-reg scaling "|D1 u|_2^2 <= |u|_4^{4/3} |D2 u|_2^{2/3}"   -> REJECTED (exit 1)
    moss-reg scaling "|D1 u|_2 <= |u|_4^{4/5} |D2 u|_2^{1/5}"    -> ADMISSIBLE (exit 0)
    A mismatch of dilation or amplitude exponents proves the inequality false
    for every constant; run it on any inequality before reading its proof.
moss-reg benchmark --type {decay,particles,fluid} [options]      (legacy)
```

| figure | content |
| --- | --- |
| `assets/01_velocity_decay.png` | exact decay law vs. explicit Euler, first-order convergence |
| `assets/02_shell_crossing_arrest.png` | 1D collapse, classical vs. damped at compactness 0.1 |
| `assets/02_shell_crossing_sweep.png` | crossing time vs. $c$ and $N$, speed cap, sensitivities, Jacobian histories |
| `assets/03_cfd_stability.png` | 2D Taylor–Green: enstrophy, $L^4$ norm, energy |
| `assets/05_tg3d.png` | 3D Taylor–Green: energy, enstrophy, $\|\omega\|_\infty$, $L^4$ norm, under-resolution marker |
| `assets/06_tg3d_sweep.png` | 3D sweep: classical enstrophy vs. $N$, damped enstrophy vs. $\lambda_{\mathrm{code}}$, peak values with the $4\lambda/Re = 1$ threshold, spectral tails |

## Layout

```
src/moss_reg/
├── constants.py, dimensions.py   # constants with dimensional notes; exact (M, L, T) bookkeeping
├── validate.py, report.py        # PASS/FAIL harness; report.json provenance
├── supplement.py                 # LaTeX supplement generated from report.json
├── core/                         # exact damping substep (alpha = 2 and general), timestep limits
├── diagnostics/                  # norms, energy budget, DiagnosticsLog
├── fluid/                        # SpectralNS2D, SpectralNS3D, coupling helpers
├── particles/                    # SPH kernels, LagrangianSystem (symmetric KDK + damping), Jacobian
├── integrations/                 # NumPy and PyTorch adapters
├── benchmarks/                   # decay, compare (particles, fluid, fluid3d), sweep
└── cli.py
integrations/                     # C/C++ header + test, C# + test, Dedalus, OpenFOAM
examples/                         # analytical decay, collapse comparison, under-resolved shear layer
```

## Testing and CI

```bash
pytest              # ~60 s (skips PyTorch / .NET checks when not installed)
moss-reg validate
```

CI runs the test-suite on Python 3.11–3.13, `moss-reg validate`, the C
self-test with `-Wall -Wextra -Werror` (C99 and C++), the C# self-test on
.NET 8, and the cross-checks of every adapter against the Python reference.

## Out of scope

Compressible flow and shock tubes (the solvers are incompressible), and
relativistic regimes, where the term would first matter physically but
where the incompressible Navier–Stokes equations and Newtonian gravity are
no longer the right description.

## Citation

E. Moss, *Physical Regularization of Navier–Stokes Blow-Up via Nonlinear
Vacuum Response*, version 3.14 (2026), with the generated supplementary
material; software: `moss-regularization`, this repository (DOI above).

## License

MIT — see [LICENSE](LICENSE).
