[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22716096.svg)](https://doi.org/10.5281/zenodo.22716096)

# moss-regularization

**Physically grounded nonlinear vacuum-damping regularization for CFD and
N-body simulations.**

`moss-reg` is a research library for regularizing the singularities of
classical continuum dynamics — turbulent blow-up in the Navier–Stokes
equations and shell crossing in self-gravitating collapse — with a
nonlinear damping term whose coupling is fixed by fundamental constants:

$$
\lambda = \frac{G}{c^3} = \frac{1}{8\pi\,\Xi\,c}\,,
\qquad
\Xi = \frac{c^4}{8\pi G}
\quad\text{(vacuum shear modulus)}.
$$

There are **no free parameters**: the damping is set by $c$, $G$ and the
local density $\rho$, and it becomes appreciable only where it is needed —
at ultra-relativistic speeds and in regions approaching singular
compression.

## Core philosophy

Classical treatments regularize singularities by hand (artificial
viscosity, fixed softening, cutoffs). The moss regularization instead
asks: what does the vacuum itself do to a flow that tries to form a
singularity? If the vacuum is treated as an elastic medium with shear
modulus $\Xi = c^4/(8\pi G)$ and the coupling $\lambda = G/c^3$, then the
leading back-reaction on a velocity field $\mathbf{v}$ is

$$
\frac{d\mathbf{v}}{dt} = -\lambda_{\mathrm{eff}}\,|\mathbf{v}|^2\,\mathbf{v},
\qquad
\lambda_{\mathrm{eff}} = \lambda\,\rho = \frac{G\,\rho}{c^3},
$$

the exact analytical solution of which is

$$
\mathbf{v}(t) = \frac{\mathbf{v}_0}{\sqrt{1 + 2\,\lambda_{\mathrm{eff}}\,|\mathbf{v}_0|^2\,t}},
\qquad
\hat{\mathbf{v}}(t) = \hat{\mathbf{v}}_0 .
$$

The substep is integrated **exactly** (no stability limit on dt), the
direction of motion is preserved, and the speed decays as
$|\mathbf{v}| \sim t^{-1/2}$ — deceleration that grows cubically with
velocity, so it is negligible in everyday flows and dominant exactly in
the violent regimes that would otherwise blow up or shell-cross.

## Mathematical summary

For an incompressible flow the regularized Navier–Stokes system reads

$$
\partial_t \mathbf{u} + \mathbf{u}\cdot\nabla\mathbf{u}
= -\nabla p + \nu\nabla^2\mathbf{u} - \lambda_{\mathrm{eff}}\,|\mathbf{u}|^2\mathbf{u},
\qquad
\nabla\cdot\mathbf{u} = 0 .
$$

Multiplying by $\mathbf{u}$ and integrating gives the energy identity

$$
\frac{1}{2}\frac{d}{dt}\|\mathbf{u}\|_{L^2}^2
= -\nu\,\|\nabla\mathbf{u}\|_{L^2}^2
- \lambda_{\mathrm{eff}}\,\|\mathbf{u}\|_{L^4}^4 .
$$

The $L^4$ dissipation term is closed by the Gagliardo–Nirenberg
interpolation (in $d = 2$),

$$
\|\mathbf{u}\|_{L^4}^4 \;\le\; C_{\mathrm{GN}}\,\|\mathbf{u}\|_{L^2}^2\,\|\nabla\mathbf{u}\|_{L^2}^2,
$$

so the damping dissipation is slaved to the enstrophy: whenever the
nonlinearity tries to pump small scales, the $L^4$ channel removes energy
at a rate bounded by the enstrophy itself, excluding the classical
finite-time blow-up route. The benchmark `03_cfd_stability.png` verifies
this identity numerically.

For particles, the same substep is applied per particle with
$\lambda_i = G\rho_i/c^3$ from the local SPH density, inside a
kick–drift–kick integrator. The Lagrangian map $\mathbf{q} \mapsto
\mathbf{x}(\mathbf{q}, t)$ then keeps a positive Jacobian determinant

$$
J = \det\!\left(\frac{\partial x_i}{\partial q_j}\right) > 0
\qquad\text{(Theorem D.1)},
$$

so mass shells do not cross: the damping suppresses the shell-crossing
singularity of cold collapse. This is demonstrated by benchmark
`02_shell_crossing_arrest.png`.

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e .            # library + moss-reg CLI
pip install -e ".[dev]"     # + pytest for development
```

## Quickstart

```bash
pip install -e .
moss-reg run-all            # runs all three benchmarks into assets/
```

or per benchmark:

```bash
moss-reg benchmark --type decay      --lambda 1.0
moss-reg benchmark --type particles  --n 200 --c 0.005
moss-reg benchmark --type fluid      --n 64 --lambda 0.5 --re 200
```

## CLI reference

```
moss-reg run-all [--output DIR]
    Run all three benchmark suites; writes figures into DIR (default assets/).

moss-reg benchmark --type {decay,particles,fluid} [options]
    --output DIR     output directory (default: assets)
    --n N            resolution (particle count / grid size)
    --lambda X       damping coupling (decay, fluid)
    --c C            speed of light in code units, lambda_i = G rho_i / c^3 (particles)
    --t-max T        integration horizon
    --dt DT          maximum timestep
    --re RE          Reynolds number (fluid)
    --softening E    Plummer softening length (particles)
```

Generated figures:

| figure | content |
| --- | --- |
| `assets/01_velocity_decay.png` | analytical decay law vs. explicit Euler, first-order convergence of the discrete scheme |
| `assets/02_shell_crossing_arrest.png` | 1D cold collapse: $\min_i J_i(t)$ and velocity profiles, with and without damping |
| `assets/03_cfd_stability.png` | 2D Taylor–Green: enstrophy $\Omega(t)$, $L^4$ norm $\|u\|_{L^4}(t)$, kinetic energy $E(t)$ for $\lambda = 0$ vs. $\lambda > 0$ |

## Layout

```
src/moss_reg/
├── constants.py       # c, G, hbar, vacuum shear modulus Xi, lambda, Planck scales
├── core/
│   ├── analytical.py  # exact ODE damping solution & operator-splitting step
│   └── timestep.py    # stability criteria and adaptive dt
├── fluid/
│   └── spectral.py    # 2D spectral NS solver with Moss damping (Taylor-Green)
├── particles/
│   ├── kernel.py      # cubic spline SPH kernels, density estimation, adaptive h
│   ├── integrator.py  # LagrangianSystem: KDK + damping substeps, adaptive dt
│   └── jacobian.py    # deformation gradient tracking, J > 0 monitor
├── benchmarks/        # decay / shell-crossing / CFD benchmark suites
└── cli.py             # moss-reg command line interface
```

## Testing

```bash
pytest
```

The suite covers dimensional relations between the constants
($\lambda\,F_P = 1$, $\Xi = c^4/8\pi G$), kernel normalization and mass
conservation, the exact damping solution against Euler integration, the
shell-crossing arrest, and the spectral solver (exact Taylor–Green decay,
energy identity, divergence-free constraint).

## License

MIT — see [LICENSE](LICENSE).
