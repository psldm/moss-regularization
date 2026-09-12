"""Dedalus v3 example: 2D periodic incompressible Navier-Stokes with the
cubic damping term  -lam |u|^2 u  as a source term.

Run:  python moss_damping_dedalus.py            (requires dedalus >= 3)

This file shows *where the term goes*; it is not executed in moss-reg's
test-suite.  Compare with moss_reg.fluid.spectral.SpectralNS2D, which
solves the same equations with its own spectral code.  Note that with
the term inside the implicit/explicit time stepper it is treated
explicitly and inherits the stiffness limit dt <= 2/(3 lam |u|_max^2)
(see README); the exact substep is only available in operator-split codes.
"""

import numpy as np

try:
    import dedalus.public as d3
except ImportError as exc:  # pragma: no cover - example only
    raise SystemExit("this example needs dedalus (pip install dedalus)") from exc

# ---- parameters (code units, box 2 pi, Taylor-Green initial condition) ----
Lx = Ly = 2 * np.pi
Nx = Ny = 64
nu = 1.0 / 200.0          # Re = 200
lam = 0.5                 # damping number lambda_code = lam_eff U L (tuned, see README)
stop_time = 10.0
timestep = 0.005

coords = d3.CartesianCoordinates("x", "y")
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords["x"], size=Nx, bounds=(0, Lx), dealias=3 / 2)
ybasis = d3.RealFourier(coords["y"], size=Ny, bounds=(0, Ly), dealias=3 / 2)

u = dist.VectorField(coords, name="u", bases=(xbasis, ybasis))
p = dist.Field(name="p", bases=(xbasis, ybasis))
tau_p = dist.Field(name="tau_p")

# ---- equations: the Moss term is the last item on the right-hand side ----
problem = d3.IVP([u, p, tau_p], namespace=locals())
problem.add_equation("dt(u) + grad(p) - nu*lap(u) = - u@grad(u) - lam*(u@u)*u")
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0")

solver = problem.build_solver(d3.RK443)
solver.stop_sim_time = stop_time

x, y = dist.local_grids(xbasis, ybasis)
u["g"][0] = np.sin(x) * np.cos(y)
u["g"][1] = -np.cos(x) * np.sin(y)

# ---- diagnostics: energy, damping power (feeds E_vac), max speed --------------
energy = d3.integ(0.5 * u @ u) / (Lx * Ly)
vac_power = d3.integ(lam * (u @ u) ** 2) / (Lx * Ly)
while solver.proceed:
    solver.step(timestep)
    if solver.iteration % 200 == 0:
        print(f"t = {solver.sim_time:6.3f}  E = {energy.evaluate()['g'].ravel()[0]:.6f}  "
              f"lam<|u|^4> = {vac_power.evaluate()['g'].ravel()[0]:.3e}")
