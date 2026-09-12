# OpenFOAM: cubic damping as a semi-implicit `fvOptions` source

The Moss term enters the momentum equation as a pointwise source
`S(U) = -lam |U|^2 U`.  In OpenFOAM it is most naturally added as a coded
source that supplies the explicit part and its linearised implicit
coefficient (semi-implicit form, unconditionally stable for the damping
itself):

```
dS/dU ~ -lam |U|^2 (diagonal)     =>    fvm::Sp(-lam*magSqr(U), U)
```

`fvOptions.mossDamping` contains a complete `vectorCodedSource` entry for
`incompressibleFluid` / `pimpleFoam`-type solvers (OpenFOAM.org v9+ syntax;
adapt `codeAddSup` to your version).  Copy it into `system/fvOptions` (or
`constant/fvOptions` for older versions) and set `lam` to the damping number
of your nondimensionalisation, `lam_code = lam_eff U^2 L` (see README).

Notes for engineers:

* With the physical coupling `lam_eff = G rho l / c^3` the source is
  numerically zero in any engineering flow; a value of `lam_code` that
  visibly acts on the solution is a tuning parameter, comparable to an
  artificial viscosity.  Report it with the results.
* The implicit treatment removes the stiffness limit `dt <= 2/(3 lam |U|^2)`
  that an explicit source would have.  For an operator-split alternative,
  apply the exact substep `U <- U / sqrt(1 + 2 lam |U|^2 dt)` to the
  velocity field after the momentum corrector (see `integrations/c/moss_damp.h`
  for the three-line kernel).
* This directory is an example and is not compiled or run by the moss-reg
  test-suite.
