# Integrations

The damping substep is three lines of arithmetic; these adapters put it
where an existing solver can call it.

| adapter | file | tested |
| --- | --- | --- |
| C / C++ header-only | `c/moss_damp.h`, self-test `c/test_moss_damp.c` | yes: compiled and run in CI (gcc, C99 and C++ syntax) and cross-checked against the Python reference in `tests/test_integrations.py` |
| C# / .NET | `csharp/MossDamp.cs`, self-test `csharp/MossDamp.Test` | yes: `dotnet run` in CI (.NET 8) |
| NumPy | `moss_reg.integrations.damp`, `damp_with_energy` | yes |
| PyTorch | `moss_reg.integrations.torch_ops.damp` (CPU/GPU, autograd) | yes when torch is installed (skipped otherwise) |
| Dedalus v3 | `dedalus/moss_damping_dedalus.py` (source term in an IVP) | example only |
| OpenFOAM | `openfoam/fvOptions.mossDamping` (semi-implicit coded source) | example only |

All adapters implement the same kernel, exact for any timestep:

```
alpha = 2 :  v <- v / sqrt(1 + 2 lam |v|^2 dt)
general   :  v <- v (1 + alpha lam |v|^alpha dt)^(-1/alpha)
```

`lam` is the effective coupling in the solver's own units: `G rho / c^3` in
code units with `G = 1`, or the dimensionless damping number
`lam_code = lam_eff U^alpha L`.  What that number means physically, and why a
value that visibly changes an engineering flow is a tuning parameter, is
explained in the top-level README.
