# Contributing

Thank you for looking at `moss-reg`. The rules below keep the package in the
state its README promises: every claim backed by a running check.

## Ground rules

1. **No number without a check.** A new benchmark returns a metrics dictionary
   with a `checks` block (PASS/FAIL booleans) and a `params` block; the CLI
   records both in `report.json`. Anything quoted in the README or the paper
   must come from there.
2. **Physics claims stay separate from numerics.** If a result contradicts a
   statement in the README, change the statement, not the result.
3. **The exact substep is the reference.** Any adapter (C, C#, NumPy,
   PyTorch, ...) must reproduce `moss_reg.core.analytical.exact_damping` to
   round-off in `tests/test_integrations.py`.
4. **Known open issues are XFAIL, not silence.** The dimensional gap of
   `G rho / c^3` is a strict `xfail` in the test-suite and an `XFAIL` row in
   `moss-reg validate`; do not "fix" it by removing the check.

## Development

```bash
pip install -e ".[dev]"
pytest                      # ~60 s; PyTorch / .NET checks skip when not installed
moss-reg validate           # must print 0 failed
```

CI runs the test-suite on Python 3.11–3.13, `moss-reg validate`, the C/C++
header self-test with `-Wall -Wextra -Werror`, the C# self-test on .NET 8,
and the PyTorch adapter on CPU.

## Adding a benchmark

* Put it in `src/moss_reg/benchmarks/`, expose `run_<name>(outdir, **params)`
  returning `{..., "params": {...}, "checks": {...}, "figure": path}`.
* Record per-step diagnostics with `moss_reg.diagnostics.DiagnosticsLog` and
  write them as CSV next to the figure.
* Wire it into `cli.py` (`compare` or a new subcommand) and, if it belongs in
  the default run, into `run-all`.
* Use the palette in `benchmarks/_style.py` (classical = blue, damped = orange).
* Add a smoke test with a tiny grid and a short horizon.

## Releases

1. Commit the code. 2. `moss-reg run-all --output assets` on a clean tree
(`report.json` records the commit and a `dirty` flag). 3. Commit the assets.
4. Tag `vX.Y.Z` and push. 5. Publish the GitHub release with the generated
supplement (`moss-reg supplement`) and `timeseries_<version>.zip`; the
release event triggers the PyPI publication workflow.

## Reporting results

Please open an issue with the *benchmark result* template: it asks for the
`report.json` entry, which contains the commit, environment and parameters
needed to reproduce what you saw.
