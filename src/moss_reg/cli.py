"""Command-line interface for moss-regularization.

Usage:
    moss-reg validate  [--json PATH] [--strict]
    moss-reg compare   --type {particles,fluid,fluid3d} [options]
    moss-reg sweep     [options]
    moss-reg run-all   [--output DIR] [--quick]
    moss-reg supplement [--report PATH] [--output PATH] [--copy-figures]
    moss-reg benchmark --type {decay,particles,fluid} [options]   (legacy)

``validate`` runs the PASS/FAIL self-checks (formulas, integrators,
convergence, energy balance, Jacobian monitor) and exits 1 on failure.

``compare`` runs one classical-vs-moss comparison, writes the figure and
merges a reproducibility entry (commit SHA, environment, parameters,
metrics, checks) into ``<outdir>/report.json``.

``sweep`` runs the shell-crossing parameter sweep (c x N, plus dt / h /
softening sensitivity) and writes ``02_shell_crossing_sweep.png``.

``run-all`` executes decay, compare:particles, sweep, compare:fluid and
validate, writing all figures and ``report.json`` into the output
directory (``assets/`` by default).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import validate as _validate
from .benchmarks import (
    run_cfd_benchmark,
    run_decay_benchmark,
    run_shell_crossing_benchmark,
    run_shell_crossing_sweep,
    run_tg3d_benchmark,
)
from .report import git_info, make_entry, sanitize, write_report
from .supplement import write_supplement

__all__ = ["main"]


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output", default="assets", help="output directory (default: assets)"
    )


def _add_particle_params(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=None, help="number of particles")
    parser.add_argument(
        "--compactness", type=float, default=None,
        help="G M / (L c^2) of the collapsing line in code units (G = M = L = 1); "
        "sets c = 1 / sqrt(compactness). Physical bodies have compactness <= 0.5",
    )
    parser.add_argument(
        "--c", type=float, default=None,
        help="speed of light in code units (overrides --compactness); "
        "lambda_i = G rho_i / c^3",
    )
    parser.add_argument("--t-max", type=float, default=None, help="integration horizon")
    parser.add_argument("--dt", type=float, default=None, help="maximum timestep")
    parser.add_argument("--softening", type=float, default=None, help="Plummer softening")
    parser.add_argument("--h", type=float, default=None, help="SPH smoothing length")


def _add_fluid_params(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=None, help="grid resolution")
    parser.add_argument(
        "--lambda", dest="lambda_eff", type=float, default=None,
        help="dimensionless damping number lambda_code = lambda_eff U L",
    )
    parser.add_argument("--re", type=float, default=None, help="Reynolds number")
    parser.add_argument("--t-max", type=float, default=None, help="integration horizon")
    parser.add_argument("--dt", type=float, default=None, help="maximum timestep")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moss-reg",
        description="Nonlinear vacuum-damping regularization: validation, "
        "classical-vs-moss comparisons and parameter sweeps.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    val = sub.add_parser("validate", help="run PASS/FAIL self-checks")
    val.add_argument("--json", dest="json_path", default=None, help="write results as JSON")
    val.add_argument(
        "--strict", action="store_true",
        help="treat known open issues (XFAIL) as failures",
    )

    cmp_ = sub.add_parser("compare", help="classical vs moss comparison, PNG + report.json")
    cmp_.add_argument(
        "--type", dest="btype", required=True, choices=["particles", "fluid", "fluid3d"]
    )
    _add_output(cmp_)
    cmp_.add_argument("--n", type=int, default=None, help="resolution (particles / grid)")
    cmp_.add_argument("--compactness", type=float, default=None,
                      help="particles: G M / (L c^2), sets c = 1/sqrt(compactness)")
    cmp_.add_argument("--c", type=float, default=None,
                      help="particles: speed of light in code units (overrides --compactness)")
    cmp_.add_argument("--lambda", dest="lambda_eff", type=float, default=None,
                      help="fluid/fluid3d: dimensionless damping number")
    cmp_.add_argument("--re", type=float, default=None, help="fluid/fluid3d: Reynolds number")
    cmp_.add_argument("--t-max", type=float, default=None, help="integration horizon")
    cmp_.add_argument("--dt", type=float, default=None, help="maximum timestep")
    cmp_.add_argument("--softening", type=float, default=None, help="particles: Plummer softening")
    cmp_.add_argument("--h", type=float, default=None, help="particles: SPH smoothing length")

    swp = sub.add_parser("sweep", help="shell-crossing sweep over c x N, dt, h, softening")
    _add_output(swp)
    swp.add_argument("--quick", action="store_true", help="reduced grid (smoke test)")
    swp.add_argument("--t-max", type=float, default=None, help="integration horizon")
    swp.add_argument("--dt", type=float, default=None, help="baseline maximum timestep")
    swp.add_argument("--h", type=float, default=None, help="baseline smoothing length")
    swp.add_argument("--softening", type=float, default=None, help="baseline softening")

    sup = sub.add_parser("supplement", help="LaTeX supplementary material generated from report.json")
    sup.add_argument("--report", default="assets/report.json", help="input report.json")
    sup.add_argument("--output", default="supplement.tex", help="output .tex path")
    sup.add_argument("--figures-dir", default="figures",
                     help="figure directory relative to the output file (default: figures)")
    sup.add_argument("--copy-figures", action="store_true",
                     help="copy the report's figures into <output dir>/<figures-dir>")
    sup.add_argument("--title", default="Supplementary Numerical Material")
    sup.add_argument("--subtitle", default="")

    run_all = sub.add_parser("run-all", help="run every benchmark, sweep and validate")
    _add_output(run_all)
    run_all.add_argument("--quick", action="store_true", help="reduced sweep grid and 3D resolution")
    run_all.add_argument("--no-3d", action="store_true", help="skip the 3D Taylor-Green benchmark")

    bench = sub.add_parser(
        "benchmark", help="(legacy) run a single benchmark suite with custom parameters"
    )
    bench.add_argument(
        "--type", dest="btype", required=True, choices=["decay", "particles", "fluid"]
    )
    _add_output(bench)
    bench.add_argument("--n", type=int, default=None, help="resolution (particles / grid)")
    bench.add_argument("--lambda", dest="lambda_eff", type=float, default=None,
                       help="damping coupling (decay/fluid)")
    bench.add_argument("--c", type=float, default=None,
                       help="speed of light in code units (particles)")
    bench.add_argument("--compactness", type=float, default=None,
                       help="G M / (L c^2) (particles)")
    bench.add_argument("--t-max", type=float, default=None, help="integration horizon")
    bench.add_argument("--dt", type=float, default=None, help="maximum timestep")
    bench.add_argument("--re", type=float, default=None, help="Reynolds number (fluid)")
    bench.add_argument("--softening", type=float, default=None, help="Plummer softening (particles)")
    bench.add_argument("--h", type=float, default=None, help="SPH smoothing length (particles)")
    return parser


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _print_metrics(title: str, metrics: Dict[str, Any], indent: int = 2,
                   as_checks: bool = False) -> None:
    """Print a metrics dictionary; booleans read PASS/FAIL only inside ``checks``."""
    pad = " " * indent
    if indent == 2:
        print(f"[{title}]")
    for key, value in metrics.items():
        key = str(key)
        if isinstance(value, dict):
            print(f"{pad}{key}:")
            _print_metrics(key, value, indent + 2, as_checks=(key == "checks"))
        elif isinstance(value, bool):
            if as_checks:
                print(f"{pad}{key:28s} {'PASS' if value else 'FAIL'}")
            else:
                print(f"{pad}{key:28s} {'true' if value else 'false'}")
        elif isinstance(value, float):
            print(f"{pad}{key:28s} {value:.6g}")
        elif isinstance(value, (list, tuple)) and len(value) > 8:
            print(f"{pad}{key:28s} [{len(value)} values]")
        else:
            print(f"{pad}{key:28s} {value}")


def _kwargs(args: argparse.Namespace, mapping: Dict[str, str]) -> Dict[str, Any]:
    out = {}
    for attr, key in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            out[key] = value
    return out


_PARTICLE_MAP = {
    "n": "n", "compactness": "compactness", "c": "c", "t_max": "t_max",
    "dt": "dt_max", "softening": "softening", "h": "h",
}
_FLUID_MAP = {"n": "n", "lambda_eff": "lam", "re": "re", "t_max": "t_max", "dt": "dt_max"}
_SWEEP_MAP = {"t_max": "t_max", "dt": "dt_max", "h": "h", "softening": "softening"}


def _record(outdir: Path, command: str, params: Dict[str, Any],
            metrics: Dict[str, Any], argv: Optional[List[str]],
            git: Optional[Dict[str, Any]] = None) -> Path:
    metrics = dict(metrics)
    checks = metrics.pop("checks", {})
    figures = [metrics.pop("figure")] if "figure" in metrics else []
    entry = make_entry(command, params, metrics, figures, checks=checks, argv=argv, git=git)
    return write_report(outdir, entry)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_validate(args: argparse.Namespace) -> int:
    results = _validate.run_checks()
    print(_validate.format_table(results))
    if args.json_path:
        payload = sanitize({
            "results": [r.__dict__ for r in results],
            "failed": _validate.has_failures(results, strict=args.strict),
        })
        Path(args.json_path).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"results written to {args.json_path}")
    return 1 if _validate.has_failures(results, strict=args.strict) else 0


def cmd_compare(args: argparse.Namespace, argv: Optional[List[str]] = None) -> int:
    outdir = Path(args.output)
    git = git_info()
    if args.btype == "particles":
        kw = _kwargs(args, _PARTICLE_MAP)
        metrics = run_shell_crossing_benchmark(outdir, **kw)
        _print_metrics("compare:particles  1D cold collapse, classical vs moss", metrics)
        path = _record(outdir, "compare:particles", metrics.get("params", kw), metrics, argv, git)
    elif args.btype == "fluid3d":
        kw = _kwargs(args, _FLUID_MAP)
        metrics = run_tg3d_benchmark(outdir, **kw)
        _print_metrics("compare:fluid3d  3D Taylor-Green, classical vs moss", metrics)
        path = _record(outdir, "compare:fluid3d", metrics.get("params", kw), metrics, argv, git)
    else:
        kw = _kwargs(args, _FLUID_MAP)
        metrics = run_cfd_benchmark(outdir, **kw)
        _print_metrics("compare:fluid  2D Taylor-Green, classical vs moss", metrics)
        path = _record(outdir, "compare:fluid", metrics.get("params", kw), metrics, argv, git)
    print(f"report merged into {path}")
    checks = metrics.get("checks", {})
    return 0 if all(bool(v) for v in checks.values()) else 1


def cmd_sweep(args: argparse.Namespace, argv: Optional[List[str]] = None) -> int:
    outdir = Path(args.output)
    git = git_info()
    kw = _kwargs(args, _SWEEP_MAP)
    metrics = run_shell_crossing_sweep(outdir, quick=args.quick, **kw)
    _print_metrics("sweep:particles  shell-crossing sweep", metrics)
    path = _record(outdir, "sweep:particles", metrics.get("params", kw), metrics, argv, git)
    print(f"report merged into {path}")
    return 0


def cmd_run_all(args: argparse.Namespace, argv: Optional[List[str]] = None) -> int:
    outdir = Path(args.output)
    git = git_info()          # source-tree state before any output is written
    rc = 0

    metrics = run_decay_benchmark(outdir)
    _print_metrics("decay: analytical damping vs. Euler", metrics)
    _record(outdir, "decay", metrics.get("params", {}), metrics, argv, git)

    metrics = run_shell_crossing_benchmark(outdir)
    _print_metrics("compare:particles  1D cold collapse (physical regime)", metrics)
    _record(outdir, "compare:particles", metrics.get("params", {}), metrics, argv, git)
    rc |= int(not all(bool(v) for v in metrics.get("checks", {}).values()))

    metrics = run_shell_crossing_sweep(outdir, quick=args.quick)
    _print_metrics("sweep:particles  shell-crossing sweep", metrics)
    _record(outdir, "sweep:particles", metrics.get("params", {}), metrics, argv, git)

    metrics = run_cfd_benchmark(outdir)
    _print_metrics("compare:fluid  2D Taylor-Green", metrics)
    _record(outdir, "compare:fluid", metrics.get("params", {}), metrics, argv, git)
    rc |= int(not all(bool(v) for v in metrics.get("checks", {}).values()))

    if not args.no_3d:
        metrics = run_tg3d_benchmark(outdir, n=16 if args.quick else 32, t_max=2.0 if args.quick else 6.0)
        _print_metrics("compare:fluid3d  3D Taylor-Green", metrics)
        _record(outdir, "compare:fluid3d", metrics.get("params", {}), metrics, argv, git)
        rc |= int(not all(bool(v) for v in metrics.get("checks", {}).values()))

    results = _validate.run_checks()
    print(_validate.format_table(results))
    entry = make_entry(
        "validate", {},
        {"failed": _validate.has_failures(results), "results": [r.__dict__ for r in results]},
        [], checks={r.name: r.status for r in results}, argv=argv, git=git,
    )
    write_report(outdir, entry)
    rc |= int(_validate.has_failures(results))

    print(f"\nfigures and report.json written to {outdir.resolve()}/")
    return rc


def cmd_supplement(args: argparse.Namespace) -> int:
    path = write_supplement(
        Path(args.report), Path(args.output), figures_dir=args.figures_dir,
        copy_figures=args.copy_figures, title=args.title, subtitle=args.subtitle,
    )
    print(f"supplement written to {path}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Legacy entry point kept for backwards compatibility."""
    outdir = Path(args.output)
    if args.btype == "decay":
        kw = _kwargs(args, {"t_max": "t_max", "dt": "dt_max", "lambda_eff": "lambda_eff"})
        metrics = run_decay_benchmark(outdir, **kw)
        _print_metrics("decay: analytical damping vs. Euler", metrics)
    elif args.btype == "particles":
        metrics = run_shell_crossing_benchmark(outdir, **_kwargs(args, _PARTICLE_MAP))
        _print_metrics("particles: shell-crossing comparison", metrics)
    else:
        metrics = run_cfd_benchmark(outdir, **_kwargs(args, _FLUID_MAP))
        _print_metrics("fluid: Taylor-Green comparison", metrics)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    try:
        if args.command == "validate":
            return cmd_validate(args)
        if args.command == "compare":
            return cmd_compare(args, argv_list)
        if args.command == "sweep":
            return cmd_sweep(args, argv_list)
        if args.command == "run-all":
            return cmd_run_all(args, argv_list)
        if args.command == "supplement":
            return cmd_supplement(args)
        return cmd_benchmark(args)
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"moss-reg: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
