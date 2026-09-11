"""Command-line interface for moss-regularization benchmarks.

Usage:
    moss-reg run-all [--output DIR]
    moss-reg benchmark --type {decay,particles,fluid} [options]

``run-all`` executes all three benchmark suites and writes the figures
into the output directory (assets/ by default):

    01_velocity_decay.png       analytical damping vs. Euler integration
    02_shell_crossing_arrest.png shell-crossing arrest in 1D collapse
    03_cfd_stability.png        Taylor-Green: classical NS vs. Moss damping
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .benchmarks import run_cfd_benchmark, run_decay_benchmark, run_shell_crossing_benchmark

__all__ = ["main"]


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output", default="assets", help="output directory (default: assets)"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moss-reg",
        description="Physically-grounded nonlinear vacuum damping regularization "
        "benchmarks for CFD and N-body simulations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_all = sub.add_parser("run-all", help="run all three benchmark suites")
    _add_common(run_all)

    bench = sub.add_parser(
        "benchmark", help="run a single benchmark suite with custom parameters"
    )
    bench.add_argument(
        "--type",
        dest="btype",
        required=True,
        choices=["decay", "particles", "fluid"],
        help="benchmark suite to run",
    )
    _add_common(bench)
    bench.add_argument("--n", type=int, default=None, help="resolution (particles / grid)")
    bench.add_argument(
        "--lambda", dest="lambda_eff", type=float, default=None,
        help="damping coupling (decay/fluid)",
    )
    bench.add_argument(
        "--c", type=float, default=None,
        help="speed of light in code units, lambda_i = G rho_i / c^3 (particles)",
    )
    bench.add_argument("--t-max", type=float, default=None, help="integration horizon")
    bench.add_argument("--dt", type=float, default=None, help="maximum timestep")
    bench.add_argument(
        "--re", type=float, default=None, help="Reynolds number (fluid)"
    )
    bench.add_argument(
        "--softening", type=float, default=None, help="Plummer softening (particles)"
    )
    return parser


def _report(title: str, metrics: dict) -> None:
    print(f"[{title}]")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:28s} {value:.6g}")
        else:
            print(f"  {key:28s} {value}")


def run_benchmark(args: argparse.Namespace) -> int:
    outdir = Path(args.output)
    kwargs = {
        k: v
        for k, v in (
            ("n", args.n),
            ("t_max", args.t_max),
            ("dt_max", args.dt),
        )
        if v is not None
    }
    if args.btype == "decay":
        if args.lambda_eff is not None:
            kwargs["lambda_eff"] = args.lambda_eff
        metrics = run_decay_benchmark(outdir, **kwargs)
        _report("decay: analytical damping vs. Euler", metrics)
    elif args.btype == "particles":
        if args.c is not None:
            kwargs["c"] = args.c
        if args.softening is not None:
            kwargs["softening"] = args.softening
        metrics = run_shell_crossing_benchmark(outdir, **kwargs)
        _report("particles: shell-crossing arrest", metrics)
    else:  # fluid
        if args.lambda_eff is not None:
            kwargs["lam"] = args.lambda_eff
        if args.re is not None:
            kwargs["re"] = args.re
        metrics = run_cfd_benchmark(outdir, **kwargs)
        _report("fluid: Taylor-Green stability", metrics)
    return 0


def run_all(args: argparse.Namespace) -> int:
    outdir = Path(args.output)
    metrics = run_decay_benchmark(outdir)
    _report("decay: analytical damping vs. Euler", metrics)
    metrics = run_shell_crossing_benchmark(outdir)
    _report("particles: shell-crossing arrest", metrics)
    metrics = run_cfd_benchmark(outdir)
    _report("fluid: Taylor-Green stability", metrics)
    print(f"\nfigures written to {outdir.resolve()}/")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run-all":
            return run_all(args)
        return run_benchmark(args)
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"moss-reg: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
