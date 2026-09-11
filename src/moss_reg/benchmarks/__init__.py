"""Automated benchmark suites for the moss regularization library.

Each module exposes a ``run_*_benchmark(outdir, **params)`` entry point
that runs a benchmark, writes a publication-quality figure into the
output directory, and returns a metrics dictionary.
"""

from .decay import run_decay_benchmark
from .fluid import run_cfd_benchmark
from .particles import run_shell_crossing_benchmark

__all__ = [
    "run_decay_benchmark",
    "run_shell_crossing_benchmark",
    "run_cfd_benchmark",
]
