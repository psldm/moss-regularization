"""Benchmark suites for the moss regularization library.

Each module exposes a ``run_*(outdir, **params)`` entry point that runs a
benchmark, writes a figure into the output directory and returns a
metrics dictionary (with ``params``, ``checks`` and ``figure`` keys used
by ``moss-reg compare`` / ``sweep`` / ``run-all`` for report.json).
"""

from .decay import run_decay_benchmark
from .fluid import run_cfd_benchmark
from .particles import free_fall_reference, run_shell_crossing_benchmark
from .sweep import run_shell_crossing_sweep
from .tg3d import run_tg3d_benchmark
from .tg3d_sweep import run_tg3d_sweep

__all__ = [
    "run_decay_benchmark",
    "run_shell_crossing_benchmark",
    "run_shell_crossing_sweep",
    "run_cfd_benchmark",
    "run_tg3d_benchmark",
    "run_tg3d_sweep",
    "free_fall_reference",
]
