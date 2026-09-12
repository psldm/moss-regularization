"""1D cold collapse: classical vs. vacuum damping, one comparison.

Thin wrapper around ``moss_reg.benchmarks.run_shell_crossing_benchmark``
(the same code path as ``moss-reg compare --type particles``).

Code units: G = M = L = 1.  The physical dial is the compactness
G M / (L c^2) = 1 / c^2; physical bodies have compactness <= 0.5, so the
default 0.1 (c ~ 3.2, v_ff / c ~ 0.5) is in the v < c regime, where the
damping leaves the collapse essentially unchanged.  Pass ``c=0.005``
(compactness 4e4) to reproduce the strong-coupling figure of release
0.1.0, in which the collapse is superluminal in the model's own units
and the damping acts as a speed limiter near c.  See
``moss-reg sweep`` for the full (c, N, dt, h, softening) picture.

Saves 02_shell_crossing_arrest.png in the current directory.
"""

from __future__ import annotations

import sys

from moss_reg.benchmarks import run_shell_crossing_benchmark


def main() -> None:
    c = float(sys.argv[1]) if len(sys.argv) > 1 else None
    metrics = run_shell_crossing_benchmark(outdir=".", n=200, compactness=0.1, c=c)
    for key in (
        "c", "compactness", "regime", "v_ff_over_c",
        "crossing_time_classical", "crossing_time_moss", "vpeak_moss_over_c",
    ):
        print(f"{key:26s} {metrics[key]}")
    print(f"saved {metrics['figure']}")


if __name__ == "__main__":
    main()
