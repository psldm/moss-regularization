"""Reproducibility reports for benchmark and comparison runs.

Every ``moss-reg compare`` / ``sweep`` / ``run-all`` invocation writes (or
merges into) ``<outdir>/report.json`` an entry containing the package
version, git commit (and dirty flag), timestamp, environment, the exact
parameters, the metrics and the PASS/FAIL checks of the run, plus the
figures it produced.  Entries are keyed by ``command`` so repeated runs
into the same directory replace their own entry only.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import __version__

__all__ = ["git_info", "environment", "make_entry", "write_report", "sanitize"]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(args: List[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_info(cwd: Optional[Path] = None) -> Dict[str, Any]:
    """Commit SHA and dirty flag of the repository containing the package
    (falls back to ``cwd``); ``None`` values when git is unavailable."""
    for root in (cwd or _REPO_ROOT, Path.cwd()):
        sha = _git(["rev-parse", "HEAD"], root)
        if sha:
            status = _git(["status", "--porcelain"], root)
            return {
                "commit": sha,
                "dirty": bool(status) if status is not None else None,
                "root": str(root),
            }
    return {"commit": None, "dirty": None, "root": None}


def environment() -> Dict[str, Any]:
    import matplotlib
    import scipy

    return {
        "moss_reg": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
    }


def sanitize(obj: Any) -> Any:
    """Convert numpy scalars/arrays and NaN/inf into plain JSON values."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(obj, Path):
        return str(obj)
    return obj


def make_entry(
    command: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    figures: List[str],
    checks: Optional[Dict[str, Any]] = None,
    argv: Optional[List[str]] = None,
    git: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a report entry.  Pass ``git`` captured *before* the run writes
    any output so the dirty flag describes the source tree, not the
    figures being (re)generated."""
    return sanitize(
        {
            "command": command,
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "argv": list(argv) if argv is not None else sys.argv[1:],
            "git": git if git is not None else git_info(),
            "environment": environment(),
            "params": params,
            "metrics": metrics,
            "checks": checks or {},
            "figures": figures,
        }
    )


def write_report(outdir: Path, entry: Dict[str, Any], filename: str = "report.json") -> Path:
    """Merge ``entry`` into ``outdir/filename`` (keyed by ``entry['command']``)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    data: Dict[str, Any] = {"schema": "moss-reg-report/1", "runs": []}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
                data = loaded
        except (OSError, ValueError):
            pass
    data["runs"] = [r for r in data["runs"] if r.get("command") != entry["command"]]
    data["runs"].append(entry)
    data["generated_utc"] = entry["timestamp_utc"]
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
    return path
