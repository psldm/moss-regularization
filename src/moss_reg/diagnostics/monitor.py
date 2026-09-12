"""Per-step diagnostics recorder with CSV / JSON export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .energy import EnergyBudget

__all__ = ["DiagnosticsLog"]


class DiagnosticsLog:
    """Append-only table of per-step diagnostics.

    >>> log = DiagnosticsLog()
    >>> log.record(0.0, **solver.diagnostics())
    >>> log.series("E"), log.t
    >>> log.write_csv("run.csv"); log.write_json("run.json")
    >>> log.budget(nu=1e-3, lam=0.5).summary()
    """

    def __init__(self) -> None:
        self._rows: List[Dict[str, Any]] = []
        self._keys: List[str] = ["t"]

    def record(self, t: float, **values: Any) -> None:
        row = {"t": float(t)}
        for k, v in values.items():
            if isinstance(v, (np.ndarray, list, tuple)):
                continue                       # scalars only
            row[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
            if k not in self._keys:
                self._keys.append(k)
        self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def keys(self) -> List[str]:
        return list(self._keys)

    @property
    def t(self) -> np.ndarray:
        return self.series("t")

    def series(self, key: str) -> np.ndarray:
        return np.asarray([r.get(key, np.nan) for r in self._rows], dtype=np.float64)

    def last(self) -> Dict[str, Any]:
        return dict(self._rows[-1]) if self._rows else {}

    def as_dict(self) -> Dict[str, list]:
        return {k: [r.get(k) for r in self._rows] for k in self._keys}

    def write_csv(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self._keys)
            w.writeheader()
            for r in self._rows:
                w.writerow({k: r.get(k, "") for k in self._keys})
        return path

    def write_json(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=1))
        return path

    @classmethod
    def read_csv(cls, path: Path | str) -> "DiagnosticsLog":
        log = cls()
        with Path(path).open() as fh:
            for row in csv.DictReader(fh):
                t = float(row.pop("t"))
                log.record(t, **{k: (float(v) if v not in ("", None) else np.nan) for k, v in row.items()})
        return log

    def budget(self, nu: float, lam: Optional[float] = None,
               grad_key: str = "grad_l2_sq", vac_key: str = "vac_power") -> EnergyBudget:
        """Energy budget from the recorded series.

        ``vac_power`` is used if present; otherwise ``lam * l4_pow4``.
        """
        if vac_key in self._keys:
            vac = self.series(vac_key)
        elif lam is not None and "l4_pow4" in self._keys:
            vac = lam * self.series("l4_pow4")
        else:
            raise KeyError("need a 'vac_power' series or lam and 'l4_pow4'")
        return EnergyBudget(self.t, self.series("E"), nu * self.series(grad_key), vac)
