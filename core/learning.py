"""LEARN step: online forecast-error learning.

Every tick the agent stores what it forecast for ``lead`` ticks ahead; when that tick arrives
it compares with what actually happened and updates, per resource (solar / wind / demand):

* a multiplicative **bias correction** (EMA of actual ÷ forecast), and
* a **band width factor** so that ~80% of outcomes fall inside P10–P90.

Corrections are applied to future forecasts immediately and persisted to ``data/learning.json``
at the end of each day, so tomorrow starts from today's lessons.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import numpy as np

from core.models import Forecast, GridState

KEYS = ("solar", "wind", "demand")
MIN_SIGNAL_MW = 5.0     # ignore near-zero periods (e.g. solar at night)


class ForecastLearner:
    """Learns forecast bias and calibration online; see module docstring."""

    def __init__(self, lead: int = 4, rate: float = 0.15, path: Optional[Path] = None) -> None:
        self.lead, self.rate, self.path = lead, rate, path
        self.ratio: Dict[str, float] = {k: 1.0 for k in KEYS}
        self.widen: Dict[str, float] = {k: 1.0 for k in KEYS}
        self.days_trained = 0
        self.samples: Dict[str, int] = {k: 0 for k in KEYS}
        self._pending: Dict[int, Dict[str, tuple]] = {}
        self._err_raw: Dict[str, Deque[float]] = {k: deque(maxlen=96) for k in KEYS}
        self._err_fix: Dict[str, Deque[float]] = {k: deque(maxlen=96) for k in KEYS}
        self._inside: Dict[str, Deque[int]] = {k: deque(maxlen=48) for k in KEYS}
        if path is not None and Path(path).exists():
            self.load(Path(path))

    # ------------------------------------------------------------ apply
    def correct(self, fc: Forecast) -> Forecast:
        """Return a copy of ``fc`` with learned bias and band width applied (lead 0 untouched)."""
        new = {}
        for key in KEYS:
            q = getattr(fc, key)
            r, w = self.ratio[key], self.widen[key]
            p50 = q["p50"].copy()
            p50[1:] *= r
            out = {"p50": p50}
            for name in ("p10", "p90"):
                band = q[name] * r - p50
                v = p50 + band * w
                v[0] = q[name][0]
                out[name] = np.clip(v, 0, None)
            if key != "demand":  # never above the forecast physical upper bound
                cap = np.maximum(q["p90"], q["p50"]) * max(1.0, r) * max(1.0, w)
                out = {k: np.minimum(v, cap) for k, v in out.items()}
            new[key] = out
        return replace(fc, **new)

    # ------------------------------------------------------------ learn
    def observe(self, raw: Forecast, corrected: Forecast, state: GridState) -> None:
        """Record the lead-``lead`` forecast now and score any forecast that targeted this tick."""
        actual = {"solar": float(state.solar_mw.sum()), "wind": float(state.wind_mw.sum()),
                  "demand": float(state.demand_mw.sum())}
        done = self._pending.pop(state.tick, None)
        if done:
            for key, (raw_p50, p10, p50, p90) in done.items():
                a = actual[key]
                if raw_p50 < MIN_SIGNAL_MW and a < MIN_SIGNAL_MW:
                    continue
                r = float(np.clip(a / max(raw_p50, 1e-6), 0.85, 1.2))   # robust to one-off shocks
                self.ratio[key] = float(np.clip((1 - self.rate) * self.ratio[key] + self.rate * r, 0.6, 1.4))
                self._err_raw[key].append(abs(raw_p50 - a))
                self._err_fix[key].append(abs(p50 - a))
                self._inside[key].append(int(p10 - 1e-6 <= a <= p90 + 1e-6))
                cov = np.mean(self._inside[key])
                if len(self._inside[key]) >= 8:
                    if cov < 0.7:
                        self.widen[key] = min(1.8, self.widen[key] * 1.02)
                    elif cov > 0.9:
                        self.widen[key] = max(0.6, self.widen[key] * 0.98)
                self.samples[key] += 1
        if raw.horizon > self.lead:
            k = self.lead
            self._pending[raw.start_tick + k] = {
                key: (float(getattr(raw, key)["p50"][k].sum()), float(getattr(corrected, key)["p10"][k].sum()),
                      float(getattr(corrected, key)["p50"][k].sum()), float(getattr(corrected, key)["p90"][k].sum()))
                for key in KEYS}

    # ------------------------------------------------------------ report & persist
    def report(self) -> Dict[str, Dict[str, Any]]:
        """Per resource: learned correction, band coverage and error before/after learning."""
        out = {}
        for k in KEYS:
            raw, fix = list(self._err_raw[k]), list(self._err_fix[k])
            out[k] = {"correction_pct": round((self.ratio[k] - 1) * 100, 1), "band_width": round(self.widen[k], 2),
                      "coverage_pct": round(100 * float(np.mean(self._inside[k])), 0) if self._inside[k] else None,
                      "mae_raw_mw": round(float(np.mean(raw)), 2) if raw else None,
                      "mae_learned_mw": round(float(np.mean(fix)), 2) if fix else None,
                      "samples": self.samples[k]}
        return out

    def end_of_day(self) -> None:
        """Persist lessons for tomorrow."""
        self.days_trained += 1
        if self.path is not None:
            self.save(Path(self.path))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ratio": self.ratio, "widen": self.widen, "days_trained": self.days_trained,
                                    "samples": self.samples}, indent=2))

    def load(self, path: Path) -> None:
        try:
            d = json.loads(path.read_text())
        except (OSError, ValueError):
            return
        self.ratio.update({k: float(v) for k, v in d.get("ratio", {}).items() if k in KEYS})
        self.widen.update({k: float(v) for k, v in d.get("widen", {}).items() if k in KEYS})
        self.days_trained = int(d.get("days_trained", 0))
        self.samples.update({k: int(v) for k, v in d.get("samples", {}).items() if k in KEYS})
