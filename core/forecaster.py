"""Probabilistic (P10/P50/P90) forecasts: future truth + autocorrelated error growing with horizon."""
from __future__ import annotations

from typing import Dict

import numpy as np

from core.models import Forecast
from core.simulator import Simulator

Z90 = 1.2816  # standard-normal 90th percentile

# per-step error std (fraction); total std at lead k is sigma * sqrt(k)
SIGMA = {"solar": 0.035, "wind": 0.05, "demand": 0.012, "price": 0.03}


class Forecaster:
    """Generates quantile forecasts from the digital twin's (event-adjusted) truth."""

    def __init__(self, sim: Simulator, sigma: Dict[str, float] = None) -> None:
        self.sim = sim
        self.sigma = {**SIGMA, **(sigma or {})}

    def forecast(self, horizon: int = 16) -> Forecast:
        """Forecast ``horizon`` ticks starting at the current tick (lead 0 = nowcast = state)."""
        sim, t0 = self.sim, self.sim.tick
        rng = np.random.default_rng(sim.seed * 100_003 + t0)
        ticks = range(t0, t0 + horizon)
        solar = np.array([sim.solar_at(t) for t in ticks])
        wind = np.array([sim.wind_at(t) for t in ticks])
        demand = np.array([sim.demand_at(t) for t in ticks])
        price = np.array([sim.mcp_at(t) for t in ticks])
        cap_s = np.array([f.capacity_mw for f in sim.cfg.solar])
        cap_w = np.array([f.capacity_mw for f in sim.cfg.wind])
        return Forecast(
            start_tick=t0, horizon=horizon,
            solar=self._quantiles(solar, "solar", rng, cap_s),
            wind=self._quantiles(wind, "wind", rng, cap_w),
            demand=self._quantiles(demand, "demand", rng, None),
            price=self._quantiles(price[:, None], "price", rng, None, lo=0.0),
            critical_frac=np.array([c.critical_frac for c in sim.cfg.consumers]),
            battery_available=np.array([sim.battery_available_at(t) for t in ticks]),
            import_limit=np.array([sim.import_limit_at(t) for t in ticks]),
            export_limit=np.array([sim.export_limit_at(t) for t in ticks]),
        )

    def _quantiles(self, truth: np.ndarray, key: str, rng: np.random.Generator,
                   cap: np.ndarray = None, lo: float = 0.0) -> Dict[str, np.ndarray]:
        H = truth.shape[0]
        s = self.sigma[key]
        lead_std = s * np.sqrt(np.arange(H))[:, None]                      # (H,1), zero at lead 0
        walk = np.cumsum(rng.normal(0, s, truth.shape), axis=0)
        walk -= walk[0]                                                    # nowcast is exact
        p50 = truth * (1 + walk)
        p10 = p50 * (1 - Z90 * lead_std)
        p90 = p50 * (1 + Z90 * lead_std)
        hi = cap if cap is not None else np.inf
        out = {k: np.clip(v, lo, hi) for k, v in (("p10", p10), ("p50", p50), ("p90", p90))}
        if truth.shape[1] == 1 and key == "price":
            out = {k: v[:, 0] for k, v in out.items()}
        return out
