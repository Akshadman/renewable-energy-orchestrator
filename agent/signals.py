"""DIAGNOSE: turn everything the agent perceives into continuous risk signals — all at once.

No situation labels, no priority order: every signal is a number, every signal is kept, and they
all flow into the stress test and the choice together. ``headline`` only *describes* the picture
for humans (it does not drive the decision).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.models import Alert, Config, Forecast, GridState


@dataclass
class Signals:
    storm_prob: float = 0.0            # probability a storm hits within the look-ahead
    storm_eta_min: Optional[int] = None
    storm_duration_ticks: int = 0
    price_ratio: float = 1.0           # near-term peak price ÷ typical price
    line_headroom: float = 1.0         # min import capacity in next hour ÷ nominal
    battery_available: float = 1.0     # share of battery fleet in service
    generation_offline: float = 0.0    # share of generation capacity broken / under maintenance
    demand_deviation: float = 0.0      # demand now vs expected (+0.10 = 10% above)
    renewable_deviation: float = 0.0   # renewables now vs day-ahead expectation
    battery_soc: float = 0.5           # fleet state of charge (0–1)
    carbon_pressure: float = 1.0       # CO₂ so far ÷ prorated daily budget (>1 = over budget)
    cost_pressure: float = 1.0         # ₹/MWh so far ÷ target
    storms: List[Tuple[float, int, int]] = field(default_factory=list)   # (prob, eta_ticks, duration)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("storms")
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}

    def headline(self) -> List[Tuple[str, str]]:
        """Human-readable tags for every signal that is out of its normal range: (text, level)."""
        tags: List[Tuple[str, str]] = []
        if self.storm_prob >= 0.05:
            lvl = "critical" if self.storm_prob >= 0.5 else "warning"
            tags.append((f"Storm risk {self.storm_prob:.0%} in {self.storm_eta_min} min", lvl))
        if self.price_ratio >= 1.4:
            tags.append((f"Price {self.price_ratio:.1f}× normal", "warning"))
        elif self.price_ratio <= 0.7:
            tags.append((f"Cheap power ({self.price_ratio:.1f}× normal)", "info"))
        if self.line_headroom < 0.8:
            tags.append((f"Grid line limited to {self.line_headroom:.0%}", "warning"))
        if self.battery_available < 1:
            tags.append((f"Batteries in service {self.battery_available:.0%}", "critical"))
        if self.generation_offline > 0.01:
            tags.append((f"{self.generation_offline:.0%} of generation offline", "warning"))
        if abs(self.demand_deviation) >= 0.08:
            tags.append((f"Demand {self.demand_deviation:+.0%} vs expected", "warning"))
        if self.renewable_deviation <= -0.2:
            tags.append((f"Renewables {self.renewable_deviation:+.0%} vs plan", "warning"))
        if self.carbon_pressure > 1.15:
            tags.append((f"CO₂ {self.carbon_pressure:.0%} of budget pace", "warning"))
        return tags or [("All signals normal", "ok")]


def diagnose(state: GridState, fc: Forecast, alerts: List[Alert], cfg: Config, ref_price: float,
             expected_demand: float, expected_renewable: Optional[float], offline_frac: float,
             co2_so_far: float, cost_so_far: float, energy_so_far: float, elapsed_ticks: int) -> Signals:
    """Compute every signal from the current perception."""
    s = Signals()
    storms = [(float(a.probability), int(a.eta_ticks), int(max(1, a.duration_ticks)))
              for a in alerts if a.kind == "storm_alert" and a.eta_ticks < fc.horizon]
    if storms:
        s.storms = storms
        s.storm_prob = float(1 - np.prod([1 - p for p, _, _ in storms]))      # any storm hits
        first = min(storms, key=lambda x: x[1])
        s.storm_eta_min, s.storm_duration_ticks = first[1] * 15, first[2]
    near = float(fc.price["p50"][:4].max()) + cfg.market.buy_adder
    s.price_ratio = max(state.price_buy, near) / max(ref_price, 1.0)
    nominal = cfg.grid.import_limit_mw or 1.0
    s.line_headroom = float(min(state.import_limit_mw, fc.import_limit[:4].min())) / nominal
    s.battery_available = float(np.mean(state.battery_available)) if len(state.battery_available) else 1.0
    s.generation_offline = offline_frac
    s.demand_deviation = state.total_demand_mw / max(expected_demand, 1.0) - 1
    if expected_renewable and expected_renewable > 10:
        s.renewable_deviation = state.renewable_mw / expected_renewable - 1
    cap = sum(b.energy_mwh for b in cfg.batteries)
    s.battery_soc = float(state.soc_mwh.sum() / cap) if cap else 0.0
    daily_budget = float(cfg.targets.get("annual_carbon_target_t", 0)) / 365
    if daily_budget > 0 and elapsed_ticks >= 24:     # judge pace only after 6 h of data
        s.carbon_pressure = co2_so_far / (daily_budget * elapsed_ticks / 96)
    target = float(cfg.targets.get("target_cost_rs_per_mwh", 0))
    if target > 0 and energy_so_far > 1:
        s.cost_pressure = (cost_so_far / energy_so_far) / target
    return s
