"""Operation = the AI-controlled digital twin + a baseline twin running the same day in lockstep.

The baseline (naive rule-based controller) sees identical weather, demand, prices and shocks, so
"money saved" and "carbon saved" are measured live against what business-as-usual would have done.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from agent.agent import OrchestratorAgent
from core.baseline import NaiveController
from core.models import TICK_HOURS, Config, Event
from core.simulator import Simulator


class Operation:
    """Owns the AI twin, the baseline twin and the shared event stream."""

    def __init__(self, cfg: Config, sim: Simulator, use_llm: Optional[bool] = None, persist_learning: bool = True,
                 log_path: Any = "default") -> None:
        self.cfg = cfg
        self.sim = sim
        kw = {} if log_path == "default" else {"log_path": log_path}
        self.agent = OrchestratorAgent(sim, cfg, use_llm=use_llm, persist_learning=persist_learning, **kw)
        self.base_sim = Simulator(cfg, seed=sim.seed, weather=copy.deepcopy(sim.weather), day=sim.day)
        self.baseline = NaiveController(cfg)

    @property
    def done(self) -> bool:
        return self.sim.done

    def inject(self, event: Event) -> Event:
        """Apply the same shock to both twins."""
        twin = copy.deepcopy(event)
        ev = self.sim.apply_event(event)
        twin.params = dict(ev.params)          # same hidden outcome (e.g. whether the storm hits)
        self.base_sim.apply_event(twin)
        return ev

    def step(self):
        """Advance both twins one tick."""
        d = self.agent.step()
        if not self.base_sim.done:
            self.base_sim.step(self.baseline.decide(self.base_sim.state()))
        return d

    def outputs(self) -> Dict[str, float]:
        """The headline outcomes: energy, money & carbon saved vs baseline, efficiency, clean share."""
        ai, base = self.sim.totals(), self.base_sim.totals()
        h = self.sim.history
        dt = TICK_HOURS
        solar = sum(r["solar_mw"] for r in h) * dt
        wind = sum(r["wind_mw"] for r in h) * dt
        avail = sum(r["solar_avail_mw"] + r["wind_avail_mw"] for r in h) * dt
        battery = sum(r["discharge_mw"] for r in h) * dt
        return {
            "solar_mwh": solar, "wind_mwh": wind, "battery_mwh": battery, "grid_mwh": ai["import_mwh"],
            "energy_produced_mwh": solar + wind,
            "money_saved_rs": base["cost_rs"] - ai["cost_rs"],
            "carbon_saved_t": base["co2_t"] - ai["co2_t"],
            "generation_efficiency_pct": 100 * (solar + wind) / avail if avail > 0 else 100.0,
            "clean_pct": ai["clean_pct"], "baseline_clean_pct": base["clean_pct"],
            "cost_rs": ai["cost_rs"], "baseline_cost_rs": base["cost_rs"],
            "co2_t": ai["co2_t"], "baseline_co2_t": base["co2_t"],
            "unserved_mwh": ai["unserved_mwh"], "baseline_unserved_mwh": base["unserved_mwh"],
            "violations": ai["violations"], "baseline_violations": base["violations"],
            "curtailed_mwh": ai["curtailed_mwh"],
        }
