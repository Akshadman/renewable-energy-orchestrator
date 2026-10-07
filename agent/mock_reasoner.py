"""Deterministic offline reasoner: rule-based situation detection, weights and explanations.

Used whenever no API key is configured, for Monte Carlo runs, and as the fallback if an
LLM call fails. It is also the cheap "has the situation changed?" detector that decides
when the LLM is worth consulting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from core.models import Alert, Config, Forecast, GridState
from core.optimizer import Criteria

CRITERIA_TABLE: Dict[str, Tuple[Criteria, str]] = {
    "normal": (Criteria(0.5, 0.3, 0.2, 0.20, "p50"),
               "conditions are normal, so I lean on cost while keeping a 20% battery reserve"),
    "price_spike": (Criteria(0.7, 0.1, 0.2, 0.10, "p50"),
                    "market prices are spiking, so I prioritise cost and let the batteries cover load "
                    "instead of importing expensive power"),
    "storm_risk": (Criteria(0.2, 0.1, 0.7, 0.60, "p10"),
                   "a storm is approaching, so I prioritise reliability, plan on pessimistic (P10) renewables "
                   "and pre-charge the batteries to a 60% reserve"),
    "asset_failure": (Criteria(0.3, 0.2, 0.5, 0.35, "p10"),
                      "an asset is out of service, so I weight reliability and plan conservatively "
                      "until it returns"),
    "congestion": (Criteria(0.3, 0.45, 0.25, 0.25, "p50"),
                   "the grid line is congested, so I favour clean energy use and store surplus "
                   "in the batteries rather than curtailing it"),
    "demand_surge": (Criteria(0.3, 0.2, 0.5, 0.35, "p10"),
                     "demand is surging, so I weight reliability, use demand response where it is cheap "
                     "and keep a healthy reserve"),
}


@dataclass
class Assessment:
    situation: str
    signals: List[str]


class MockReasoner:
    """Rule-based stand-in for the LLM."""

    name = "mock"

    def __init__(self, config: Config) -> None:
        self.cfg = config

    def assess(self, state: GridState, fc: Forecast, alerts: List[Alert], ref_price: float,
               expected_demand_mw: float) -> Assessment:
        """Classify the situation from state, forecast and alerts (priority ordered)."""
        kinds = {a.kind for a in alerts}
        sig: List[str] = []
        nominal_imp = self.cfg.grid.import_limit_mw
        storm = "storm_alert" in kinds
        failure = (not bool(np.all(state.battery_available))) or not bool(fc.battery_available.all())
        surge = "demand_surge" in state.active_events or state.total_demand_mw > 1.12 * expected_demand_mw
        congestion = float(fc.import_limit[:4].min()) < 0.7 * nominal_imp
        near_price = max(state.price_buy, float(fc.price["p50"][:4].max()) + self.cfg.market.buy_adder)
        spike = "price_spike" in state.active_events or near_price > 1.6 * ref_price
        if storm:
            sig.append(next(a.message for a in alerts if a.kind == "storm_alert"))
        if failure:
            off = [b.id for b, ok in zip(self.cfg.batteries, state.battery_available) if not ok]
            sig.append(f"battery offline: {', '.join(off) or 'upcoming'}")
        if surge:
            sig.append(f"demand {state.total_demand_mw:.0f} MW vs expected {expected_demand_mw:.0f} MW")
        if congestion:
            sig.append(f"import limit down to {fc.import_limit[:4].min():.0f} MW")
        if spike:
            sig.append(f"price ₹{near_price:,.0f}/MWh vs typical ₹{ref_price:,.0f}")
        for label, flag in (("storm_risk", storm), ("asset_failure", failure), ("demand_surge", surge),
                            ("congestion", congestion), ("price_spike", spike)):
            if flag:
                return Assessment(label, sig)
        return Assessment("normal", sig)

    def choose_criteria(self, situation: str) -> Tuple[Criteria, str]:
        """Weights, reserve and uncertainty mode for a situation, with the reason."""
        crit, why = CRITERIA_TABLE.get(situation, CRITERIA_TABLE["normal"])
        return Criteria(**crit.to_dict()), why

    def explain(self, situation: str, signals: List[str], criteria: Criteria, outcome: Dict) -> str:
        """Templated 2–3 sentence rationale."""
        _, why = CRITERIA_TABLE.get(situation, CRITERIA_TABLE["normal"])
        seen = signals[0] if signals else "no unusual signals"
        act = outcome.get("summary", "dispatched the optimised plan")
        first = f"Situation: {situation.replace('_', ' ')} ({seen})."
        second = f"Because {why}, I set weights cost {criteria.w_cost:.2f} / clean {criteria.w_clean:.2f} / " \
                 f"reliability {criteria.w_reliability:.2f} with a {criteria.reserve_pct:.0%} reserve " \
                 f"on {criteria.uncertainty_mode.upper()} forecasts."
        third = f"This tick I {act}."
        if outcome.get("fallback"):
            third += " The optimizer could not produce a valid plan, so the safe fallback controller acted."
        return " ".join([first, second, third])
