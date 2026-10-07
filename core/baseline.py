"""Naive rule-based controller: the "business as usual" comparison point.

It reacts only to the current tick, ignores prices and forecasts, and plans against the
*nominal* line limits and battery fleet — so congestion and battery faults surprise it.
"""
from __future__ import annotations

from core.fallback import greedy_action
from core.models import Action, Config, GridState


class NaiveController:
    """Charge on surplus, discharge on deficit, trade the rest with the grid."""

    def __init__(self, config: Config, use_battery: bool = True) -> None:
        self.cfg = config
        self.use_battery = use_battery

    @property
    def name(self) -> str:
        return "baseline" if self.use_battery else "no_battery"

    def decide(self, state: GridState) -> Action:
        """Return the naive dispatch for ``state``."""
        return greedy_action(state, self.cfg, reserve_pct=0.0, respect_live_limits=False,
                             use_battery=self.use_battery, source=self.name)
