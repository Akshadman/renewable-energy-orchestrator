"""Safe rule-based controller used when the LLM or the solver fails. Always respects physics."""
from __future__ import annotations

import numpy as np

from core.models import TICK_HOURS, Action, Config, GridState


def safe_action(state: GridState, config: Config, reserve_pct: float = 0.2) -> Action:
    """Greedy merit order that never violates hard limits.

    Surplus: charge available batteries → export (≤ live limit) → curtail pro-rata.
    Deficit: discharge to reserve → import (≤ live limit) → discharge to SoC min → DR →
    shed flexible → (last resort) unserved critical.
    """
    return greedy_action(state, config, reserve_pct, respect_live_limits=True, use_battery=True, source="fallback")


def greedy_action(state: GridState, config: Config, reserve_pct: float, respect_live_limits: bool,
                  use_battery: bool, source: str) -> Action:
    """Shared greedy dispatch used by the fallback (safe) and the naive baseline (unsafe)."""
    dt = TICK_HOURS
    n_s, n_w, n_b, n_c = len(config.solar), len(config.wind), len(config.batteries), len(config.consumers)
    a = Action.zeros(n_s, n_w, n_b, n_c, source)
    imp_lim = state.import_limit_mw if respect_live_limits else config.grid.import_limit_mw
    exp_lim = state.export_limit_mw if respect_live_limits else config.grid.export_limit_mw
    avail = state.battery_available if respect_live_limits else np.ones(n_b, dtype=bool)
    net = state.renewable_mw - state.total_demand_mw

    def room_charge(b: int) -> float:
        bat = config.batteries[b]
        return max(0.0, min(bat.power_mw, (bat.max_mwh - state.soc_mwh[b]) / (bat.eff_one_way * dt)))

    def room_discharge(b: int, floor_frac: float) -> float:
        bat = config.batteries[b]
        floor = max(floor_frac, bat.soc_min) * bat.energy_mwh
        return max(0.0, min(bat.power_mw - a.discharge_mw[b], (state.soc_mwh[b] - floor) * bat.eff_one_way / dt
                            - a.discharge_mw[b]))

    if net >= 0:
        left = net
        for b in range(n_b if use_battery else 0):
            if avail[b]:
                a.charge_mw[b] = min(left, room_charge(b))
                left -= a.charge_mw[b]
        a.grid_export_mw = min(left, exp_lim)
        left -= a.grid_export_mw
        if left > 0:
            total = state.renewable_mw
            a.curtail_solar_mw = state.solar_mw * left / total
            a.curtail_wind_mw = state.wind_mw * left / total
        return a

    need = -net
    for floor in (reserve_pct, None):
        if floor is None:                    # import before digging below the reserve
            a.grid_import_mw = min(need, imp_lim)
            need -= a.grid_import_mw
            floor_frac = 0.0
        else:
            floor_frac = floor
        for b in range(n_b if use_battery else 0):
            if avail[b] and need > 0:
                d = min(need, room_discharge(b, floor_frac))
                a.discharge_mw[b] += d
                need -= d
    elig = np.array([c.dr_eligible for c in config.consumers], dtype=float)
    for arr, cap in ((a.dr_mw, state.flexible_mw * elig), (a.shed_mw, None), (a.unserved_critical_mw, None)):
        if need <= 1e-9:
            break
        if cap is None:
            cap = state.flexible_mw - a.dr_mw - a.shed_mw if arr is a.shed_mw else state.critical_mw
        take = _pro_rata(need, cap)
        arr += take
        need -= take.sum()
    return a


def _pro_rata(amount: float, caps: np.ndarray) -> np.ndarray:
    caps = np.clip(caps, 0, None)
    total = caps.sum()
    if total <= 0:
        return np.zeros_like(caps)
    return caps * min(1.0, amount / total)
