"""STRESS-TEST: replay each candidate plan across many possible futures.

Futures are sampled from the P10/P50/P90 forecasts (solar, wind, demand, price) and from storm
alerts *with their probabilities* — a 35% storm hits in ~35% of futures. In each future the plan's
first tick is executed exactly, then a recovery policy follows the plan and absorbs surprises
(battery → grid → curtail / shed), the way the real agent would by re-planning each tick.

Returns per plan: average ₹, worst-case (P95) ₹, P(critical shortfall), expected CO₂ etc.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from core.events import STORM_EFFECT
from core.models import TICK_HOURS, Action, Config, Forecast, GridState

Z90 = 1.2816


@dataclass
class StressResult:
    cost_avg: float
    cost_p95: float
    p_shortfall: float          # P(any critical load unserved in the look-ahead)
    unserved_critical_mwh: float
    shed_flexible_mwh: float
    co2_t: float
    curtailed_mwh: float
    battery_throughput_mwh: float

    def to_dict(self) -> Dict[str, float]:
        return {k: round(float(v), 4) for k, v in asdict(self).items()}


@dataclass
class Futures:
    """Sampled scenarios, arrays shaped (N, H)."""

    solar: np.ndarray
    wind: np.ndarray
    demand: np.ndarray
    critical: np.ndarray
    buy: np.ndarray
    sell: np.ndarray
    imp_lim: np.ndarray
    exp_lim: np.ndarray
    storm_hit: np.ndarray       # (N,) bool — any storm in this future


def sample_futures(state: GridState, fc: Forecast, cfg: Config, storms: List[tuple], n: int = 100,
                   seed: int = 0) -> Futures:
    """Draw ``n`` correlated futures from the forecast quantiles and storm probabilities."""
    rng = np.random.default_rng(seed)
    H = fc.horizon

    def draw(q: Dict[str, np.ndarray]) -> np.ndarray:
        p10, p50, p90 = (np.asarray(q[k]).reshape(H, -1).sum(1) for k in ("p10", "p50", "p90"))
        std = (p90 - p10) / (2 * Z90)
        z = rng.normal(0, 1, (n, 1)) * 0.8 + np.cumsum(rng.normal(0, 0.12, (n, H)), axis=1)
        return np.clip(p50[None, :] + z * std[None, :], 0, None)

    solar, wind, demand = draw(fc.solar), draw(fc.wind), draw(fc.demand)
    price = draw({k: np.asarray(v)[:, None] for k, v in fc.price.items()})
    crit_cap = sum(c.base_mw * c.critical_frac for c in cfg.consumers)
    imp = np.tile(fc.import_limit, (n, 1)).astype(float)
    exp = np.tile(fc.export_limit, (n, 1)).astype(float)
    # lead 0 is observed, not uncertain
    solar[:, 0], wind[:, 0], demand[:, 0] = state.solar_mw.sum(), state.wind_mw.sum(), state.total_demand_mw
    imp[:, 0], exp[:, 0] = state.import_limit_mw, state.export_limit_mw
    hit_any = np.zeros(n, dtype=bool)
    for prob, eta, dur in storms:
        hit = rng.random(n) < prob
        hit_any |= hit
        a, b = max(1, eta), min(H, eta + dur)
        if a < b:
            sl = (hit[:, None]) & (np.arange(H)[None, :] >= a) & (np.arange(H)[None, :] < b)
            wind = np.where(sl, wind * STORM_EFFECT["wind"], wind)
            solar = np.where(sl, solar * STORM_EFFECT["solar"], solar)
            imp = np.where(sl, imp * STORM_EFFECT["line"], imp)
            exp = np.where(sl, exp * STORM_EFFECT["line"], exp)
    m = cfg.market
    price[:, 0] = state.price_buy - m.buy_adder
    return Futures(solar, wind, demand, np.minimum(demand, crit_cap), price + m.buy_adder, price * m.sell_factor,
                   imp, exp, hit_any)


def stress_test(first: Action, plan: Dict[str, List[float]], state: GridState, fc: Forecast, cfg: Config,
                futures: Futures, value_per_mwh: Optional[float] = None) -> StressResult:
    """Simulate ``plan`` (first tick = ``first``) in every future and summarise outcomes."""
    dt, m, H = TICK_HOURS, cfg.market, fc.horizon
    n = futures.solar.shape[0]
    bats = cfg.batteries
    avail = fc.battery_available.reshape(H, -1)
    p_max = np.array([sum(b.power_mw for b, ok in zip(bats, avail[t]) if ok) for t in range(H)])
    e_min = sum(b.min_mwh for b in bats)
    e_max = sum(b.max_mwh for b in bats)
    eff = float(np.mean([b.eff_one_way for b in bats])) if bats else 1.0
    deg = float(np.mean([b.degradation_cost for b in bats])) if bats else 0.0
    soc0 = float(state.soc_mwh.sum())
    soc = np.full(n, soc0)
    planned_b = np.array(plan["net_battery_mw"], dtype=float)
    planned_g = np.array(plan["net_grid_mw"], dtype=float)
    planned_c = np.array(plan["curtail_mw"], dtype=float)
    planned_dr = np.array(plan["dr_mw"], dtype=float)
    planned_shed = np.array(plan["shed_mw"], dtype=float)
    planned_b[0] = first.discharge_mw.sum() - first.charge_mw.sum()
    planned_g[0] = first.grid_import_mw - first.grid_export_mw
    planned_c[0] = first.curtail_solar_mw.sum() + first.curtail_wind_mw.sum()
    planned_dr[0] = first.dr_mw.sum()
    planned_shed[0] = first.shed_mw.sum() + first.unserved_critical_mw.sum()

    cost = np.zeros(n)
    co2 = np.zeros(n)
    curt = np.zeros(n)
    unserved_crit = np.zeros(n)
    shed = np.zeros(n)
    thru = np.zeros(n)
    ci = cfg.grid.carbon_intensity_t_per_mwh
    for t in range(H):
        renew = futures.solar[:, t] + futures.wind[:, t]
        used = np.maximum(renew - planned_c[t], 0)
        load = np.maximum(futures.demand[:, t] - planned_dr[t] - planned_shed[t], 0)
        # battery: planned power + absorbs the deviation, within power & energy limits
        lo = -min(p_max[t], np.inf) * np.ones(n)
        hi = p_max[t] * np.ones(n)
        lo = np.maximum(lo, -(e_max - soc) / (eff * dt))       # charging limited by headroom
        hi = np.minimum(hi, (soc - e_min) * eff / dt)          # discharging limited by energy
        lo, hi = np.minimum(lo, 0), np.maximum(hi, 0)
        b = np.clip(planned_b[t], lo, hi)                       # battery follows its plan where it can
        g = np.clip(load - used - b, -futures.exp_lim[:, t], futures.imp_lim[:, t])   # grid absorbs surprises
        resid = load - used - b - g                             # >0 deficit, <0 surplus
        extra = np.clip(resid, lo - b, hi - b)                  # battery covers what the grid can't
        b, resid = b + extra, resid - extra
        deficit = np.maximum(resid, 0)
        flex = np.maximum(load - futures.critical[:, t], 0)
        shed_t = np.minimum(deficit, flex) + planned_shed[t]
        crit_t = np.maximum(deficit - flex, 0)
        curt_t = np.maximum(-resid, 0) + np.minimum(planned_c[t], renew)
        soc = soc + np.where(b < 0, -b * eff, -b / eff) * dt
        imp, exp = np.maximum(g, 0), np.maximum(-g, 0)
        cost += dt * (imp * futures.buy[:, t] - exp * futures.sell[:, t] + deg * np.abs(b)
                      + m.dr_incentive * planned_dr[t] + m.voll_flexible * shed_t + m.voll_critical * crit_t)
        co2 += imp * dt * ci
        curt += curt_t * dt
        shed += shed_t * dt
        unserved_crit += crit_t * dt
        thru += np.abs(b) * dt
    value = value_per_mwh if value_per_mwh is not None else float(np.mean(fc.price["p50"])) * eff ** 2
    cost -= value * (soc - soc0)          # energy left in the batteries is worth something tomorrow
    return StressResult(float(cost.mean()), float(np.percentile(cost, 95)), float(np.mean(unserved_crit > 0.05)),
                        float(unserved_crit.mean()), float(shed.mean()), float(co2.mean()), float(curt.mean()),
                        float(thru.mean()))
