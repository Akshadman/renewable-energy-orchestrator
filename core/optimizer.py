"""Rolling-horizon MILP dispatch optimizer (PuLP + CBC)."""
from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np
import pulp

from core.models import TICK_HOURS, Action, Config, Forecast, GridState

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pulp")

SITUATIONS = ["normal", "price_spike", "storm_risk", "asset_failure", "congestion", "demand_surge"]


@dataclass
class Criteria:
    """Decision criteria chosen by the agent (weights are normalised to sum to 1)."""

    w_cost: float = 0.5
    w_clean: float = 0.3
    w_reliability: float = 0.2
    reserve_pct: float = 0.2
    uncertainty_mode: str = "p50"          # "p50" or "p10" (conservative renewables, P90 demand)
    allow_import: bool = True
    allow_shedding: bool = False
    allow_unserved: bool = False
    max_export_mw: Optional[float] = None

    def normalized(self) -> "Criteria":
        """Clamp inputs and rescale the weights to sum to one."""
        w = np.clip([self.w_cost, self.w_clean, self.w_reliability], 0.0, None)
        w = w / w.sum() if w.sum() > 0 else np.array([0.5, 0.3, 0.2])
        mode = self.uncertainty_mode if self.uncertainty_mode in ("p10", "p50") else "p50"
        return replace(self, w_cost=float(w[0]), w_clean=float(w[1]), w_reliability=float(w[2]),
                       reserve_pct=float(np.clip(self.reserve_pct, 0.0, 0.9)), uncertainty_mode=mode)

    def relaxed(self, attempt: int) -> "Criteria":
        """Progressively looser constraints: attempt 1 lowers reserve & allows import + shedding,
        attempt 2 additionally permits unserved load (always feasible, validator will judge)."""
        c = replace(self, reserve_pct=max(0.0, self.reserve_pct * 0.5), allow_import=True, allow_shedding=True)
        return replace(c, allow_unserved=True, max_export_mw=None) if attempt >= 2 else c

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizerResult:
    """Outcome of one solve. ``action`` is only the first tick of the plan."""

    status: str                          # optimal | infeasible | error
    action: Optional[Action]
    objective: Dict[str, float] = field(default_factory=dict)
    plan: Dict[str, List[float]] = field(default_factory=dict)
    solve_time_s: float = 0.0
    message: str = ""
    criteria: Optional[Criteria] = None


def optimize(state: GridState, forecast: Forecast, config: Config, criteria: Criteria,
             time_limit_s: float = 10.0, soc_target_mwh: Optional[float] = None,
             relax_binaries: bool = False) -> OptimizerResult:
    """Solve the horizon MILP and return the first-tick dispatch. Never raises.

    ``soc_target_mwh`` — day-ahead plan's fleet energy at the end of this horizon (soft target).
    ``relax_binaries`` — solve as an LP (fast; used for the 24 h day-ahead backbone).
    """
    t0 = time.perf_counter()
    try:
        return _solve(state, forecast, config, criteria.normalized(), time_limit_s, t0, soc_target_mwh,
                      relax_binaries)
    except Exception as exc:  # solver crash, bad data — report, don't raise
        return OptimizerResult("error", None, message=f"optimizer error: {exc!r}",
                               solve_time_s=time.perf_counter() - t0, criteria=criteria)


def _solve(st: GridState, fc: Forecast, cfg: Config, cr: Criteria, limit: float, t0: float,
           soc_target: Optional[float] = None, relax_binaries: bool = False) -> OptimizerResult:
    H, dt, m = fc.horizon, TICK_HOURS, cfg.market
    B, S, W, C = range(len(cfg.batteries)), range(len(cfg.solar)), range(len(cfg.wind)), range(len(cfg.consumers))
    T = range(H)
    q_ren = "p10" if cr.uncertainty_mode == "p10" else "p50"
    q_dem = "p90" if cr.uncertainty_mode == "p10" else "p50"
    solar = fc.solar[q_ren].copy()
    wind = fc.wind[q_ren].copy()
    demand = fc.demand[q_dem].copy()
    solar[0], wind[0], demand[0] = st.solar_mw, st.wind_mw, st.demand_mw        # lead 0 = observed
    mcp = fc.price["p50"]
    buy = mcp + m.buy_adder
    sell = mcp * m.sell_factor
    buy[0], sell[0] = st.price_buy, st.price_sell
    crit_cap = np.array([c.base_mw * c.critical_frac for c in cfg.consumers])
    crit = np.minimum(demand, crit_cap)
    flex = demand - crit
    elig = np.array([c.dr_eligible for c in cfg.consumers], dtype=float)
    avail = fc.battery_available.astype(float)
    avail[0] = st.battery_available
    imp_lim = fc.import_limit.copy()
    exp_lim = fc.export_limit.copy()
    imp_lim[0], exp_lim[0] = st.import_limit_mw, st.export_limit_mw
    if cr.max_export_mw is not None:
        exp_lim = np.minimum(exp_lim, cr.max_export_mw)
    if not cr.allow_import:
        imp_lim = np.zeros(H)

    p = pulp.LpProblem("dispatch", pulp.LpMinimize)
    V = pulp.LpVariable
    ch = {(b, t): V(f"ch_{b}_{t}", 0, cfg.batteries[b].power_mw * avail[t, b]) for b in B for t in T}
    dis = {(b, t): V(f"dis_{b}_{t}", 0, cfg.batteries[b].power_mw * avail[t, b]) for b in B for t in T}
    bin_kw = {"lowBound": 0, "upBound": 1} if relax_binaries else {"cat": "Binary"}
    mode = {(b, t): V(f"u_{b}_{t}", **bin_kw) for b in B for t in T}
    soc = {(b, t): V(f"soc_{b}_{t}", cfg.batteries[b].min_mwh, cfg.batteries[b].max_mwh) for b in B for t in T}
    imp = {t: V(f"imp_{t}", 0, imp_lim[t]) for t in T}
    exp = {t: V(f"exp_{t}", 0, exp_lim[t]) for t in T}
    gdir = {t: V(f"g_{t}", **bin_kw) for t in T}
    cs = {(s, t): V(f"cs_{s}_{t}", 0, solar[t, s]) for s in S for t in T}
    cw = {(w, t): V(f"cw_{w}_{t}", 0, wind[t, w]) for w in W for t in T}
    dr = {(c, t): V(f"dr_{c}_{t}", 0, flex[t, c] * elig[c]) for c in C for t in T}
    shed = {(c, t): V(f"shed_{c}_{t}", 0, flex[t, c] if cr.allow_shedding else 0) for c in C for t in T}
    uc = {(c, t): V(f"uc_{c}_{t}", 0, crit[t, c] if cr.allow_unserved else 0) for c in C for t in T}
    rs = {t: V(f"rs_{t}", 0) for t in T}

    for t in T:
        for b in B:
            bat = cfg.batteries[b]
            prev = float(np.clip(st.soc_mwh[b], bat.min_mwh, bat.max_mwh)) if t == 0 else soc[b, t - 1]
            p += soc[b, t] == prev + (ch[b, t] * bat.eff_one_way - dis[b, t] * (1 / bat.eff_one_way)) * dt
            p += ch[b, t] <= bat.power_mw * avail[t, b] * mode[b, t]
            p += dis[b, t] <= bat.power_mw * avail[t, b] * (1 - mode[b, t])
        p += imp[t] <= float(imp_lim[t]) * gdir[t]
        p += exp[t] <= float(exp_lim[t]) * (1 - gdir[t])
        for c in C:
            p += dr[c, t] + shed[c, t] <= float(flex[t, c])
        supply = (pulp.lpSum(float(solar[t, s]) - cs[s, t] for s in S) + pulp.lpSum(float(wind[t, w]) - cw[w, t] for w in W)
                  + pulp.lpSum(dis[b, t] for b in B) + imp[t])
        sink = (pulp.lpSum(float(demand[t, c]) - dr[c, t] - shed[c, t] - uc[c, t] for c in C)
                + pulp.lpSum(ch[b, t] for b in B) + exp[t])
        p += supply == sink, f"balance_{t}"
        floor = sum(max(cr.reserve_pct, cfg.batteries[b].soc_min) * cfg.batteries[b].energy_mwh * avail[t, b] for b in B)
        p += pulp.lpSum(soc[b, t] * avail[t, b] for b in B) + rs[t] >= floor

    ci = cfg.grid.carbon_intensity_t_per_mwh
    terminal_value = float(np.mean(mcp)) * (float(np.mean([b.round_trip_eff for b in cfg.batteries]))
                                            if cfg.batteries else 0.0)
    short = V("terminal_short", 0)
    if soc_target is not None and cfg.batteries:
        p += pulp.lpSum(soc[b, H - 1] for b in B) + short >= float(soc_target)
    cost = (pulp.lpSum(dt * (buy[t] * imp[t] - sell[t] * exp[t]) for t in T)
            + pulp.lpSum(dt * cfg.batteries[b].degradation_cost * (ch[b, t] + dis[b, t]) for b in B for t in T)
            + pulp.lpSum(dt * m.dr_incentive * dr[c, t] + dt * m.voll_flexible * shed[c, t] for c in C for t in T)
            - pulp.lpSum(terminal_value * soc[b, H - 1] for b in B)
            + float(np.mean(buy)) * short)
    clean = (pulp.lpSum(dt * m.curtailment_penalty * (cs[s, t]) for s in S for t in T)
             + pulp.lpSum(dt * m.curtailment_penalty * (cw[w, t]) for w in W for t in T)
             + pulp.lpSum(dt * m.carbon_price * ci * imp[t] for t in T))
    reliability = (pulp.lpSum(dt * m.reserve_shortfall_penalty * rs[t] for t in T)
                   + pulp.lpSum(dt * m.voll_critical * uc[c, t] for c in C for t in T))
    p += cr.w_cost * cost + cr.w_clean * clean + cr.w_reliability * reliability

    status = p.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=limit))
    label = pulp.LpStatus.get(status, "Undefined")
    elapsed = time.perf_counter() - t0
    if label == "Infeasible":
        return OptimizerResult("infeasible", None, solve_time_s=elapsed, criteria=cr,
                               message="No dispatch satisfies the hard constraints (supply < critical demand "
                                       "under current import/shedding permissions).")
    if label != "Optimal":
        return OptimizerResult("error", None, solve_time_s=elapsed, criteria=cr, message=f"solver status {label}")

    val = lambda x: float(pulp.value(x) or 0.0)  # noqa: E731
    action = Action(
        charge_mw=np.array([val(ch[b, 0]) for b in B]), discharge_mw=np.array([val(dis[b, 0]) for b in B]),
        grid_import_mw=val(imp[0]), grid_export_mw=val(exp[0]),
        curtail_solar_mw=np.array([val(cs[s, 0]) for s in S]), curtail_wind_mw=np.array([val(cw[w, 0]) for w in W]),
        dr_mw=np.array([val(dr[c, 0]) for c in C]), shed_mw=np.array([val(shed[c, 0]) for c in C]),
        unserved_critical_mw=np.array([val(uc[c, 0]) for c in C]), source="optimizer",
    )
    _clean_small(action)
    fleet = sum(b.energy_mwh for b in cfg.batteries) or 1.0
    plan = {
        "soc_pct": [sum(val(soc[b, t]) for b in B) / fleet * 100 for t in T],
        "soc_mwh": [sum(val(soc[b, t]) for b in B) for t in T],
        "net_battery_mw": [sum(val(dis[b, t]) - val(ch[b, t]) for b in B) for t in T],
        "net_grid_mw": [val(imp[t]) - val(exp[t]) for t in T],
        "curtail_mw": [sum(val(cs[s, t]) for s in S) + sum(val(cw[w, t]) for w in W) for t in T],
        "dr_mw": [sum(val(dr[c, t]) for c in C) for t in T],
        "shed_mw": [sum(val(shed[c, t]) + val(uc[c, t]) for c in C) for t in T],
    }
    objective = {"cost": pulp.value(cost), "clean": pulp.value(clean), "reliability": pulp.value(reliability),
                 "total": pulp.value(p.objective)}
    return OptimizerResult("optimal", action, {k: float(v or 0.0) for k, v in objective.items()}, plan,
                           time.perf_counter() - t0, "ok", cr)


def _clean_small(a: Action, eps: float = 1e-6) -> None:
    """Zero out solver noise so validators don't see e.g. 1e-9 MW simultaneous flows."""
    for k, v in a.__dict__.items():
        if isinstance(v, np.ndarray):
            v[np.abs(v) < eps] = 0.0
    a.grid_import_mw = 0.0 if abs(a.grid_import_mw) < eps else a.grid_import_mw
    a.grid_export_mw = 0.0 if abs(a.grid_export_mw) < eps else a.grid_export_mw
