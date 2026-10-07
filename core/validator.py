"""Physics guardrails. Every action is checked here before it reaches the grid."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from core.models import TICK_HOURS, Action, Config, GridState

TOL = 1e-3


@dataclass
class ValidationResult:
    passed: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "violations": self.violations, "warnings": self.warnings}


def validate(action: Action, state: GridState, config: Config, balance_tol_mw: float = 0.5) -> ValidationResult:
    """Hard checks: SoC bounds, rate limits, line capacity, power balance, critical load, availability."""
    v: List[str] = []
    w: List[str] = []
    a, dt = action, TICK_HOURS
    arrays = {"charge": a.charge_mw, "discharge": a.discharge_mw, "curtail_solar": a.curtail_solar_mw,
              "curtail_wind": a.curtail_wind_mw, "dr": a.dr_mw, "shed": a.shed_mw,
              "unserved_critical": a.unserved_critical_mw}
    for name, arr in arrays.items():
        if not np.all(np.isfinite(arr)) or np.any(arr < -TOL):
            v.append(f"{name} has negative or non-finite values")
    if not np.isfinite([a.grid_import_mw, a.grid_export_mw]).all() or min(a.grid_import_mw, a.grid_export_mw) < -TOL:
        v.append("grid flows negative or non-finite")

    for b, bat in enumerate(config.batteries):
        ch, dis = a.charge_mw[b], a.discharge_mw[b]
        if not state.battery_available[b] and (ch > TOL or dis > TOL):
            v.append(f"{bat.id} is unavailable but dispatched")
        if ch > bat.power_mw + TOL or dis > bat.power_mw + TOL:
            v.append(f"{bat.id} exceeds {bat.power_mw} MW rate limit")
        if ch > TOL and dis > TOL:
            v.append(f"{bat.id} charges and discharges simultaneously")
        soc_next = state.soc_mwh[b] + (ch * bat.eff_one_way - dis / bat.eff_one_way) * dt
        if soc_next < bat.min_mwh - TOL or soc_next > bat.max_mwh + TOL:
            v.append(f"{bat.id} SoC would reach {soc_next / bat.energy_mwh:.1%} "
                     f"(limits {bat.soc_min:.0%}–{bat.soc_max:.0%})")

    if a.grid_import_mw > state.import_limit_mw + TOL:
        v.append(f"import {a.grid_import_mw:.1f} MW exceeds line limit {state.import_limit_mw:.1f} MW")
    if a.grid_export_mw > state.export_limit_mw + TOL:
        v.append(f"export {a.grid_export_mw:.1f} MW exceeds line limit {state.export_limit_mw:.1f} MW")
    if a.grid_import_mw > TOL and a.grid_export_mw > TOL:
        v.append("simultaneous import and export")
    if np.any(a.curtail_solar_mw > state.solar_mw + TOL) or np.any(a.curtail_wind_mw > state.wind_mw + TOL):
        v.append("curtailment exceeds available generation")

    elig = np.array([c.dr_eligible for c in config.consumers])
    if np.any(a.dr_mw[~elig] > TOL):
        v.append("demand response on ineligible consumer")
    if np.any(a.dr_mw + a.shed_mw > state.flexible_mw + TOL):
        v.append("DR + shedding exceeds flexible load")
    if a.unserved_critical_mw.sum() > TOL:
        v.append(f"critical load not served ({a.unserved_critical_mw.sum():.2f} MW)")

    gap = a.supply_mw(state) - a.sink_mw(state)
    if abs(gap) > balance_tol_mw:
        v.append(f"power balance off by {gap:+.2f} MW")

    if a.shed_mw.sum() > TOL:
        w.append(f"sheds {a.shed_mw.sum():.1f} MW of flexible load")
    reserve = sum(state.soc_mwh) / sum(b.energy_mwh for b in config.batteries)
    if reserve < 0.15:
        w.append(f"fleet SoC low ({reserve:.0%})")
    return ValidationResult(not v, v, w)
