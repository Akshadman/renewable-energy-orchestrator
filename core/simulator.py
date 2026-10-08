"""Digital twin of the portfolio, advanced in 15-minute ticks."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import Any, Dict, List, Optional

import numpy as np

from core.config import resolve_path
from core.data_sources import WeatherDay, load_iex_prices, load_weather_day, today_ist, weather_alerts
from core.models import TICK_HOURS, Action, Alert, Config, Event, GridState, TickKPI
from core.power_curves import solar_mw, wind_mw

TOL = 1e-3


@dataclass
class MaintenanceRequest:
    """A maintenance job that must be scheduled (agent may move it within a window)."""

    id: str
    asset_id: str
    requested_start: int
    duration: int
    window: int
    scheduled_start: Optional[int] = None


@dataclass
class DayProfile:
    """Undisturbed ("truth") profiles before events are applied."""

    solar: np.ndarray       # (T, n_solar) MW
    wind: np.ndarray        # (T, n_wind) MW
    demand: np.ndarray      # (T, n_cons) MW
    mcp: np.ndarray         # (T,) market clearing price ₹/MWh
    price_source: str


class Simulator:
    """Seeded digital twin: weather → farm output, demand, prices, batteries, grid.

    ``state()`` shows the current tick; ``step(action)`` applies physics (clipping any
    infeasible command and counting it as a violation), balances residual imbalance and
    returns the realised :class:`TickKPI`.
    """

    def __init__(self, config: Config, seed: int = 0, source: str = "synthetic",
                 day: Optional[Date] = None, weather: Optional[WeatherDay] = None,
                 use_batteries: bool = True) -> None:
        self.cfg = config
        self.seed = seed
        self.rng = np.random.default_rng(seed + 1000)
        self.T = config.ticks_per_day
        self.day = day or today_ist()
        self.weather = weather or load_weather_day(source, config, self.day, seed)
        self.profile = self._build_profile()
        n_s, n_w, n_c, n_b = len(config.solar), len(config.wind), len(config.consumers), len(config.batteries)
        self.solar_mult = np.ones((self.T, n_s))
        self.wind_mult = np.ones((self.T, n_w))
        self.demand_mult = np.ones((self.T, n_c))
        self.price_mult = np.ones(self.T)
        self.line_mult = np.ones(self.T)
        self.battery_avail = np.full((self.T, n_b), bool(use_batteries))
        self.solar_avail = np.ones((self.T, n_s))       # 0..1 — breakdowns & maintenance
        self.wind_avail = np.ones((self.T, n_w))
        # Hidden (not forecastable) effects: e.g. whether a probabilistic storm actually hits.
        self.hidden_solar = np.ones((self.T, n_s))
        self.hidden_wind = np.ones((self.T, n_w))
        self.hidden_line = np.ones(self.T)
        self.outages: List[Dict[str, Any]] = []
        self.critical_mw = np.array([c.base_mw * c.critical_frac for c in config.consumers])
        self.soc = np.array([b.soc_init * b.energy_mwh for b in config.batteries])
        self.tick = 0
        self.frequency = float(config.sim.get("frequency_nominal_hz", 50.0))
        self.events: List[Event] = []
        self.event_alerts: List[Alert] = []
        self.maintenance: List[MaintenanceRequest] = []
        self.inspections: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.kpis: List[TickKPI] = []
        self._apply_schedules()

    # ------------------------------------------------------------ profiles
    def _build_profile(self) -> DayProfile:
        cfg, w = self.cfg, self.weather
        T = self.T
        solar = _stack([solar_mw(w.irradiance[:, i], f.capacity_mw, f.performance_ratio)
                        for i, f in enumerate(cfg.solar)], T)
        wind = _stack([wind_mw(w.wind_speed[:, j], f.capacity_mw, f.cut_in_ms, f.rated_ms, f.cut_out_ms)
                       for j, f in enumerate(cfg.wind)], T)
        rng = np.random.default_rng(self.seed + 7)
        hours = np.arange(T) * 24.0 / T
        cols = []
        for c in cfg.consumers:
            noise = 1 + _noise(rng, T, 0.02)
            custom = cfg.custom_profiles.get(c.id)
            if custom is not None and len(custom) > 0:     # uploaded utility consumption data
                series = np.interp(np.linspace(0, len(custom) - 1, T), np.arange(len(custom)), custom)
                cols.append(np.asarray(series, dtype=float) * noise)
            else:
                cols.append(c.base_mw * _load_shape(c.profile, hours) * noise)
        demand = _stack(cols, T)
        for win in cfg.schedules.get("conservation", []):  # industry's planned demand reduction
            a, b = tick_of(win["start"], T), tick_of(win["end"], T)
            demand[a:b] *= 1 - float(win.get("reduction_pct", 0)) / 100
        if w.source == "synthetic":
            mcp, src = self._synthetic_price(demand.sum(1) - solar.sum(1) - wind.sum(1), rng), "synthetic"
        else:
            path = resolve_path(cfg.data.get("iex_prices_csv", "data/iex_prices.csv"))
            mcp, picked = load_iex_prices(path, Date.fromisoformat(w.date), self.seed)
            src = f"IEX {picked}"
        return DayProfile(solar, wind, np.clip(demand, 0, None), mcp, src)

    def _synthetic_price(self, net: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Price correlated with net demand plus an evening scarcity premium."""
        base = self.cfg.market.base_price
        z = (net - net.mean()) / (net.std() + 1e-6)
        hours = np.arange(self.T) * 24.0 / self.T
        evening = 0.35 * np.exp(-((hours - 19.75) / 1.5) ** 2)
        price = base * (1 + 0.28 * z + evening) * (1 + _noise(rng, self.T, 0.04))
        return np.clip(price, 1500, 10000)

    # ------------------------------------------------------------ truth accessors
    def _i(self, t: int) -> int:
        return t % self.T

    def solar_at(self, t: int, forecast: bool = False) -> np.ndarray:
        """Solar MW available at ``t``. ``forecast=True`` hides effects that cannot be foreseen."""
        i = self._i(t)
        out = self.profile.solar[i] * self.solar_mult[i] * self.solar_avail[i]
        return out if forecast else out * self.hidden_solar[i]

    def wind_at(self, t: int, forecast: bool = False) -> np.ndarray:
        i = self._i(t)
        out = self.profile.wind[i] * self.wind_mult[i] * self.wind_avail[i]
        return out if forecast else out * self.hidden_wind[i]

    def demand_at(self, t: int) -> np.ndarray:
        return self.profile.demand[self._i(t)] * self.demand_mult[self._i(t)]

    def mcp_at(self, t: int) -> float:
        return float(self.profile.mcp[self._i(t)] * self.price_mult[self._i(t)])

    def prices_at(self, t: int) -> tuple:
        """(buy, sell) ₹/MWh at tick ``t``."""
        m = self.cfg.market
        mcp = self.mcp_at(t)
        return mcp + m.buy_adder, mcp * m.sell_factor

    def import_limit_at(self, t: int, forecast: bool = False) -> float:
        i = self._i(t)
        return self.cfg.grid.import_limit_mw * float(self.line_mult[i] * (1.0 if forecast else self.hidden_line[i]))

    def export_limit_at(self, t: int, forecast: bool = False) -> float:
        i = self._i(t)
        return self.cfg.grid.export_limit_mw * float(self.line_mult[i] * (1.0 if forecast else self.hidden_line[i]))

    def battery_available_at(self, t: int) -> np.ndarray:
        return self.battery_avail[self._i(t)].copy()

    @property
    def done(self) -> bool:
        return self.tick >= self.T

    # ------------------------------------------------------------ perception
    def state(self) -> GridState:
        """Observable state at the current tick."""
        t = self.tick
        demand = self.demand_at(t)
        critical = np.minimum(self.critical_mw, demand)
        buy, sell = self.prices_at(t)
        return GridState(
            tick=t, hour=(t % self.T) * 24.0 / self.T,
            solar_mw=self.solar_at(t), wind_mw=self.wind_at(t), demand_mw=demand,
            critical_mw=critical, flexible_mw=demand - critical, price_buy=buy, price_sell=sell,
            soc_mwh=self.soc.copy(), battery_available=self.battery_available_at(t),
            import_limit_mw=self.import_limit_at(t), export_limit_mw=self.export_limit_at(t),
            frequency_hz=self.frequency,
            active_events=[e.kind for e in self.events if e.active(t)],
        )

    def alerts(self, horizon: int = 16) -> List[Alert]:
        """Active/upcoming alerts from events, weather data and pending maintenance."""
        t = self.tick
        out = [a for a in self.event_alerts if a.tick <= t < a.tick + a.eta_ticks + max(a.duration_ticks, 1)]
        out = [_aged(a, t) for a in out]
        out += weather_alerts(self.weather, self.cfg, min(t, self.T - 1), horizon)
        for m in self.maintenance:
            if m.scheduled_start is None:
                out.append(Alert("maintenance_request", t,
                                 f"Maintenance {m.id} on {m.asset_id} requested for {clock(m.requested_start)} "
                                 f"({m.duration * 15} min, movable ±{m.window * 15} min)",
                                 "info", max(0, m.requested_start - t), 1.0, m.duration))
        for o in self.outages:
            if o["start"] <= t < o["end"]:
                out.append(Alert("asset_outage", t, f"{o['asset']} {o['cause']} until {clock(o['end'])}",
                                 "critical" if o["cause"] == "breakdown" else "warning", 0, 1.0, o["end"] - t))
        return out

    # ------------------------------------------------------------ events & ops
    def apply_event(self, event: Event) -> Event:
        """Inject a shock (see :mod:`core.events`)."""
        from core.events import apply_event
        return apply_event(self, event)

    def schedule_maintenance(self, req_id: str, start: int) -> MaintenanceRequest:
        """Commit a maintenance job: asset output/availability is zero for its duration."""
        req = next(m for m in self.maintenance if m.id == req_id)
        req.scheduled_start = int(start)
        self.add_outage(req.asset_id, int(start), req.duration, "maintenance")
        return req

    def add_outage(self, asset_id: str, start: int, duration: int, cause: str = "breakdown",
                   severity: float = 1.0) -> Dict[str, Any]:
        """Take (part of) an asset offline: ``severity`` 1.0 = fully unavailable."""
        kind, idx = self.asset_index(asset_id)
        sl = slice(max(0, start), min(self.T, start + duration))
        if kind == "solar":
            self.solar_avail[sl, idx] *= 1 - severity
        elif kind == "wind":
            self.wind_avail[sl, idx] *= 1 - severity
        else:
            self.battery_avail[sl, idx] = False
        o = {"asset": asset_id, "kind": kind, "idx": idx, "start": sl.start, "end": sl.stop, "cause": cause,
             "severity": severity}
        self.outages.append(o)
        return o

    def _apply_schedules(self) -> None:
        """Industry-provided maintenance (movable) and known breakdowns (fixed)."""
        ids = {a.id for a in (*self.cfg.solar, *self.cfg.wind, *self.cfg.batteries)}
        for job in self.cfg.schedules.get("maintenance", []):
            if job.get("asset") not in ids:
                continue
            start, dur = tick_of(job["start"], self.T), max(1, int(round(float(job.get("hours", 1)) * 4)))
            if job.get("kind", "maintenance") == "breakdown":
                self.add_outage(job["asset"], start, dur, "breakdown", float(job.get("severity", 1.0)))
            else:
                self.maintenance.append(MaintenanceRequest(
                    f"M{len(self.maintenance) + 1}", job["asset"], start, dur,
                    max(0, int(round(float(job.get("flexible_hours", 0)) * 4)))))

    def dispatch_inspection(self, asset_id: str) -> str:
        """Send a crew: an unavailable battery returns to service after 4 ticks."""
        kind, idx = self.asset_index(asset_id)
        self.inspections.append({"tick": self.tick, "asset": asset_id})
        for o in self.outages:
            if o["asset"] == asset_id and o["cause"] == "breakdown" and o["start"] <= self.tick < o["end"]:
                back = min(o["end"], self.tick + 4)
                sl = slice(back, o["end"])
                if kind == "solar":
                    self.solar_avail[sl, idx] = 1.0
                elif kind == "wind":
                    self.wind_avail[sl, idx] = 1.0
                else:
                    self.battery_avail[sl, idx] = True
                was, o["end"] = o["end"], back
                return f"Crew dispatched to {asset_id}: repair expected by {clock(back)} (was {clock(was)})."
        return f"Crew dispatched to {asset_id}; no active breakdown found."

    def maintenance_active(self, asset_id: str) -> bool:
        """True if scheduled maintenance covers the current tick for ``asset_id``."""
        return any(m.asset_id == asset_id and m.scheduled_start is not None
                   and m.scheduled_start <= self.tick < m.scheduled_start + m.duration for m in self.maintenance)

    def asset_index(self, asset_id: str) -> tuple:
        """Map an asset id to (kind, index)."""
        for kind, items in (("solar", self.cfg.solar), ("wind", self.cfg.wind), ("battery", self.cfg.batteries)):
            for i, a in enumerate(items):
                if a.id == asset_id:
                    return kind, i
        raise KeyError(asset_id)

    # ------------------------------------------------------------ physics
    def step(self, action: Action) -> TickKPI:
        """Apply ``action`` for the current tick and advance the clock."""
        for m in self.maintenance:  # unscheduled jobs start at their requested time
            if m.scheduled_start is None and self.tick >= m.requested_start:
                self.schedule_maintenance(m.id, self.tick)
        st = self.state()
        cfg, dt, notes = self.cfg, TICK_HOURS, []
        a = action.copy()

        # --- clip commands to physics (each clip is a violation)
        avail = st.battery_available
        for b, bat in enumerate(cfg.batteries):
            if not avail[b] and (a.charge_mw[b] > TOL or a.discharge_mw[b] > TOL):
                notes.append(f"{bat.id} dispatched while unavailable")
                a.charge_mw[b] = a.discharge_mw[b] = 0.0
            if a.charge_mw[b] > TOL and a.discharge_mw[b] > TOL:
                notes.append(f"{bat.id} simultaneous charge/discharge")
                net = a.charge_mw[b] - a.discharge_mw[b]
                a.charge_mw[b], a.discharge_mw[b] = max(net, 0), max(-net, 0)
            for arr, label in ((a.charge_mw, "charge"), (a.discharge_mw, "discharge")):
                if arr[b] > bat.power_mw + TOL:
                    notes.append(f"{bat.id} {label} {arr[b]:.1f} > rating {bat.power_mw}")
                    arr[b] = bat.power_mw
            max_ch = max(0.0, (bat.max_mwh - self.soc[b]) / (bat.eff_one_way * dt))
            max_dis = max(0.0, (self.soc[b] - bat.min_mwh) * bat.eff_one_way / dt)
            if a.charge_mw[b] > max_ch + TOL:
                notes.append(f"{bat.id} would exceed SoC max")
                a.charge_mw[b] = max_ch
            if a.discharge_mw[b] > max_dis + TOL:
                notes.append(f"{bat.id} would breach SoC min")
                a.discharge_mw[b] = max_dis
        if a.grid_import_mw > st.import_limit_mw + TOL:
            notes.append(f"import {a.grid_import_mw:.1f} > line limit {st.import_limit_mw:.1f}")
            a.grid_import_mw = st.import_limit_mw
        if a.grid_export_mw > st.export_limit_mw + TOL:
            notes.append(f"export {a.grid_export_mw:.1f} > line limit {st.export_limit_mw:.1f}")
            a.grid_export_mw = st.export_limit_mw
        a.curtail_solar_mw = np.clip(a.curtail_solar_mw, 0, st.solar_mw)
        a.curtail_wind_mw = np.clip(a.curtail_wind_mw, 0, st.wind_mw)
        a.dr_mw = np.clip(a.dr_mw, 0, st.flexible_mw * np.array([c.dr_eligible for c in cfg.consumers]))
        a.shed_mw = np.clip(a.shed_mw, 0, st.flexible_mw - a.dr_mw)

        # --- realised renewables (nowcast error)
        err = float(cfg.sim.get("nowcast_error", 0.03))
        solar_real = np.clip(st.solar_mw * (1 + self.rng.normal(0, err, st.solar_mw.shape)), 0,
                             [f.capacity_mw for f in cfg.solar])
        wind_real = np.clip(st.wind_mw * (1 + self.rng.normal(0, err, st.wind_mw.shape)), 0,
                            [f.capacity_mw for f in cfg.wind])
        curt_s = np.minimum(a.curtail_solar_mw, solar_real)
        curt_w = np.minimum(a.curtail_wind_mw, wind_real)
        renew_used = float(solar_real.sum() - curt_s.sum() + wind_real.sum() - curt_w.sum())
        load = st.demand_mw - a.dr_mw - a.shed_mw - a.unserved_critical_mw
        imbalance = (renew_used + a.discharge_mw.sum() + a.grid_import_mw
                     - load.sum() - a.charge_mw.sum() - a.grid_export_mw)
        nominal = float(cfg.sim.get("frequency_nominal_hz", 50.0))
        sens = float(cfg.sim.get("frequency_sensitivity_hz", 0.5))
        self.frequency = nominal + sens * imbalance / max(st.total_demand_mw, 1.0)

        # --- real-time balancing: batteries → grid → curtail / shed
        balancing_mwh, forced_curt, extra_shed = 0.0, 0.0, 0.0
        resid = imbalance
        for b, bat in enumerate(cfg.batteries):
            if not avail[b] or abs(resid) < TOL:
                continue
            if resid > 0:   # surplus → extra charge
                room = min(bat.power_mw - a.charge_mw[b] + a.discharge_mw[b],
                           (bat.max_mwh - self.soc[b]) / (bat.eff_one_way * dt) - a.charge_mw[b] + a.discharge_mw[b])
                use = float(np.clip(resid, 0, max(room, 0)))
                _shift(a, b, +use)
            else:           # deficit → extra discharge
                room = min(bat.power_mw - a.discharge_mw[b] + a.charge_mw[b],
                           (self.soc[b] - bat.min_mwh) * bat.eff_one_way / dt - a.discharge_mw[b] + a.charge_mw[b])
                use = float(np.clip(-resid, 0, max(room, 0)))
                _shift(a, b, -use)
            resid -= use if resid > 0 else -use
        if resid > TOL:
            room = max(0.0, min(a.grid_import_mw, resid))           # first cut planned import
            a.grid_import_mw -= room
            resid -= room
            add = min(resid, max(0.0, st.export_limit_mw - a.grid_export_mw))
            a.grid_export_mw += add
            balancing_mwh += (room + add) * dt
            resid -= add
            forced_curt = max(resid, 0.0)
        elif resid < -TOL:
            need = -resid
            room = max(0.0, min(a.grid_export_mw, need))
            a.grid_export_mw -= room
            need -= room
            add = min(need, max(0.0, st.import_limit_mw - a.grid_import_mw))
            a.grid_import_mw += add
            balancing_mwh += (room + add) * dt
            extra_shed = need - add

        # --- shed any remaining deficit: flexible first, then critical
        flex_left = float((st.flexible_mw - a.dr_mw - a.shed_mw).sum())
        shed_flex = min(extra_shed, flex_left)
        unserved_crit = a.unserved_critical_mw.sum() + max(0.0, extra_shed - shed_flex)
        reliability_notes = [f"critical load unserved {unserved_crit:.2f} MW"] if unserved_crit > TOL else []

        # --- battery state update
        for b, bat in enumerate(cfg.batteries):
            self.soc[b] += (a.charge_mw[b] * bat.eff_one_way - a.discharge_mw[b] / bat.eff_one_way) * dt
            self.soc[b] = float(np.clip(self.soc[b], bat.min_mwh - 1e-6, bat.max_mwh + 1e-6))

        # --- KPIs
        m = cfg.market
        served = float(load.sum()) - shed_flex - max(0.0, extra_shed - shed_flex)
        energy_cost = (a.grid_import_mw * st.price_buy - a.grid_export_mw * st.price_sell) * dt
        degradation = sum(bat.degradation_cost * (a.charge_mw[b] + a.discharge_mw[b]) * dt
                          for b, bat in enumerate(cfg.batteries))
        dr_pay = float(a.dr_mw.sum()) * m.dr_incentive * dt
        balancing = balancing_mwh * m.deviation_penalty
        co2 = a.grid_import_mw * dt * cfg.grid.carbon_intensity_t_per_mwh
        carbon = co2 * m.carbon_price
        kpi = TickKPI(
            tick=st.tick, cost_rs=energy_cost + degradation + dr_pay + balancing + carbon,
            energy_cost_rs=energy_cost, degradation_rs=degradation, dr_payment_rs=dr_pay,
            balancing_rs=balancing, carbon_rs=carbon, renewable_mwh=renew_used * dt - forced_curt * dt,
            served_mwh=served * dt, import_mwh=a.grid_import_mw * dt, export_mwh=a.grid_export_mw * dt,
            curtailed_mwh=(curt_s.sum() + curt_w.sum() + forced_curt) * dt,
            unserved_mwh=(float(a.shed_mw.sum()) + shed_flex + unserved_crit) * dt,
            unserved_critical_mwh=unserved_crit * dt, co2_t=co2, violations=len(notes),
            frequency_hz=self.frequency, violation_notes=notes + reliability_notes,
        )
        self.kpis.append(kpi)
        self.history.append(self._history_row(st, a, kpi, solar_real, wind_real, curt_s, curt_w, forced_curt))
        self.tick += 1
        return kpi

    def _history_row(self, st: GridState, a: Action, kpi: TickKPI, solar_real: np.ndarray,
                     wind_real: np.ndarray, curt_s: np.ndarray, curt_w: np.ndarray, forced: float) -> Dict[str, Any]:
        row = {
            "tick": st.tick, "hour": st.hour, "solar_mw": float(solar_real.sum() - curt_s.sum()),
            "wind_mw": float(wind_real.sum() - curt_w.sum()), "solar_avail_mw": float(solar_real.sum()),
            "wind_avail_mw": float(wind_real.sum()), "demand_mw": st.total_demand_mw,
            "served_mw": kpi.served_mwh / TICK_HOURS, "discharge_mw": float(a.discharge_mw.sum()),
            "charge_mw": float(a.charge_mw.sum()), "import_mw": a.grid_import_mw, "export_mw": a.grid_export_mw,
            "curtailed_mw": float(curt_s.sum() + curt_w.sum() + forced), "forced_curtail_mw": forced,
            "price_buy": st.price_buy,
            "price_sell": st.price_sell, "frequency_hz": kpi.frequency_hz, "cost_rs": kpi.cost_rs,
            "import_limit_mw": st.import_limit_mw, "export_limit_mw": st.export_limit_mw,
            "dr_mw": float(a.dr_mw.sum()), "shed_mw": kpi.unserved_mwh / TICK_HOURS, "violations": kpi.violations,
        }
        for b, bat in enumerate(self.cfg.batteries):
            row[f"soc_{bat.id}"] = float(self.soc[b] / bat.energy_mwh * 100)
        return row

    def totals(self) -> Dict[str, float]:
        """Aggregate KPIs over all completed ticks."""
        k = self.kpis
        served = sum(x.served_mwh for x in k)
        imports = sum(x.import_mwh for x in k)
        return {
            "cost_rs": sum(x.cost_rs for x in k),
            "clean_pct": 100.0 * max(0.0, served - imports) / served if served > 0 else 0.0,
            "curtailed_mwh": sum(x.curtailed_mwh for x in k),
            "unserved_mwh": sum(x.unserved_mwh for x in k),
            "unserved_critical_mwh": sum(x.unserved_critical_mwh for x in k),
            "violations": sum(x.violations for x in k),
            "co2_t": sum(x.co2_t for x in k),
            "import_mwh": imports,
            "export_mwh": sum(x.export_mwh for x in k),
            "served_mwh": served,
        }


def _aged(a: Alert, t: int) -> Alert:
    """Alert as seen at tick ``t`` (eta counts down)."""
    from dataclasses import replace
    return replace(a, eta_ticks=max(0, a.tick + a.eta_ticks - t),
                   duration_ticks=max(1, a.tick + a.eta_ticks + a.duration_ticks - max(t, a.tick + a.eta_ticks)))


def tick_of(hhmm: Any, T: int = 96) -> int:
    """'14:30' → tick index (also accepts an int tick)."""
    if isinstance(hhmm, (int, np.integer)):
        return int(np.clip(hhmm, 0, T))
    h, m = str(hhmm).strip().split(":")[:2]
    return int(np.clip(round((int(h) * 60 + int(m)) / (1440 / T)), 0, T))


def clock(tick: int, T: int = 96) -> str:
    """Tick index → 'HH:MM'."""
    minutes = int(tick) % T * 1440 // T
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _stack(cols: List[np.ndarray], T: int) -> np.ndarray:
    return np.column_stack(cols) if cols else np.zeros((T, 0))


def _shift(a: Action, b: int, delta: float) -> None:
    """Move battery ``b`` net power by ``delta`` MW (+ = more charging) without simultaneous flows."""
    net = a.charge_mw[b] - a.discharge_mw[b] + delta
    a.charge_mw[b], a.discharge_mw[b] = max(net, 0.0), max(-net, 0.0)


def _load_shape(profile: str, hours: np.ndarray) -> np.ndarray:
    """Normalised daily load shape with morning and evening peaks."""
    morning = np.exp(-((hours - 9.5) / 1.8) ** 2)
    evening = np.exp(-((hours - 19.5) / 1.8) ** 2)
    if profile == "commercial":
        day = 1 / (1 + np.exp(-(hours - 8.5) * 2)) - 1 / (1 + np.exp(-(hours - 21) * 2))
        return 0.55 + 0.35 * day + 0.12 * evening + 0.05 * morning
    if profile == "flat":
        return 0.95 + 0.04 * np.sin(2 * np.pi * (hours - 9) / 24)
    return 0.82 + 0.12 * morning + 0.16 * evening      # industrial


def _noise(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.normal(0, sigma * np.sqrt(1 - 0.81))
    return x
