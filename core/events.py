"""Shock injectors for the digital twin. Each is callable from the UI and from Monte Carlo."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List

import numpy as np

from core.models import Alert, Event

if TYPE_CHECKING:  # pragma: no cover
    from core.simulator import Simulator

EventFn = Callable[["Simulator", Event], str]


def _window(sim: "Simulator", ev: Event, offset: int = 0) -> slice:
    start = min(sim.T, ev.start_tick + offset)
    return slice(start, min(sim.T, start + ev.duration))


def _pick(sim: "Simulator", ev: Event, key: str, n: int) -> List[int]:
    rng = np.random.default_rng(sim.seed + ev.start_tick + sum(map(ord, ev.kind)))
    chosen = ev.params.get(key)
    if chosen is None:
        return sorted(rng.choice(n, size=max(1, n // 2), replace=False).tolist())
    return [int(i) for i in (chosen if isinstance(chosen, list) else [chosen])]


def cloud_cover(sim: "Simulator", ev: Event) -> str:
    """Thick cloud bank over some solar farms (sensitivity-weighted output loss)."""
    sev = float(ev.params.get("severity", 0.7))
    farms = _pick(sim, ev, "farms", len(sim.cfg.solar))
    for i in farms:
        loss = min(0.95, sev * sim.cfg.solar[i].cloud_sensitivity)
        sim.solar_mult[_window(sim, ev), i] *= 1 - loss
    return f"Cloud cover over {', '.join(sim.cfg.solar[i].id for i in farms)} (−{sev:.0%})"


def wind_surge(sim: "Simulator", ev: Event) -> str:
    """Gusty front raises wind output."""
    f = float(ev.params.get("factor", 1.6))
    sim.wind_mult[_window(sim, ev)] *= f
    cap = np.array([w.capacity_mw for w in sim.cfg.wind])
    sl = _window(sim, ev)
    sim.wind_mult[sl] = np.minimum(sim.wind_mult[sl], cap / np.maximum(sim.profile.wind[sl], 1e-6))
    return f"Wind surge ×{f:.1f}"


def wind_drop(sim: "Simulator", ev: Event) -> str:
    """Wind lull across the fleet."""
    f = float(ev.params.get("factor", 0.2))
    sim.wind_mult[_window(sim, ev)] *= f
    return f"Wind drop to {f:.0%} of expected"


def price_spike(sim: "Simulator", ev: Event) -> str:
    """Market price jumps (scarcity)."""
    f = float(ev.params.get("factor", 2.5))
    sim.price_mult[_window(sim, ev)] *= f
    sim.price_mult[:] = np.minimum(sim.price_mult, 12000 / np.maximum(sim.profile.mcp, 1))
    return f"Price spike ×{f:.1f}"


def battery_unavailable(sim: "Simulator", ev: Event) -> str:
    """Battery trips offline (fault)."""
    if not sim.cfg.batteries:
        return "No battery to fail"
    b = int(ev.params.get("battery", len(sim.cfg.batteries) - 1)) % len(sim.cfg.batteries)
    sim.add_outage(sim.cfg.batteries[b].id, ev.start_tick, ev.duration, "breakdown")
    return f"{sim.cfg.batteries[b].id} offline (fault)"


def generator_breakdown(sim: "Simulator", ev: Event) -> str:
    """A solar or wind farm (or part of it) breaks down — inverter / turbine failure."""
    farms = [f.id for f in (*sim.cfg.solar, *sim.cfg.wind)]
    asset = ev.params.get("asset") or farms[(sim.seed + ev.start_tick) % len(farms)]
    sev = float(ev.params.get("severity", 1.0))
    sim.add_outage(asset, ev.start_tick, ev.duration, "breakdown", sev)
    return f"{asset} breakdown ({sev:.0%} of capacity lost)"


def line_congestion(sim: "Simulator", ev: Event) -> str:
    """Transmission constraint reduces import/export capacity."""
    f = float(ev.params.get("factor", 0.35))
    sim.line_mult[_window(sim, ev)] *= f
    return f"Line congestion: interconnection limited to {f:.0%}"


def demand_surge(sim: "Simulator", ev: Event) -> str:
    """Industrial demand jumps (heatwave / extra shift)."""
    f = float(ev.params.get("factor", 1.25))
    sim.demand_mult[_window(sim, ev)] *= f
    return f"Demand surge ×{f:.2f}"


def forecast_update(sim: "Simulator", ev: Event) -> str:
    """Revised met forecast: renewable output for the coming hours is re-estimated."""
    rng = np.random.default_rng(sim.seed + ev.start_tick)
    f_s = float(ev.params.get("solar_factor", rng.uniform(0.6, 1.2)))
    f_w = float(ev.params.get("wind_factor", rng.uniform(0.6, 1.3)))
    sl = _window(sim, ev)
    sim.solar_mult[sl] *= f_s
    sim.wind_mult[sl] *= f_w
    cap = np.array([w.capacity_mw for w in sim.cfg.wind])
    sim.wind_mult[sl] = np.minimum(sim.wind_mult[sl], cap / np.maximum(sim.profile.wind[sl], 1e-6))
    return f"Forecast revised: solar ×{f_s:.2f}, wind ×{f_w:.2f}"


STORM_EFFECT = {"wind": 0.05, "solar": 0.3, "line": 0.6}   # shared with the agent's scenario model


def storm_alert(sim: "Simulator", ev: Event) -> str:
    """Storm *forecast* with probability ``probability``; whether it hits is drawn now but hidden
    from forecasts (the agent only sees the alert). If it hits: turbines cut out, solar dims,
    the line is derated."""
    lead = int(ev.params.get("lead", 8))
    prob = float(np.clip(ev.params.get("probability", 0.6), 0.0, 1.0))
    hits = ev.params.get("hits")
    if hits is None:
        hits = bool(np.random.default_rng(sim.seed * 31 + ev.start_tick).random() < prob)
    ev.params.update(lead=lead, probability=prob, hits=bool(hits))
    if hits:
        sl = _window(sim, ev, offset=lead)
        sim.hidden_wind[sl] *= STORM_EFFECT["wind"]
        sim.hidden_solar[sl] *= STORM_EFFECT["solar"]
        sim.hidden_line[sl] *= STORM_EFFECT["line"]
    return f"Storm forecast: {prob:.0%} chance in {lead * 15} min, lasting {ev.duration * 15} min"


def maintenance_window(sim: "Simulator", ev: Event) -> str:
    """Planned maintenance request; the agent may shift it inside a window."""
    from core.simulator import MaintenanceRequest
    farms = [f.id for f in (*sim.cfg.wind, *sim.cfg.solar)]
    asset = ev.params.get("asset") or farms[0]
    lead = int(ev.params.get("lead", 4))
    req = MaintenanceRequest(id=f"M{len(sim.maintenance) + 1}", asset_id=asset,
                             requested_start=min(sim.T - 1, ev.start_tick + lead), duration=ev.duration,
                             window=int(ev.params.get("window", 16)))
    sim.maintenance.append(req)
    return f"Maintenance {req.id} requested on {asset} at tick {req.requested_start} ({ev.duration * 15} min)"


EVENTS: Dict[str, Dict[str, Any]] = {
    "cloud_cover": {"fn": cloud_cover, "duration": 12, "severity": "warning"},
    "wind_surge": {"fn": wind_surge, "duration": 12, "severity": "info"},
    "wind_drop": {"fn": wind_drop, "duration": 16, "severity": "warning"},
    "price_spike": {"fn": price_spike, "duration": 8, "severity": "warning"},
    "battery_unavailable": {"fn": battery_unavailable, "duration": 16, "severity": "critical"},
    "generator_breakdown": {"fn": generator_breakdown, "duration": 16, "severity": "critical"},
    "line_congestion": {"fn": line_congestion, "duration": 12, "severity": "warning"},
    "demand_surge": {"fn": demand_surge, "duration": 12, "severity": "warning"},
    "forecast_update": {"fn": forecast_update, "duration": 16, "severity": "info"},
    "storm_alert": {"fn": storm_alert, "duration": 8, "severity": "critical"},
    "maintenance_window": {"fn": maintenance_window, "duration": 8, "severity": "info"},
}


def make_event(kind: str, start_tick: int, duration: int = 0, **params: Any) -> Event:
    """Build an :class:`Event` with the registered default duration."""
    if kind not in EVENTS:
        raise ValueError(f"unknown event {kind!r}")
    return Event(kind, start_tick, duration or EVENTS[kind]["duration"], params)


def apply_event(sim: "Simulator", ev: Event) -> Event:
    """Mutate the simulator's truth profiles and raise a matching alert."""
    ev.description = EVENTS[ev.kind]["fn"](sim, ev)
    sim.events.append(ev)
    if ev.kind == "storm_alert":   # probabilistic warning: eta = lead, shown until the window passes
        sim.event_alerts.append(Alert(ev.kind, ev.start_tick, ev.description, EVENTS[ev.kind]["severity"],
                                      eta_ticks=int(ev.params["lead"]), probability=float(ev.params["probability"]),
                                      duration_ticks=ev.duration))
    elif ev.kind not in ("maintenance_window", "battery_unavailable", "generator_breakdown"):
        sim.event_alerts.append(Alert(ev.kind, ev.start_tick, ev.description, EVENTS[ev.kind]["severity"],
                                      eta_ticks=0, duration_ticks=ev.duration))
    return ev


def random_schedule(rng: np.random.Generator, T: int, n_batteries: int, n_events: int = 3) -> List[Event]:
    """Random event schedule for Monte Carlo days."""
    kinds = list(EVENTS)
    out = []
    for _ in range(n_events):
        kind = str(rng.choice(kinds))
        start = int(rng.integers(8, T - 12))
        params: Dict[str, Any] = {}
        if kind == "battery_unavailable":
            params["battery"] = int(rng.integers(max(1, n_batteries)))
        if kind == "storm_alert":
            params["probability"] = float(np.round(rng.uniform(0.1, 0.95), 2))
        out.append(make_event(kind, start, **params))
    return sorted(out, key=lambda e: e.start_tick)
