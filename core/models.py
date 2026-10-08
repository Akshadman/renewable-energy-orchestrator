"""Core dataclasses shared by the simulator, optimizer, validator and agent."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

TICK_HOURS = 0.25  # 15-minute ticks


# ---------------------------------------------------------------- assets
@dataclass(frozen=True)
class SolarFarm:
    id: str
    name: str
    capacity_mw: float
    lat: float
    lon: float
    performance_ratio: float = 0.9
    cloud_sensitivity: float = 1.0


@dataclass(frozen=True)
class WindFarm:
    id: str
    name: str
    capacity_mw: float
    lat: float
    lon: float
    cut_in_ms: float = 3.0
    rated_ms: float = 12.0
    cut_out_ms: float = 25.0


@dataclass(frozen=True)
class Battery:
    id: str
    name: str
    energy_mwh: float
    power_mw: float
    round_trip_eff: float = 0.9
    soc_min: float = 0.1
    soc_max: float = 0.95
    soc_init: float = 0.5
    degradation_cost: float = 400.0

    @property
    def eff_one_way(self) -> float:
        """Charge and discharge efficiency (sqrt of round trip)."""
        return float(np.sqrt(self.round_trip_eff))

    @property
    def min_mwh(self) -> float:
        return self.soc_min * self.energy_mwh

    @property
    def max_mwh(self) -> float:
        return self.soc_max * self.energy_mwh


@dataclass(frozen=True)
class Consumer:
    id: str
    name: str
    base_mw: float
    critical_frac: float
    dr_eligible: bool
    profile: str = "industrial"


@dataclass(frozen=True)
class GridConfig:
    import_limit_mw: float
    export_limit_mw: float
    carbon_intensity_t_per_mwh: float


@dataclass(frozen=True)
class MarketConfig:
    base_price: float
    buy_adder: float
    sell_factor: float
    carbon_price: float
    dr_incentive: float
    curtailment_penalty: float
    voll_flexible: float
    voll_critical: float
    reserve_shortfall_penalty: float
    deviation_penalty: float


@dataclass
class Config:
    """Full system configuration loaded from ``config/assets.yaml``."""

    solar: List[SolarFarm]
    wind: List[WindFarm]
    batteries: List[Battery]
    consumers: List[Consumer]
    grid: GridConfig
    market: MarketConfig
    sim: Dict[str, Any]
    agent: Dict[str, Any]
    data: Dict[str, Any]
    targets: Dict[str, Any] = field(default_factory=dict)        # industry goals (carbon, cost, risk)
    schedules: Dict[str, Any] = field(default_factory=dict)      # maintenance / breakdowns / conservation
    custom_profiles: Dict[str, List[float]] = field(default_factory=dict)  # consumer id -> 96 MW values
    name: str = "Default portfolio"

    @property
    def ticks_per_day(self) -> int:
        return int(self.sim.get("ticks_per_day", 96))


# ---------------------------------------------------------------- runtime state
@dataclass
class GridState:
    """Snapshot of the system at the start of a tick (what the controller perceives)."""

    tick: int
    hour: float
    solar_mw: np.ndarray          # available (pre-curtailment) per solar farm
    wind_mw: np.ndarray           # available per wind farm
    demand_mw: np.ndarray         # per consumer
    critical_mw: np.ndarray       # per consumer
    flexible_mw: np.ndarray       # per consumer (demand - critical)
    price_buy: float
    price_sell: float
    soc_mwh: np.ndarray           # per battery
    battery_available: np.ndarray  # bool per battery
    import_limit_mw: float
    export_limit_mw: float
    frequency_hz: float = 50.0
    active_events: List[str] = field(default_factory=list)

    @property
    def renewable_mw(self) -> float:
        return float(self.solar_mw.sum() + self.wind_mw.sum())

    @property
    def total_demand_mw(self) -> float:
        return float(self.demand_mw.sum())

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly representation."""
        return _jsonable(asdict(self))


@dataclass
class Forecast:
    """Quantile forecasts over the horizon. Arrays are shaped (H, n_assets) or (H,)."""

    start_tick: int
    horizon: int
    solar: Dict[str, np.ndarray]    # keys p10/p50/p90 -> (H, n_solar)
    wind: Dict[str, np.ndarray]     # (H, n_wind)
    demand: Dict[str, np.ndarray]   # (H, n_consumers)
    price: Dict[str, np.ndarray]    # (H,) buy-side market price
    critical_frac: np.ndarray       # (n_consumers,)
    battery_available: np.ndarray   # (H, n_batteries) planned availability
    import_limit: np.ndarray        # (H,)
    export_limit: np.ndarray        # (H,)

    def summary(self) -> Dict[str, Any]:
        """Compact numeric summary for the LLM (totals per quantile, next 4h)."""
        out: Dict[str, Any] = {"start_tick": self.start_tick, "horizon_ticks": self.horizon}
        for name, q in (("solar_mw", self.solar), ("wind_mw", self.wind), ("demand_mw", self.demand)):
            out[name] = {k: [round(float(x), 1) for x in v.sum(axis=1)[::4]] for k, v in q.items()}
        out["price_rs_mwh"] = {k: [round(float(x)) for x in v[::4]] for k, v in self.price.items()}
        out["min_import_limit_mw"] = round(float(self.import_limit.min()), 1)
        out["battery_available_all_horizon"] = bool(self.battery_available.all())
        return out


@dataclass
class Action:
    """Dispatch decision for a single tick (all MW, average over the tick)."""

    charge_mw: np.ndarray            # per battery
    discharge_mw: np.ndarray         # per battery
    grid_import_mw: float
    grid_export_mw: float
    curtail_solar_mw: np.ndarray     # per solar farm
    curtail_wind_mw: np.ndarray      # per wind farm
    dr_mw: np.ndarray                # paid demand-response reduction per consumer
    shed_mw: np.ndarray              # involuntary flexible-load shedding per consumer
    unserved_critical_mw: np.ndarray  # must be zero for a valid action
    source: str = "optimizer"

    @classmethod
    def zeros(cls, n_solar: int, n_wind: int, n_bat: int, n_cons: int, source: str = "noop") -> "Action":
        z = np.zeros
        return cls(z(n_bat), z(n_bat), 0.0, 0.0, z(n_solar), z(n_wind), z(n_cons), z(n_cons), z(n_cons), source)

    def copy(self) -> "Action":
        return Action(**{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()})

    def supply_mw(self, state: GridState) -> float:
        """Total power injected into the bus."""
        return float(state.solar_mw.sum() - self.curtail_solar_mw.sum() + state.wind_mw.sum()
                     - self.curtail_wind_mw.sum() + self.discharge_mw.sum() + self.grid_import_mw)

    def sink_mw(self, state: GridState) -> float:
        """Total power withdrawn from the bus."""
        served = state.demand_mw - self.dr_mw - self.shed_mw - self.unserved_critical_mw
        return float(served.sum() + self.charge_mw.sum() + self.grid_export_mw)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(dict(self.__dict__))


@dataclass
class Event:
    """A shock injected into the digital twin."""

    kind: str
    start_tick: int
    duration: int
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def active(self, tick: int) -> bool:
        return self.start_tick <= tick < self.start_tick + self.duration


@dataclass
class Alert:
    """Operator-facing alert (from events, weather data or the agent)."""

    kind: str
    tick: int
    message: str
    severity: str = "warning"     # info | warning | critical
    eta_ticks: int = 0             # how far ahead the condition is expected
    probability: float = 1.0       # chance the condition materialises
    duration_ticks: int = 0        # expected length once it starts


@dataclass
class TickKPI:
    """Realised outcome of one tick."""

    tick: int
    cost_rs: float
    energy_cost_rs: float
    degradation_rs: float
    dr_payment_rs: float
    balancing_rs: float
    carbon_rs: float
    renewable_mwh: float
    served_mwh: float
    import_mwh: float
    export_mwh: float
    curtailed_mwh: float
    unserved_mwh: float
    unserved_critical_mwh: float
    co2_t: float
    violations: int                  # executed commands that breached a physical limit
    frequency_hz: float
    violation_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy types to plain Python for JSON logging."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 4)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def jsonable(obj: Any) -> Any:
    """Public wrapper around the JSON conversion helper."""
    return _jsonable(obj)
