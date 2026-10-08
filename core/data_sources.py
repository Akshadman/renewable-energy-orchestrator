"""Weather & price data layer.

Three interchangeable :class:`WeatherProvider` implementations feed the digital twin:

* :class:`OpenMeteoProvider`   — live 15-minute data (``minutely_15``) from api.open-meteo.com
* :class:`LastYearProvider`    — proxy: the same calendar date one year earlier (Open-Meteo archive)
* :class:`ReplayProvider`      — a cached CSV of a past day (``data/replay/*.csv``)
* :class:`SyntheticProvider`   — statistically generated weather (the original simulator)

:func:`load_weather_day` applies the fallback chain Live → Last year → Replay → Synthetic and
reports which source is actually active. Prices come from ``data/iex_prices.csv``
(15-minute IEX blocks); a sample file is generated if it is missing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import resolve_path
from core.models import Alert, Config
from core.power_curves import cloud_attenuation

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OM_VARS = ["shortwave_radiation", "wind_speed_100m", "cloud_cover", "weather_code"]
SOURCES = ["openmeteo", "lastyear", "replay", "synthetic"]
SOURCE_LABELS = {"openmeteo": "Live forecast (Open-Meteo, 15-min)", "lastyear": "Proxy: same day last year",
                 "replay": "Replay (saved day)", "synthetic": "Synthetic (simulated weather)"}


class DataSourceError(RuntimeError):
    """Raised when a provider cannot supply a complete day."""


@dataclass
class WeatherDay:
    """One day (96 × 15 min) of weather for every farm site."""

    source: str
    date: str
    irradiance: np.ndarray      # (T, n_solar) W/m² GHI
    cloud_cover: np.ndarray     # (T, n_solar) %
    solar_codes: np.ndarray     # (T, n_solar) WMO weather code
    wind_speed: np.ndarray      # (T, n_wind) m/s at 100 m
    wind_codes: np.ndarray      # (T, n_wind) WMO weather code
    notes: List[str] = field(default_factory=list)

    @property
    def ticks(self) -> int:
        return int(self.irradiance.shape[0])


def today_ist() -> Date:
    """Current calendar date in India."""
    return datetime.now(IST).date()


# ---------------------------------------------------------------- providers
class WeatherProvider(ABC):
    """Interface every weather source implements."""

    name: str = "base"

    @abstractmethod
    def get_day(self, config: Config, day: Date, seed: int = 0) -> WeatherDay:
        """Return a full day of 15-min weather for all farms or raise :class:`DataSourceError`."""


class SyntheticProvider(WeatherProvider):
    """Seeded statistical weather: clear-sky bell curve × AR(1) clouds, AR(1) hub-height wind."""

    name = "synthetic"

    def get_day(self, config: Config, day: Date, seed: int = 0) -> WeatherDay:
        rng = np.random.default_rng(seed)
        T = config.ticks_per_day
        hours = np.arange(T) * 24.0 / T
        doy = day.timetuple().tm_yday
        daylen = 12.0 + 1.3 * np.sin(2 * np.pi * (doy - 80) / 365)       # ~26°N day length
        sunrise, sunset = 12.8 - daylen / 2, 12.8 + daylen / 2            # solar noon ≈ 12:48 IST (Rajasthan)
        frac = np.clip((hours + 0.125 - sunrise) / (sunset - sunrise), 0, 1)
        clear = 980.0 * np.sin(np.pi * frac) ** 1.25 * rng.uniform(0.85, 1.02)

        regional = _ar1(rng, T, phi=0.97, sigma=9.0, mean=rng.uniform(5, 45))
        n_s, n_w = len(config.solar), len(config.wind)
        cloud = np.zeros((T, n_s))
        for i, farm in enumerate(config.solar):
            local = _ar1(rng, T, phi=0.9, sigma=6.0, mean=0.0)
            cloud[:, i] = np.clip(regional + local * farm.cloud_sensitivity, 0, 100)
        irr = clear[:, None] * cloud_attenuation(cloud)

        wind = np.zeros((T, n_w))
        diurnal = 1.0 + 0.18 * np.sin(2 * np.pi * (hours - 11) / 24)       # evening peak
        for j, _farm in enumerate(config.wind):
            mean = rng.uniform(6.5, 10.0)
            wind[:, j] = np.clip(mean * diurnal + _ar1(rng, T, phi=0.96, sigma=0.55, mean=0.0), 0, None)
        wind_cloud = np.clip(np.repeat(regional[:, None], n_w, axis=1), 0, 100)
        return WeatherDay(self.name, day.isoformat(), irr, cloud, _codes_from_cloud(cloud, rng),
                          wind, _codes_from_cloud(wind_cloud, rng))


class OpenMeteoProvider(WeatherProvider):
    """Live 15-minute weather from Open-Meteo with on-disk response caching."""

    name = "openmeteo"

    def __init__(self, cache_dir: Optional[Path] = None, ttl_minutes: float = 60, timeout_s: float = 10,
                 replay_dir: Optional[Path] = None, http_get: Optional[Callable[..., Any]] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else resolve_path("data/cache")
        self.replay_dir = Path(replay_dir) if replay_dir else resolve_path("data/replay")
        self.ttl_s = ttl_minutes * 60
        self.timeout_s = timeout_s
        self._get = http_get
        self.last_from_cache = False

    url = OPEN_METEO_URL

    def get_day(self, config: Config, day: Date, seed: int = 0) -> WeatherDay:
        payload = self._fetch_cached(self._params(config, day), permanent=day < today_ist())
        wd = parse_open_meteo(payload, config, day)
        try:
            save_replay_csv(wd, config, self.replay_dir)
        except OSError as exc:  # replay snapshot is best-effort
            log.warning("could not write replay CSV: %s", exc)
        return wd

    @staticmethod
    def _sites(config: Config) -> Dict[str, str]:
        sites = [(f.lat, f.lon) for f in config.solar] + [(f.lat, f.lon) for f in config.wind]
        return {"latitude": ",".join(f"{lat:.4f}" for lat, _ in sites),
                "longitude": ",".join(f"{lon:.4f}" for _, lon in sites)}

    def _params(self, config: Config, day: Date) -> Dict[str, str]:
        return {**self._sites(config), "minutely_15": ",".join(OM_VARS), "wind_speed_unit": "ms",
                "timezone": "Asia/Kolkata", "start_date": day.isoformat(), "end_date": day.isoformat()}

    def _fetch_cached(self, params: Dict[str, str], permanent: bool) -> Any:
        key = hashlib.sha1(json.dumps([self.url, params], sort_keys=True).encode()).hexdigest()[:16]
        path = self.cache_dir / f"{self.name}_{params['start_date']}_{key}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if permanent or time.time() - cached.get("fetched_at", 0) < self.ttl_s:
                self.last_from_cache = True
                return cached["payload"]
        self.last_from_cache = False
        get = self._get or _requests_get
        try:
            resp = get(self.url, params=params, timeout=self.timeout_s)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # network, HTTP, JSON — all mean "API unavailable"
            raise DataSourceError(f"{self.name} request failed: {exc}") from exc
        if isinstance(payload, dict) and payload.get("error"):
            raise DataSourceError(f"Open-Meteo error: {payload.get('reason')}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": time.time(), "params": params, "payload": payload}))
        return payload


class LastYearProvider(OpenMeteoProvider):
    """Proxy data set: the same calendar date one year earlier from the Open-Meteo archive (hourly,
    interpolated to 15 min). Used when the live forecast is unavailable."""

    name = "lastyear"
    url = ARCHIVE_URL

    def get_day(self, config: Config, day: Date, seed: int = 0) -> WeatherDay:
        proxy = _one_year_earlier(day)
        params = {**self._sites(config), "hourly": ",".join(OM_VARS), "wind_speed_unit": "ms",
                  "timezone": "Asia/Kolkata", "start_date": proxy.isoformat(), "end_date": proxy.isoformat()}
        payload = self._fetch_cached(params, permanent=True)
        wd = parse_open_meteo(_hourly_to_15min(payload), config, proxy)
        wd.source = self.name
        wd.notes.append(f"proxy weather from {proxy.isoformat()} (one year earlier)")
        return wd


class ReplayProvider(WeatherProvider):
    """Replays a stored day from ``data/replay/*.csv`` (exact date if present, else the latest)."""

    name = "replay"

    def __init__(self, replay_dir: Optional[Path] = None) -> None:
        self.replay_dir = Path(replay_dir) if replay_dir else resolve_path("data/replay")

    def available(self) -> List[Path]:
        """Replay files sorted oldest → newest."""
        return sorted(self.replay_dir.glob("*.csv")) if self.replay_dir.exists() else []

    def get_day(self, config: Config, day: Date, seed: int = 0) -> WeatherDay:
        files = self.available()
        if not files:
            raise DataSourceError(f"no replay CSVs in {self.replay_dir}")
        exact = [f for f in files if day.isoformat() in f.name]
        path = exact[-1] if exact else files[-1]
        df = pd.read_csv(path)
        missing = set(OM_VARS + ["timestamp", "site_id"]) - set(df.columns)
        if missing:
            raise DataSourceError(f"{path.name} missing columns {sorted(missing)}")

        def grid(ids: List[str], col: str) -> np.ndarray:
            cols = []
            for sid in ids:
                s = df[df.site_id == sid].sort_values("timestamp")[col].to_numpy(dtype=float)
                if len(s) != config.ticks_per_day:
                    raise DataSourceError(f"{path.name}: site {sid} has {len(s)} rows")
                cols.append(s)
            return np.nan_to_num(np.column_stack(cols))

        s_ids, w_ids = [f.id for f in config.solar], [f.id for f in config.wind]
        day_str = str(df.timestamp.iloc[0])[:10]
        return WeatherDay(self.name, day_str, grid(s_ids, "shortwave_radiation"), grid(s_ids, "cloud_cover"),
                          grid(s_ids, "weather_code").astype(int), grid(w_ids, "wind_speed_100m"),
                          grid(w_ids, "weather_code").astype(int), notes=[f"replaying {path.name}"])


# ---------------------------------------------------------------- selection & fallback
def make_provider(name: str, config: Config) -> WeatherProvider:
    """Instantiate a provider by name using paths from the config."""
    d = config.data
    if name == "openmeteo":
        return OpenMeteoProvider(resolve_path(d.get("cache_dir", "data/cache")), d.get("cache_ttl_minutes", 60),
                                 d.get("api_timeout_s", 10), resolve_path(d.get("replay_dir", "data/replay")))
    if name == "lastyear":
        return LastYearProvider(resolve_path(d.get("cache_dir", "data/cache")), d.get("cache_ttl_minutes", 60),
                                d.get("api_timeout_s", 10), resolve_path(d.get("replay_dir", "data/replay")))
    if name == "replay":
        return ReplayProvider(resolve_path(d.get("replay_dir", "data/replay")))
    if name == "synthetic":
        return SyntheticProvider()
    raise ValueError(f"unknown data source {name!r}; choose from {SOURCES}")


def load_weather_day(source: str, config: Config, day: Optional[Date] = None, seed: int = 0,
                     providers: Optional[Dict[str, WeatherProvider]] = None) -> WeatherDay:
    """Load weather from ``source``, falling back Live → Last year → Replay → Synthetic.

    The returned :attr:`WeatherDay.source` names the provider that actually succeeded and
    :attr:`WeatherDay.notes` explains any fallback.
    """
    day = day or today_ist()
    chain = SOURCES[SOURCES.index(source):] if source in SOURCES else SOURCES
    notes: List[str] = []
    for name in chain:
        provider = (providers or {}).get(name) or make_provider(name, config)
        try:
            wd = provider.get_day(config, day, seed)
            wd.notes = notes + wd.notes
            if name != source:
                wd.notes.append(f"requested '{source}', using '{name}'")
            return wd
        except DataSourceError as exc:
            log.warning("%s unavailable: %s", name, exc)
            notes.append(f"{name} failed: {exc}")
    raise DataSourceError("all weather sources failed")  # unreachable: synthetic never fails


def parse_open_meteo(payload: Any, config: Config, day: Date) -> WeatherDay:
    """Convert an Open-Meteo multi-location response into a :class:`WeatherDay`."""
    locs = payload if isinstance(payload, list) else [payload]
    n_s, n_w = len(config.solar), len(config.wind)
    if len(locs) != n_s + n_w:
        raise DataSourceError(f"expected {n_s + n_w} locations, got {len(locs)}")
    T = config.ticks_per_day
    series: Dict[str, List[np.ndarray]] = {v: [] for v in OM_VARS}
    for loc in locs:
        block = loc.get("minutely_15") or {}
        for v in OM_VARS:
            vals = block.get(v)
            if vals is None or len(vals) < T:
                raise DataSourceError(f"minutely_15.{v} missing or short ({0 if vals is None else len(vals)})")
            arr = pd.Series(np.array(vals[:T], dtype=float)).interpolate(limit_direction="both").fillna(0)
            series[v].append(arr.to_numpy())

    def cols(v: str, sl: slice) -> np.ndarray:
        return np.column_stack(series[v][sl])

    s, w = slice(0, n_s), slice(n_s, n_s + n_w)
    return WeatherDay("openmeteo", day.isoformat(), cols("shortwave_radiation", s), cols("cloud_cover", s),
                      cols("weather_code", s).astype(int), cols("wind_speed_100m", w),
                      cols("weather_code", w).astype(int))


def save_replay_csv(wd: WeatherDay, config: Config, replay_dir: Path) -> Path:
    """Persist a day in long format so :class:`ReplayProvider` can replay it later."""
    replay_dir.mkdir(parents=True, exist_ok=True)
    base = datetime.fromisoformat(wd.date)
    stamps = [(base + timedelta(minutes=15 * t)).strftime("%Y-%m-%dT%H:%M") for t in range(wd.ticks)]
    rows = []
    for i, f in enumerate(config.solar):
        rows.append(pd.DataFrame({"timestamp": stamps, "site_id": f.id, "shortwave_radiation": wd.irradiance[:, i],
                                  "wind_speed_100m": np.nan, "cloud_cover": wd.cloud_cover[:, i],
                                  "weather_code": wd.solar_codes[:, i]}))
    for j, f in enumerate(config.wind):
        rows.append(pd.DataFrame({"timestamp": stamps, "site_id": f.id, "shortwave_radiation": np.nan,
                                  "wind_speed_100m": wd.wind_speed[:, j], "cloud_cover": np.nan,
                                  "weather_code": wd.wind_codes[:, j]}))
    path = replay_dir / f"{wd.source}_{wd.date}.csv"
    pd.concat(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- alerts
def weather_alerts(wd: WeatherDay, config: Config, tick: int, horizon: int = 16) -> List[Alert]:
    """Storm alerts from WMO weather codes and hub-height wind thresholds within the horizon."""
    codes = set(config.data.get("storm_weather_codes", [95, 96, 99, 65, 67, 82]))
    wind_thr = float(config.data.get("storm_wind_ms", 20.0))
    end = min(wd.ticks, tick + horizon)
    alerts: List[Alert] = []
    sites = [(f.name, wd.solar_codes[:, i], None) for i, f in enumerate(config.solar)]
    sites += [(f.name, wd.wind_codes[:, j], wd.wind_speed[:, j]) for j, f in enumerate(config.wind)]
    for name, code_s, speed_s in sites:
        for t in range(tick, end):
            storm_code = int(code_s[t]) in codes
            high_wind = speed_s is not None and speed_s[t] >= wind_thr
            if storm_code or high_wind:
                why = f"WMO code {int(code_s[t])}" if storm_code else f"wind {speed_s[t]:.1f} m/s"
                lead = t - tick
                end_t = t
                while end_t < wd.ticks and (int(code_s[end_t]) in codes or
                                            (speed_s is not None and speed_s[end_t] >= wind_thr)):
                    end_t += 1
                prob = round(max(0.4, 0.9 - 0.03 * lead), 2)   # forecast confidence decays with lead time
                alerts.append(Alert("storm_alert", tick, f"Storm risk {prob:.0%} at {name} ({why}) in {lead * 15} min",
                                    "critical" if lead <= 4 else "warning", lead, prob, max(1, end_t - t)))
                break
    return alerts


# ---------------------------------------------------------------- IEX prices
def ensure_iex_csv(path: Path, days: int = 30, seed: int = 7, end: Optional[Date] = None) -> Path:
    """Create a sample IEX day-ahead price file (15-min blocks) if ``path`` does not exist."""
    if path.exists():
        return path
    rng = np.random.default_rng(seed)
    end = end or today_ist()
    h = np.arange(96) / 4.0
    shape = (5200 - 2600 * np.exp(-((h - 12.5) / 2.6) ** 2)            # solar-hour dip
             + 4300 * np.exp(-((h - 19.75) / 1.6) ** 2)                 # evening peak
             + 900 * np.exp(-((h - 8.0) / 1.5) ** 2))                   # morning shoulder
    frames = []
    for d in range(days):
        day = end - timedelta(days=days - 1 - d)          # window ends today
        level = rng.uniform(0.8, 1.2)
        noise = _ar1(rng, 96, phi=0.85, sigma=220, mean=0)
        mcp = np.clip(shape * level + noise, 1500, 10000)              # IEX price cap ₹10,000/MWh
        frames.append(pd.DataFrame({
            "date": day.isoformat(), "block": np.arange(1, 97),
            "time_from": [f"{int(x // 1):02d}:{int(round(x % 1 * 60)):02d}" for x in h],
            "mcp_rs_mwh": mcp.round(2)}))
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat(frames)
    path.write_text("# SAMPLE data shaped like IEX DAM 15-min MCP — replace with a real IEX export\n"
                    + df.to_csv(index=False))
    return path


def load_iex_prices(path: Path, day: Optional[Date] = None, seed: int = 0) -> Tuple[np.ndarray, str]:
    """Return 96 market clearing prices (₹/MWh) for ``day`` (or a seeded random day) and its date."""
    ensure_iex_csv(path)
    df = pd.read_csv(path, comment="#")
    df = df.sort_values(["date", "block"])
    dates = sorted(df.date.unique())
    pick = day.isoformat() if day is not None and day.isoformat() in dates else \
        dates[int(np.random.default_rng(seed).integers(len(dates)))]
    prices = df[df.date == pick].mcp_rs_mwh.to_numpy(dtype=float)
    if len(prices) != 96:
        raise DataSourceError(f"IEX file has {len(prices)} blocks for {pick}")
    return prices, pick


# ---------------------------------------------------------------- helpers
def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float, mean: float) -> np.ndarray:
    x = np.empty(n)
    x[0] = rng.normal(0, sigma / np.sqrt(1 - phi ** 2))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sigma)
    return mean + x


def _codes_from_cloud(cloud: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    codes = np.select([cloud < 20, cloud < 50, cloud < 80], [0, 1, 2], default=3)
    rain = (cloud > 90) & (rng.random(cloud.shape) < 0.3)
    return np.where(rain, 61, codes).astype(int)


def _requests_get(url: str, **kw: Any) -> Any:
    import requests  # local import keeps tests independent of the network stack
    return requests.get(url, **kw)


def _one_year_earlier(day: Date) -> Date:
    try:
        return day.replace(year=day.year - 1)
    except ValueError:  # 29 Feb
        return day.replace(year=day.year - 1, day=28)


def _hourly_to_15min(payload: Any) -> Any:
    """Convert an archive (hourly) response into the minutely_15 shape :func:`parse_open_meteo` expects."""
    locs = payload if isinstance(payload, list) else [payload]
    out = []
    for loc in locs:
        h = loc.get("hourly") or {}
        if not h.get("time"):
            raise DataSourceError("archive response has no hourly data")
        n = len(h["time"])
        x_h, x_q = np.arange(n) * 4.0, np.arange(n * 4)
        block: Dict[str, Any] = {"time": [f"{t[:-2]}{m:02d}" for t in h["time"] for m in (0, 15, 30, 45)]}
        for v in OM_VARS:
            vals = pd.Series(np.array(h.get(v) or [np.nan] * n, dtype=float)).interpolate(limit_direction="both")
            arr = np.interp(x_q, x_h, vals.fillna(0).to_numpy())
            block[v] = (np.repeat(vals.fillna(0).to_numpy(), 4) if v == "weather_code" else arr).tolist()
        out.append({**{k: v for k, v in loc.items() if k != "hourly"}, "minutely_15": block})
    return out
