from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.data_sources import (DataSourceError, LastYearProvider, OpenMeteoProvider, ReplayProvider, SyntheticProvider, ensure_iex_csv,
                               load_iex_prices, load_weather_day, parse_open_meteo, save_replay_csv, weather_alerts)
from core.simulator import Simulator

DAY = date(2026, 9, 20)


def fake_payload(cfg, storm_site=None, wind=9.0):
    """Open-Meteo-shaped multi-location response (one dict per farm, 96 × 15-min values)."""
    hours = np.arange(96) / 4
    ghi = np.clip(900 * np.sin(np.pi * (hours - 6.2) / 12.4), 0, None).round(1)
    locs = []
    for k in range(len(cfg.solar) + len(cfg.wind)):
        codes = [0] * 96
        if storm_site == k:
            codes[60:70] = [95] * 10
        locs.append({"latitude": 0, "longitude": 0, "minutely_15": {
            "time": [f"{DAY}T{int(h):02d}:{int(h % 1 * 60):02d}" for h in hours],
            "shortwave_radiation": ghi.tolist(), "wind_speed_100m": [wind] * 95 + [None],
            "cloud_cover": [10] * 96, "weather_code": codes}})
    return locs


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self.payload


class FakeHTTP:
    def __init__(self, payload=None, exc=None, status=200):
        self.payload, self.exc, self.status, self.calls = payload, exc, status, []

    def __call__(self, url, params=None, timeout=None):
        self.calls.append(params)
        if self.exc:
            raise self.exc
        return FakeResponse(self.payload, self.status)


def _om(tmp_path, http):
    return OpenMeteoProvider(cache_dir=tmp_path / "cache", replay_dir=tmp_path / "replay", http_get=http)


def _ly(tmp_path, http):
    return LastYearProvider(cache_dir=tmp_path / "cache_ly", replay_dir=tmp_path / "replay_ly", http_get=http)


def fake_hourly(cfg):
    """Open-Meteo archive-shaped response (24 hourly values per site)."""
    h = np.arange(24)
    ghi = np.clip(850 * np.sin(np.pi * (h - 6.2) / 12.4), 0, None).round(1).tolist()
    return [{"latitude": 0, "longitude": 0, "hourly": {
        "time": [f"2025-09-20T{x:02d}:00" for x in h], "shortwave_radiation": ghi,
        "wind_speed_100m": [7.0] * 24, "cloud_cover": [20] * 24, "weather_code": [1] * 24}}
        for _ in range(len(cfg.solar) + len(cfg.wind))]


def test_live_failure_uses_last_year_proxy(cfg, tmp_path):
    http = FakeHTTP(fake_hourly(cfg))
    providers = {"openmeteo": _om(tmp_path, FakeHTTP(exc=ConnectionError("offline"))),
                 "lastyear": _ly(tmp_path, http)}
    wd = load_weather_day("openmeteo", cfg, DAY, providers=providers)
    assert wd.source == "lastyear" and wd.date == "2025-09-20"
    assert http.calls[0]["start_date"] == "2025-09-20" and "hourly" in http.calls[0]
    assert wd.irradiance.shape == (96, len(cfg.solar)) and wd.irradiance[48].min() > 700
    assert any("one year earlier" in n for n in wd.notes)


def test_open_meteo_parse_and_request(cfg, tmp_path):
    http = FakeHTTP(fake_payload(cfg))
    wd = _om(tmp_path, http).get_day(cfg, DAY)
    assert wd.source == "openmeteo" and wd.irradiance.shape == (96, 5) and wd.wind_speed.shape == (96, 3)
    assert wd.wind_speed[-1, 0] == pytest.approx(9.0)          # None interpolated
    p = http.calls[0]
    assert p["minutely_15"] == "shortwave_radiation,wind_speed_100m,cloud_cover,weather_code"
    assert len(p["latitude"].split(",")) == 8 and p["start_date"] == str(DAY)
    assert (tmp_path / "replay" / f"openmeteo_{DAY}.csv").exists()   # snapshot for replay


def test_open_meteo_responses_are_cached(cfg, tmp_path):
    http = FakeHTTP(fake_payload(cfg))
    prov = _om(tmp_path, http)
    prov.get_day(cfg, DAY)
    prov.get_day(cfg, DAY)
    assert len(http.calls) == 1 and prov.last_from_cache
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_expired_cache_refetches_for_today(cfg, tmp_path):
    http = FakeHTTP(fake_payload(cfg))
    prov = OpenMeteoProvider(cache_dir=tmp_path / "c", replay_dir=tmp_path / "r", http_get=http, ttl_minutes=0)
    prov.get_day(cfg, date(2099, 1, 1))  # future date → TTL applies
    prov.get_day(cfg, date(2099, 1, 1))
    assert len(http.calls) == 2


@pytest.mark.parametrize("http", [FakeHTTP(exc=ConnectionError("offline")), FakeHTTP({}, status=503),
                                  FakeHTTP({"error": True, "reason": "bad variable"})])
def test_api_failure_falls_back_to_replay(cfg, tmp_path, http):
    replay_dir = tmp_path / "replay"
    save_replay_csv(SyntheticProvider().get_day(cfg, DAY, seed=3), cfg, replay_dir)
    providers = {"openmeteo": _om(tmp_path, http), "lastyear": _ly(tmp_path, FakeHTTP(exc=OSError("down"))),
                 "replay": ReplayProvider(replay_dir)}
    wd = load_weather_day("openmeteo", cfg, DAY, providers=providers)
    assert wd.source == "replay"
    assert any("openmeteo failed" in n for n in wd.notes) and any("using 'replay'" in n for n in wd.notes)


def test_falls_back_to_synthetic_when_no_replay(cfg, tmp_path):
    providers = {"openmeteo": _om(tmp_path, FakeHTTP(exc=TimeoutError("slow"))),
                 "lastyear": _ly(tmp_path, FakeHTTP(exc=TimeoutError("slow"))),
                 "replay": ReplayProvider(tmp_path / "empty")}
    wd = load_weather_day("openmeteo", cfg, DAY, providers=providers)
    assert wd.source == "synthetic" and len(wd.notes) >= 4


def test_malformed_payload_rejected(cfg):
    bad = fake_payload(cfg)[:3]
    with pytest.raises(DataSourceError):
        parse_open_meteo(bad, cfg, DAY)


def test_replay_roundtrip(cfg, tmp_path):
    orig = parse_open_meteo(fake_payload(cfg), cfg, DAY)
    save_replay_csv(orig, cfg, tmp_path)
    back = ReplayProvider(tmp_path).get_day(cfg, DAY)
    assert np.allclose(back.irradiance, orig.irradiance) and np.allclose(back.wind_speed, orig.wind_speed)
    assert back.date == str(DAY)


def test_irradiance_and_wind_converted_to_mw(cfg):
    wd = parse_open_meteo(fake_payload(cfg, wind=12.5), cfg, DAY)
    sim = Simulator(cfg, weather=wd, day=DAY)
    noon = sim.profile.solar[50]
    caps = np.array([f.capacity_mw for f in cfg.solar])
    assert np.all(noon > 0.7 * caps) and np.all(noon <= caps)
    assert sim.profile.solar[8].sum() == 0                                         # 02:00 dark
    assert sim.profile.wind[10, 0] == pytest.approx(cfg.wind[0].capacity_mw)     # above rated
    assert sim.profile.price_source.startswith("IEX")


def test_storm_alerts_from_weather_code_and_wind(cfg):
    wd = parse_open_meteo(fake_payload(cfg, storm_site=0), cfg, DAY)
    alerts = weather_alerts(wd, cfg, tick=56, horizon=16)
    assert len(alerts) == 1 and "WMO code 95" in alerts[0].message and alerts[0].eta_ticks == 4
    assert 0.4 <= alerts[0].probability <= 0.9 and alerts[0].duration_ticks == 10
    windy = parse_open_meteo(fake_payload(cfg, wind=22.0), cfg, DAY)
    assert sum("wind 22.0" in a.message for a in weather_alerts(windy, cfg, 0)) == len(cfg.wind)
    assert weather_alerts(parse_open_meteo(fake_payload(cfg), cfg, DAY), cfg, 0) == []


def test_storm_code_raises_agent_storm_signal(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from agent.agent import OrchestratorAgent
    wd = parse_open_meteo(fake_payload(cfg, storm_site=6), cfg, DAY)
    sim = Simulator(cfg, weather=wd, day=DAY)
    sim.tick = 55
    d = OrchestratorAgent(sim, cfg, log_path=None).step()
    assert d.signals["storm_prob"] > 0.5 and d.signals["storm_eta_min"] == 75


def test_iex_sample_generated_and_loaded(tmp_path):
    path = tmp_path / "iex.csv"
    ensure_iex_csv(path, days=5, end=DAY)
    df = pd.read_csv(path, comment="#")
    assert len(df) == 5 * 96 and set(df.block) == set(range(1, 97))
    prices, picked = load_iex_prices(path, date(2026, 9, 17))
    assert picked == "2026-09-17" and prices.shape == (96,) and 1500 <= prices.min() <= prices.max() <= 10000
    assert np.argmax(prices) > 60                                                  # evening peak
    _, other = load_iex_prices(path, date(1999, 1, 1), seed=1)                     # unknown date → a stored day
    assert other in set(df.date)
