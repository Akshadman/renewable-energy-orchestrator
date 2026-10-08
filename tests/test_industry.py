from datetime import date

import numpy as np
import pytest

from agent.agent import OrchestratorAgent
from core.config import config_from_dict, config_to_dict, load_config, save_profile, validate_config
from core.simulator import Simulator, clock, tick_of


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _raw():
    return config_to_dict(load_config())


def test_time_helpers():
    assert tick_of("14:30") == 58 and clock(58) == "14:30" and tick_of(10) == 10


def test_custom_consumption_profile_drives_demand():
    raw = _raw()
    raw["custom_profiles"] = {"C1": [40.0] * 48 + [80.0] * 48}
    sim = Simulator(config_from_dict(raw), seed=1)
    d = sim.profile.demand[:, 0]
    assert d[:40].mean() == pytest.approx(40, rel=0.08) and d[60:].mean() == pytest.approx(80, rel=0.08)


def test_conservation_window_reduces_demand():
    raw = _raw()
    raw["schedules"] = {"maintenance": [], "conservation": [{"start": "10:00", "end": "12:00", "reduction_pct": 20}]}
    a = Simulator(config_from_dict(raw), seed=1).profile.demand.sum(1)
    raw["schedules"]["conservation"] = []
    b = Simulator(config_from_dict(raw), seed=1).profile.demand.sum(1)
    assert a[40:48] == pytest.approx(b[40:48] * 0.8) and a[0] == pytest.approx(b[0])


def test_scheduled_breakdown_and_maintenance():
    raw = _raw()
    raw["schedules"] = {"conservation": [], "maintenance": [
        {"asset": "S2", "start": "11:00", "hours": 2, "kind": "breakdown", "severity": 0.5},
        {"asset": "B1", "start": "15:00", "hours": 1, "kind": "maintenance", "flexible_hours": 2}]}
    sim = Simulator(config_from_dict(raw), seed=1)
    assert sim.solar_avail[44:52, 1].tolist() == [0.5] * 8 and sim.solar_avail[52, 1] == 1
    assert len(sim.maintenance) == 1 and sim.maintenance[0].window == 8
    sim.tick = 45
    assert any(a.kind == "asset_outage" for a in sim.alerts())


def test_portfolio_without_wind_or_batteries_runs(tmp_path):
    raw = _raw()
    raw["wind_farms"], raw["batteries"] = [], []
    raw["schedules"] = {"maintenance": [], "conservation": []}
    raw["grid"]["import_limit_mw"] = 200          # solar-only site needs a bigger grid connection at night
    cfg = config_from_dict(raw)
    assert validate_config(cfg) == []
    sim = Simulator(cfg, seed=1)
    agent = OrchestratorAgent(sim, cfg, log_path=None)
    for _ in range(12):
        d = agent.step()
    assert d.validation["passed"] and sim.totals()["violations"] == 0


def test_validation_messages():
    raw = _raw()
    raw["solar_farms"][0]["capacity_mw"] = 0
    raw["consumers"][0]["critical_frac"] = 1.5
    raw["wind_farms"][0]["id"] = "S2"
    errs = validate_config(config_from_dict(raw))
    assert len(errs) == 3


def test_profile_roundtrip(tmp_path, monkeypatch):
    import core.config as C
    monkeypatch.setattr(C, "PROFILE_DIR", tmp_path)
    cfg = load_config()
    cfg.name = "Cement plant – Rajasthan"
    path = save_profile(cfg)
    again = load_config(path)
    assert again.name == cfg.name and len(again.solar) == len(cfg.solar) and again.targets == cfg.targets
    assert "Cement plant – Rajasthan" in C.list_profiles()
