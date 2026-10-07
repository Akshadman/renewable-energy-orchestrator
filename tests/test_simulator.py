import numpy as np
import pytest

from core.events import EVENTS, make_event
from core.models import Action
from core.power_curves import solar_mw, wind_mw
from core.simulator import Simulator


def _zero(cfg):
    return Action.zeros(len(cfg.solar), len(cfg.wind), len(cfg.batteries), len(cfg.consumers))


def _random_action(cfg, rng):
    a = _zero(cfg)
    a.charge_mw = rng.uniform(0, 60, len(cfg.batteries)) * rng.integers(0, 2, len(cfg.batteries))
    a.discharge_mw = rng.uniform(0, 60, len(cfg.batteries)) * (a.charge_mw == 0)
    a.grid_import_mw = float(rng.uniform(0, 150))
    a.grid_export_mw = float(rng.uniform(0, 100)) if a.grid_import_mw < 50 else 0.0
    return a


def test_profiles_have_expected_shape(sim, cfg):
    assert sim.profile.solar.shape == (96, len(cfg.solar))
    assert sim.profile.wind.shape == (96, len(cfg.wind))
    assert sim.profile.solar[:20].sum() == 0  # night before ~5am
    assert sim.profile.solar[48].sum() > 50   # midday sun


def test_energy_balance_every_tick(sim, cfg):
    rng = np.random.default_rng(0)
    while not sim.done:
        sim.step(_random_action(cfg, rng))
    for r in sim.history:
        supply = r["solar_mw"] + r["wind_mw"] + r["discharge_mw"] + r["import_mw"]
        sink = r["served_mw"] + r["charge_mw"] + r["export_mw"] + r["forced_curtail_mw"]
        assert supply == pytest.approx(sink, abs=1e-3), r["tick"]


def test_soc_stays_in_bounds_under_abusive_commands(sim, cfg):
    rng = np.random.default_rng(3)
    while not sim.done:
        a = _random_action(cfg, rng)
        a.charge_mw *= 5
        a.discharge_mw *= 5
        sim.step(a)
        for b, bat in enumerate(cfg.batteries):
            assert bat.min_mwh - 1e-5 <= sim.soc[b] <= bat.max_mwh + 1e-5
    assert sum(k.violations for k in sim.kpis) > 0  # abusive commands are counted


def test_seeded_reproducibility(cfg):
    a, b = Simulator(cfg, seed=5), Simulator(cfg, seed=5)
    assert np.allclose(a.profile.solar, b.profile.solar)
    assert np.allclose(a.profile.mcp, b.profile.mcp)
    assert not np.allclose(a.profile.wind, Simulator(cfg, seed=6).profile.wind)


@pytest.mark.parametrize("kind", list(EVENTS))
def test_every_event_applies(cfg, kind):
    sim = Simulator(cfg, seed=2)
    before = (sim.solar_mult.copy(), sim.wind_mult.copy(), sim.demand_mult.copy(), sim.price_mult.copy(),
              sim.line_mult.copy(), sim.battery_avail.copy(), len(sim.maintenance))
    ev = sim.apply_event(make_event(kind, 40))
    after = (sim.solar_mult, sim.wind_mult, sim.demand_mult, sim.price_mult, sim.line_mult,
             sim.battery_avail, len(sim.maintenance))
    changed = any(not np.array_equal(x, y) for x, y in zip(before[:-1], after[:-1])) or before[-1] != after[-1]
    assert changed and ev.description
    assert any(a.kind == kind for a in sim.event_alerts)


def test_battery_unavailable_blocks_dispatch(cfg):
    sim = Simulator(cfg, seed=2)
    sim.apply_event(make_event("battery_unavailable", 0, battery=0))
    a = _zero(cfg)
    a.discharge_mw[0] = 10
    kpi = sim.step(a)
    assert any("unavailable" in n for n in kpi.violation_notes)


def test_power_curves():
    assert solar_mw(np.array([0, 500, 1200]), 60, 0.9).tolist() == pytest.approx([0, 27, 60])
    w = wind_mw(np.array([2, 7.5, 12, 18, 26]), 80, 3, 12, 25)
    assert w[0] == 0 and 0 < w[1] < 80 and w[2] == 80 and w[3] == 80 and w[4] == 0


def test_frequency_near_nominal_when_balanced(sim, cfg):
    st = sim.state()
    a = _zero(cfg)
    net = st.total_demand_mw - st.renewable_mw
    a.grid_import_mw = max(net, 0)
    a.grid_export_mw = max(-net, 0)
    kpi = sim.step(a)
    assert abs(kpi.frequency_hz - 50) < 0.05
