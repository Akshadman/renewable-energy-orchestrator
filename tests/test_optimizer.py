import numpy as np
import pytest

from core.events import make_event
from core.forecaster import Forecaster
from core.optimizer import Criteria, optimize
from core.simulator import Simulator


def _solve(sim, cfg, **kw):
    fc = Forecaster(sim).forecast(16)
    return optimize(sim.state(), fc, cfg, Criteria(**kw)), fc


def test_forecast_quantiles_ordered_and_nowcast_exact(sim):
    sim.tick = 40
    fc = Forecaster(sim).forecast(16)
    for q in (fc.solar, fc.wind, fc.demand):
        assert np.all(q["p10"] <= q["p50"] + 1e-9) and np.all(q["p50"] <= q["p90"] + 1e-9)
    assert np.allclose(fc.solar["p50"][0], sim.state().solar_mw)
    spread = fc.wind["p90"] - fc.wind["p10"]
    assert spread[-1].sum() >= spread[1].sum()  # uncertainty grows with lead time


@pytest.mark.parametrize("tick", [0, 30, 50, 76])
def test_feasible_on_normal_day(sim, cfg, tick):
    sim.tick = tick
    res, _ = _solve(sim, cfg)
    assert res.status == "optimal", res.message
    st, a = sim.state(), res.action
    assert a.supply_mw(st) == pytest.approx(a.sink_mw(st), abs=1e-3)
    assert a.unserved_critical_mw.sum() == 0
    assert not np.any((a.charge_mw > 1e-6) & (a.discharge_mw > 1e-6))
    assert a.grid_import_mw <= st.import_limit_mw + 1e-6
    assert res.solve_time_s < 5


def test_reserve_respected(sim, cfg):
    sim.tick = 76  # evening peak: expensive, tempting to drain batteries
    res, _ = _solve(sim, cfg, w_cost=0.2, w_clean=0.1, w_reliability=0.7, reserve_pct=0.45)
    assert res.status == "optimal"
    floor = 0.45 * sum(b.energy_mwh for b in cfg.batteries) / sum(b.energy_mwh for b in cfg.batteries) * 100
    start_pct = sim.soc.sum() / sum(b.energy_mwh for b in cfg.batteries) * 100
    # never goes further below the floor than it started
    assert min(res.plan["soc_pct"]) >= min(floor, start_pct) - 0.5


def test_p10_mode_is_more_conservative(sim, cfg):
    sim.tick = 44
    p50, _ = _solve(sim, cfg, uncertainty_mode="p50")
    p10, _ = _solve(sim, cfg, uncertainty_mode="p10")
    assert p10.status == p50.status == "optimal"
    assert p10.plan["soc_pct"][-1] >= p50.plan["soc_pct"][-1] - 1e-6 or \
        sum(p10.plan["net_grid_mw"]) >= sum(p50.plan["net_grid_mw"]) - 1e-6


def test_infeasible_handled_without_crash(cfg):
    sim = Simulator(cfg, seed=4)
    sim.tick = 2  # night: no solar
    sim.apply_event(make_event("wind_drop", 0, duration=30, factor=0.0))
    sim.apply_event(make_event("battery_unavailable", 0, duration=30, battery=0))
    sim.apply_event(make_event("battery_unavailable", 0, duration=30, battery=1))
    sim.apply_event(make_event("line_congestion", 0, duration=30, factor=0.1))
    res, _ = _solve(sim, cfg)
    assert res.status == "infeasible" and res.action is None and res.message
    relaxed, _ = _solve(sim, cfg, **{k: v for k, v in Criteria().relaxed(2).to_dict().items()})
    assert relaxed.status == "optimal"
    assert relaxed.action.shed_mw.sum() + relaxed.action.unserved_critical_mw.sum() > 0


def test_price_spike_discharges(cfg):
    sim = Simulator(cfg, seed=1)
    sim.tick = 76  # evening deficit: discharge displaces spiked imports
    sim.soc[:] = [b.max_mwh for b in cfg.batteries]
    sim.apply_event(make_event("price_spike", 76, duration=4, factor=3.0))
    res, _ = _solve(sim, cfg, w_cost=0.8, w_clean=0.1, w_reliability=0.1, reserve_pct=0.1)
    assert res.action.discharge_mw.sum() > 10
