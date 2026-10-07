import numpy as np
import pytest

from core.baseline import NaiveController
from core.events import make_event
from core.fallback import safe_action
from core.forecaster import Forecaster
from core.optimizer import Criteria, optimize
from core.simulator import Simulator
from core.validator import validate


def test_optimizer_output_passes_validator(sim, cfg):
    for tick in (0, 40, 52, 78):
        sim.tick = tick
        res = optimize(sim.state(), Forecaster(sim).forecast(), cfg, Criteria())
        vr = validate(res.action, sim.state(), cfg)
        assert vr.passed, vr.violations


@pytest.mark.parametrize("mutation, needle", [
    (lambda a, s: a.discharge_mw.__setitem__(0, 999), "rate limit"),
    (lambda a, s: setattr(a, "grid_import_mw", s.import_limit_mw + 50), "line limit"),
    (lambda a, s: a.unserved_critical_mw.__setitem__(0, 5), "critical load"),
    (lambda a, s: setattr(a, "grid_export_mw", a.grid_export_mw + 30), "power balance"),
    (lambda a, s: a.dr_mw.__setitem__(3, 1.0), "ineligible"),
])
def test_validator_catches_violations(sim, cfg, mutation, needle):
    st = sim.state()
    a = safe_action(st, cfg)
    assert validate(a, st, cfg).passed
    mutation(a, st)
    vr = validate(a, st, cfg)
    assert not vr.passed and any(needle in v for v in vr.violations), vr.violations


def test_validator_blocks_unavailable_battery(cfg):
    sim = Simulator(cfg, seed=3)
    sim.apply_event(make_event("battery_unavailable", 0, battery=1))
    st = sim.state()
    a = safe_action(st, cfg)
    a.charge_mw[1] = 5.0
    a.grid_import_mw += 5.0
    assert any("unavailable" in v for v in validate(a, st, cfg).violations)


def test_fallback_always_valid_over_stressed_day(cfg):
    sim = Simulator(cfg, seed=8)
    for ev in [make_event("line_congestion", 10, duration=30), make_event("battery_unavailable", 20, battery=0),
               make_event("demand_surge", 60), make_event("storm_alert", 40)]:
        sim.apply_event(ev)
    while not sim.done:
        st = sim.state()
        a = safe_action(st, cfg)
        vr = validate(a, st, cfg)
        # only a genuine supply shortage may leave critical load unserved
        assert vr.passed or all("critical" in v for v in vr.violations), vr.violations
        sim.step(a)
    assert sum(k.violations for k in sim.kpis) == 0 or sim.totals()["unserved_critical_mwh"] > 0


def test_baseline_violates_under_congestion(cfg):
    sim = Simulator(cfg, seed=1)
    sim.apply_event(make_event("line_congestion", 0, duration=96, factor=0.3))
    ctrl = NaiveController(cfg)
    while not sim.done:
        sim.step(ctrl.decide(sim.state()))
    assert sim.totals()["violations"] > 0


def test_no_battery_baseline_never_uses_battery(sim, cfg):
    ctrl = NaiveController(cfg, use_battery=False)
    for _ in range(96):
        a = ctrl.decide(sim.state())
        assert a.charge_mw.sum() == 0 and a.discharge_mw.sum() == 0
        sim.step(a)
    assert np.allclose([h["charge_mw"] for h in sim.history][:1], 0)
