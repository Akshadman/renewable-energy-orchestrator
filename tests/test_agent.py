import json
from types import SimpleNamespace

import pytest

from agent.agent import LLMReasoner, OrchestratorAgent
from agent.recipes import DEFAULT_RECIPES, PARAMS, load_recipes
from agent.runner import Operation
from core.events import make_event
from core.simulator import Simulator


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _agent(cfg, tmp_path, seed=1, **kw):
    sim = Simulator(cfg, seed=seed)
    return sim, OrchestratorAgent(sim, cfg, log_path=tmp_path / "log.jsonl", **kw)


def _advance(agent, n):
    for _ in range(n):
        agent.step()


def test_full_autonomous_day(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path)
    assert agent.mode == "offline"
    for ev in (make_event("cloud_cover", 30), make_event("price_spike", 70), make_event("line_congestion", 45),
               make_event("generator_breakdown", 55, asset="S1")):
        sim.apply_event(ev)
    decisions = agent.run_day()
    assert len(decisions) == 96 and sim.done
    assert sim.totals()["violations"] == 0
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == 96
    rec = json.loads(lines[-1])
    assert len(rec["options"]) >= len(DEFAULT_RECIPES) and rec["explanation"] and rec["validation"]["passed"]
    assert not hasattr(agent, "pending_approvals")            # no human in the loop


def test_every_option_is_stress_tested_and_ranked(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path)
    d = agent.step()
    tested = [o for o in d.options if o["cost_avg_rs"] is not None]
    assert len(tested) >= len(DEFAULT_RECIPES) * 2
    assert all(0 <= o["p_shortfall"] <= 1 and o["cost_p95_rs"] >= o["cost_avg_rs"] - 1e-6 for o in tested)
    assert sum(o["reason"] == "chosen" for o in d.options) == 1


def test_band_search_stays_inside_bands_and_refines(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path)
    d = agent.step()
    bands = {r.name: r.bands for r in load_recipes(cfg)}
    assert {o["stage"] for o in d.options} == {"explore", "refine"}
    assert {o["name"] for o in d.options} == set(bands)
    for o in d.options:                                   # reserve & caution & DR stay inside the recipe band
        for p in ("reserve_pct", "caution", "max_dr_frac"):
            lo, hi = bands[o["name"]][p]
            assert lo - 1e-6 <= o[p] <= hi + 1e-6, (o["name"], p, o[p])
    reserves = {round(o["reserve_pct"], 3) for o in d.options}
    assert len(reserves) > len(bands)                     # many distinct points, not one per recipe


def test_industry_can_define_own_recipe_bands(cfg, tmp_path):
    cfg.agent["recipes"] = [{"name": "Only cautious", "reserve_pct": [0.5, 0.6], "w_cost": [0.3, 0.3],
                             "w_clean": [0.2, 0.2], "w_reliability": [0.5, 0.5], "caution": [0.5, 0.5],
                             "max_dr_frac": [1, 1]}]
    try:
        sim, agent = _agent(cfg, tmp_path)
        d = agent.step()
        assert {o["name"] for o in d.options} == {"Only cautious"}
        assert all(0.5 - 1e-6 <= o["reserve_pct"] <= 0.6 + 1e-6 for o in d.options)
    finally:
        cfg.agent.pop("recipes")


def _protection_for(cfg, tmp_path, prob, tick=72):
    sim, agent = _agent(cfg, tmp_path)
    _advance(agent, tick)
    if prob:
        sim.apply_event(make_event("storm_alert", tick, probability=prob, hits=False))
    d = agent.step()
    return d.criteria["reserve_pct"], d


def test_protection_rises_with_storm_probability(cfg, tmp_path):
    reserves = [_protection_for(cfg, tmp_path, p)[0] for p in (0.0, 0.3, 0.9)]
    assert reserves[0] < reserves[1] <= reserves[2]
    assert reserves[2] > reserves[0]


def test_all_signals_kept_together(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path)
    _advance(agent, 60)
    for ev in (make_event("storm_alert", 60, probability=0.35, hits=False), make_event("price_spike", 60, factor=4.0),
               make_event("battery_unavailable", 60, battery=0)):
        sim.apply_event(ev)
    d = agent.step()
    s = d.signals
    assert s["storm_prob"] == pytest.approx(0.35) and s["price_ratio"] > 1.4 and s["battery_available"] == 0.5
    text = " ".join(t for t, _ in d.headline)
    assert "Storm" in text and "Price" in text and "Batteries" in text   # nothing dropped
    assert d.trigger == "shock"


def test_breakdown_triggers_autonomous_crew(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path)
    _advance(agent, 40)
    sim.apply_event(make_event("generator_breakdown", 40, duration=20, asset="W1"))
    d = agent.step()
    assert any("Crew dispatched to W1" in n for n in d.notes)
    outage = next(o for o in sim.outages if o["asset"] == "W1")
    assert outage["end"] == 44                           # repaired early


def test_maintenance_moved_without_approval(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path)
    sim.apply_event(make_event("maintenance_window", 0, duration=8, asset="S1", lead=40, window=24))
    _advance(agent, 30)
    m = sim.maintenance[-1]
    assert m.scheduled_start is not None
    assert m.scheduled_start != 40 or sim.profile.solar[40:48, 0].sum() == 0


def test_learning_reduces_forecast_error(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path)
    agent.run_day()
    rep = agent.learner.report()
    assert rep["wind"]["correction_pct"] < -4                    # wind model over-forecasts by 12%
    assert rep["demand"]["correction_pct"] < -1                  # demand over-forecast by 3%
    assert rep["demand"]["mae_learned_mw"] < rep["demand"]["mae_raw_mw"]


def test_learning_persists_between_days(cfg, tmp_path):
    cfg.data["learning_file"] = str(tmp_path / "learn.json")
    sim, agent = _agent(cfg, tmp_path, persist_learning=True)
    agent.run_day()
    sim2, agent2 = _agent(cfg, tmp_path, seed=2, persist_learning=True)
    assert agent2.learner.days_trained == 1 and agent2.learner.ratio["wind"] < 0.97
    cfg.data.pop("learning_file")


def test_operation_measures_savings_vs_baseline(cfg, tmp_path):
    sim = Simulator(cfg, seed=3)
    op = Operation(cfg, sim, use_llm=False, persist_learning=False, log_path=None)
    op.inject(make_event("storm_alert", 0, probability=1.0))
    for _ in range(24):
        op.step()
    out = op.outputs()
    assert op.base_sim.tick == op.sim.tick == 24
    assert op.base_sim.events[0].params["hits"] == op.sim.events[0].params["hits"]
    assert {"money_saved_rs", "carbon_saved_t", "generation_efficiency_pct", "energy_produced_mwh"} <= set(out)


# ---------------------------------------------------------------- scripted fake LLM (any provider)
class FakeClient:
    """Stands in for agent.llm.LLMClient: reads the briefing, optionally proposes a plan, then decides."""

    label = "fake:model"

    def __init__(self, pick=None, fail=False, propose=None):
        self.pick, self.fail, self.propose, self.calls, self.proposal_results = pick, fail, propose, 0, []

    def run_tools(self, system, user, tools, handler, final_tool, max_turns=4, force_first=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("API down")
        assert {t["name"] for t in tools} == {"propose_plan", "submit_decision"}
        brief = json.loads(user.split("(JSON):\n", 1)[1].split("\n\nOptionally", 1)[0])
        if self.propose:
            self.proposal_results.append(handler("propose_plan", self.propose))
        choice = self.pick(brief, self.proposal_results) if self.pick else brief["recommended_by_ladder"]
        return {"option_id": choice, "diagnosis": "Calm and cheap.", "explanation": "LLM explanation."}


def _fake_llm(client):
    return LLMReasoner(client)


def test_llm_picks_among_eligible_and_is_throttled(cfg, tmp_path):
    client = FakeClient(pick=lambda b, r: b["eligible"][-1])
    sim, agent = _agent(cfg, tmp_path, llm=_fake_llm(client))
    _advance(agent, 24)
    llm_ticks = [d for d in agent.decisions if d.reasoner.startswith("llm")]
    assert 3 <= len(llm_ticks) < 24
    assert llm_ticks[0].explanation == "LLM explanation." and llm_ticks[0].diagnosis == "Calm and cheap."
    assert agent.llm_status["state"] == "on" and agent.llm_status["ok"] == len(llm_ticks)


def test_llm_can_propose_and_win_with_its_own_plan(cfg, tmp_path):
    proposal = {"reserve_pct": 0.33, "w_cost": 0.5, "w_clean": 0.3, "w_reliability": 0.2, "caution": 0.2,
                "reason": "test"}

    def pick(brief, results):
        mine = results[-1]
        return mine["option_id"] if mine["status"] == "eligible" else brief["recommended_by_ladder"]
    client = FakeClient(pick=pick, propose=proposal)
    sim, agent = _agent(cfg, tmp_path, llm=_fake_llm(client))
    d = agent.step()
    res = client.proposal_results[0]
    assert res["status"] in ("eligible", "too risky", "invalid") and "row" in res     # evaluated like any plan
    mine = next(o for o in d.options if o["stage"] == "llm")
    assert abs(mine["reserve_pct"] - 0.33) < 1e-6
    if res["status"] == "eligible":
        assert d.chosen == mine["id"] and agent.llm_status["proposals_won"] == 1


def test_llm_cannot_choose_ineligible_plan(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path, llm=_fake_llm(FakeClient(pick=lambda b, r: "O99")))
    d = agent.step()
    assert "non-eligible" in d.llm_error and d.chosen != "O99"


def test_llm_outage_keeps_running_and_is_visible(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path, llm=_fake_llm(FakeClient(fail=True)))
    _advance(agent, 12)
    assert all(d.validation["passed"] for d in agent.decisions)
    assert any("API down" in d.llm_error for d in agent.decisions)
    assert agent.llm_status["state"] == "error" and "API down" in agent.llm_status["last_error"]


def test_operator_override_hands_control_to_safe_controller(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path)
    agent.step()
    agent.operator_override = True
    d = agent.step()
    assert d.plan_source == "operator override" and d.options == [] and d.validation["passed"]
    agent.operator_override = False
    assert agent.step().plan_source != "operator override"


def test_override_actions_are_flagged_for_review(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path)
    sim.apply_event(make_event("demand_surge", 0, duration=8, factor=4.0))     # forces the safe controller to shed
    sim.apply_event(make_event("line_congestion", 0, duration=8, factor=0.1))
    agent.operator_override = True
    d = agent.step()
    assert d.reasoner == "operator" and d.action_summary["shed_mw"] > 0
    assert any("Shed" in h["action"] and not h["acknowledged"] for h in agent.high_impact)
