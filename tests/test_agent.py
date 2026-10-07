import json
from types import SimpleNamespace

import pytest

from agent.agent import LLMReasoner, OrchestratorAgent
from core.events import make_event
from core.simulator import Simulator


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _agent(cfg, tmp_path, seed=1, **kw):
    sim = Simulator(cfg, seed=seed)
    return sim, OrchestratorAgent(sim, cfg, log_path=tmp_path / "log.jsonl", **kw)


def test_full_day_without_api_key(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path)
    assert agent.mode == "mock"
    for ev in (make_event("cloud_cover", 30), make_event("price_spike", 70), make_event("line_congestion", 45)):
        sim.apply_event(ev)
    decisions = agent.run_day()
    assert len(decisions) == 96 and sim.done
    totals = sim.totals()
    assert totals["violations"] == 0
    assert totals["unserved_critical_mwh"] == 0
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == 96
    rec = json.loads(lines[-1])
    assert rec["explanation"] and rec["criteria"]["w_cost"] > 0 and rec["validation"]["passed"]
    assert {d.situation for d in decisions} >= {"normal"}


def test_storm_raises_reserve_and_uses_p10(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path)
    for _ in range(20):
        agent.step()
    sim.apply_event(make_event("storm_alert", sim.tick, lead=6))
    d = agent.step()
    assert d.situation == "storm_risk"
    assert d.criteria["uncertainty_mode"] == "p10" and d.criteria["reserve_pct"] >= 0.5


def test_relax_then_fallback_on_infeasible(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path, seed=4)
    for kind, kw in (("wind_drop", {"factor": 0.0}), ("battery_unavailable", {"battery": 0}),
                     ("battery_unavailable", {"battery": 1}), ("line_congestion", {"factor": 0.1})):
        sim.apply_event(make_event(kind, 0, duration=20, **kw))
    d = agent.step()
    assert len(d.attempts) >= 2 and d.attempts[0]["status"] == "infeasible"
    assert "relaxed" in d.plan_source or d.plan_source == "fallback"


def test_human_approval_gates_large_export(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path, auto_approve=False)
    sim.soc[:] = [b.max_mwh for b in cfg.batteries]
    sim.tick = 50                                   # midday surplus, batteries full
    d = agent.step()
    assert d.action_summary["export_mw"] <= agent.export_threshold + 1e-6
    pending = agent.pending_approvals()
    assert pending and pending[0].kind == "large_export"
    agent.resolve_approval(pending[0].id, True)
    d2 = agent.step()
    assert d2.action_summary["export_mw"] > agent.export_threshold


def test_maintenance_scheduled_to_low_impact_window(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path)
    sim.apply_event(make_event("maintenance_window", 0, duration=8, asset="S1", lead=40, window=24))
    agent.step()
    m = sim.maintenance[0]
    assert m.scheduled_start is not None
    assert m.scheduled_start != 40 or sim.profile.solar[40:48, 0].sum() == 0  # moved off peak sun


# ---------------------------------------------------------------- scripted fake Claude
class FakeClient:
    """Mimics anthropic.Anthropic().messages.create with a scripted tool-use conversation."""

    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kw):
        if self.fail:
            raise RuntimeError("API down")
        self.calls += 1
        msgs = kw["messages"]
        turn = sum(1 for m in msgs if m["role"] == "assistant")
        tu = lambda i, name, inp: SimpleNamespace(type="tool_use", id=f"t{self.calls}{i}", name=name, input=inp)  # noqa
        if turn == 0:
            return SimpleNamespace(stop_reason="tool_use", content=[tu(0, "get_state", {}), tu(1, "get_alerts", {})])
        if turn == 1:
            return SimpleNamespace(stop_reason="tool_use", content=[tu(0, "run_optimizer", {
                "situation": "normal", "w_cost": 0.6, "w_clean": 0.3, "w_reliability": 0.1, "reserve_pct": 0.25,
                "uncertainty_mode": "p50", "reasoning": "calm conditions"})])
        if turn == 2:
            plan_id = json.loads(msgs[-1]["content"][0]["content"])["plan_id"]
            return SimpleNamespace(stop_reason="tool_use", content=[tu(0, "validate_actions", {"plan_id": plan_id}),
                                                                    tu(1, "execute_actions", {"plan_id": plan_id})])
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text="Calm day; I leaned on cost.")])


def _fake_llm(client):
    llm = LLMReasoner.__new__(LLMReasoner)
    llm.client, llm.model, llm.max_turns = client, "fake", 10
    return llm


def test_full_day_with_llm_tool_use(cfg, tmp_path):
    client = FakeClient()
    sim, agent = _agent(cfg, tmp_path, llm=_fake_llm(client))
    assert agent.mode == "llm"
    decisions = agent.run_day()
    assert len(decisions) == 96 and sim.totals()["violations"] == 0
    llm_ticks = [d for d in decisions if d.reasoner == "llm"]
    assert 96 // agent.llm_every_n <= len(llm_ticks) < 96          # throttled, not every tick
    first = llm_ticks[0]
    assert first.explanation == "Calm day; I leaned on cost."
    assert first.criteria["reserve_pct"] == pytest.approx(0.25)
    assert [c["tool"] for c in first.tool_calls][:3] == ["get_state", "get_alerts", "run_optimizer"]
    assert any(d.reasoner == "llm-cached" for d in decisions)


def test_llm_outage_falls_back_to_mock(cfg, tmp_path):
    sim, agent = _agent(cfg, tmp_path, llm=_fake_llm(FakeClient(fail=True)))
    decisions = agent.run_day()
    assert sim.done and sim.totals()["violations"] == 0
    assert any(d.reasoner == "mock (llm error)" and "API down" in d.llm_error for d in decisions)
