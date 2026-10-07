"""Orchestrator agent: perceive → assess → set criteria → optimize → validate → act → explain."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agent.mock_reasoner import MockReasoner
from agent.prompts import SYSTEM_PROMPT, tick_brief
from agent.tools import ToolBox
from core.config import ROOT
from core.fallback import safe_action
from core.forecaster import Forecaster
from core.models import Action, Config, Forecast, GridState, jsonable
from core.optimizer import Criteria, optimize
from core.simulator import Simulator
from core.validator import ValidationResult, validate

DEFAULT_LOG = ROOT / "logs" / "decisions.jsonl"
DEFAULT_MODEL = "claude-sonnet-5"


# ---------------------------------------------------------------- records
@dataclass
class Plan:
    """A candidate dispatch for the current tick with its provenance."""

    id: str
    tick: int
    criteria: Criteria
    action: Action
    source: str                      # optimizer | optimizer (relaxed xN) | fallback
    validation: ValidationResult
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    objective: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """What the LLM sees after run_optimizer."""
        return jsonable({"plan_id": self.id, "source": self.source, "attempts": self.attempts,
                         "objective": self.objective, "first_tick": describe_action(self.action),
                         "validation_passed": self.validation.passed})


@dataclass
class ApprovalRequest:
    id: str
    tick: int
    kind: str                        # load_shedding | large_export | delay_maintenance | custom
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"          # pending | approved | rejected | auto-approved


@dataclass
class Decision:
    """Everything that happened in one tick — logged as one JSONL line."""

    tick: int
    hour: float
    situation: str
    signals: List[str]
    reasoner: str
    criteria: Dict[str, Any]
    criteria_reason: str
    plan_source: str
    attempts: List[Dict[str, Any]]
    validation: Dict[str, Any]
    action: Dict[str, Any]
    action_summary: Dict[str, Any]
    approvals: List[str]
    notes: List[str]
    explanation: str
    kpi: Dict[str, Any]
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    llm_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return jsonable(asdict(self))


# ---------------------------------------------------------------- LLM
class LLMReasoner:
    """Claude tool-use loop. Only consulted when the situation changes or every N ticks."""

    name = "llm"

    def __init__(self, api_key: str, model: Optional[str] = None, max_turns: int = 10) -> None:
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        self.max_turns = max_turns

    def run(self, toolbox: ToolBox, brief: str) -> str:
        """Run the agentic loop; returns the model's final explanation text."""
        messages: List[Dict[str, Any]] = [{"role": "user", "content": brief}]
        text = ""
        for _ in range(self.max_turns):
            resp = self.client.messages.create(model=self.model, max_tokens=1024, system=SYSTEM_PROMPT,
                                               tools=toolbox.schemas, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            text = " ".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip() or text
            if resp.stop_reason != "tool_use":
                break
            results = [{"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(toolbox.call(b.name, b.input))}
                       for b in resp.content if getattr(b, "type", "") == "tool_use"]
            messages.append({"role": "user", "content": results})
        return text


# ---------------------------------------------------------------- agent
class OrchestratorAgent:
    """Runs one decision per tick against a :class:`Simulator`."""

    def __init__(self, sim: Simulator, config: Config, use_llm: Optional[bool] = None, auto_approve: bool = True,
                 log_path: Optional[Path] = DEFAULT_LOG, llm_every_n: Optional[int] = None,
                 llm: Optional[LLMReasoner] = None) -> None:
        self.sim, self.cfg = sim, config
        self.forecaster = Forecaster(sim)
        self.mock = MockReasoner(config)
        self.auto_approve = auto_approve
        self.log_path = Path(log_path) if log_path else None
        a = config.agent
        self.horizon = int(a.get("horizon", 16))
        self.llm_every_n = int(llm_every_n or a.get("llm_every_n_ticks", 8))
        self.export_threshold = float(a.get("export_approval_mw", 50))
        self.permission_ticks = int(a.get("permission_ticks", 8))
        self.max_relax = int(a.get("max_relax_attempts", 2))
        self.llm: Optional[LLMReasoner] = llm
        self.llm_error = ""
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if self.llm is None and use_llm is not False and key:
            try:
                self.llm = LLMReasoner(key)
            except Exception as exc:  # SDK missing/misconfigured → offline mode
                self.llm_error = f"LLM unavailable: {exc}"
        self.toolbox = ToolBox(self)
        self.criteria, self.criteria_reason = self.mock.choose_criteria("normal")
        self.situation = "normal"
        self.last_llm_tick = -10 ** 6
        self.last_llm_text = ""
        self.last_alert_kinds: set = set()
        self.plans: Dict[str, Plan] = {}
        self.approvals: List[ApprovalRequest] = []
        self.permissions: Dict[str, tuple] = {}        # kind -> (granted, until_tick)
        self.decisions: List[Decision] = []
        self._inspected: set = set()
        self._notes: List[str] = []
        self._outcome: Dict[str, Any] = {}
        self.current_state: GridState = sim.state()
        self.current_forecast: Forecast = self.forecaster.forecast(self.horizon)
        self.current_alerts = sim.alerts(self.horizon)

    @property
    def mode(self) -> str:
        return "llm" if self.llm else "mock"

    # ------------------------------------------------------------ main loop
    def step(self) -> Decision:
        """Run one full perceive → … → explain cycle and advance the simulator one tick."""
        sim = self.sim
        self._notes, self._outcome = [], {}
        self.toolbox.reset()
        # Perceive
        st = self.current_state = sim.state()
        fc = self.current_forecast = self.forecaster.forecast(self.horizon)
        alerts = self.current_alerts = sim.alerts(self.horizon)
        # Assess (cheap detector — also decides whether the LLM is worth a call)
        ref_price = float(np.median(sim.profile.mcp)) + self.cfg.market.buy_adder
        expected = float(sim.profile.demand[st.tick % sim.T].sum())
        assessment = self.mock.assess(st, fc, alerts, ref_price, expected)
        kinds = {a.kind for a in alerts}
        changed = assessment.situation != self.situation or not kinds <= self.last_alert_kinds
        self.last_alert_kinds = kinds
        reasoner, llm_error, explanation = "mock", "", ""

        if self.llm and (changed or st.tick - self.last_llm_tick >= self.llm_every_n):
            try:
                explanation = self.llm.run(self.toolbox, tick_brief(st.tick, st.hour, assessment.situation,
                                                                      assessment.signals))
                self.last_llm_tick, self.last_llm_text = st.tick, explanation
                reasoner = "llm"
                if self.toolbox.criteria is not None:
                    self.criteria = self.toolbox.criteria
                    self.situation = self.toolbox.situation
                    self.criteria_reason = self.toolbox.reasoning or "chosen by LLM"
                else:
                    self._set_mock_criteria(assessment.situation)
                    llm_error = "LLM did not call run_optimizer; used rule-based criteria"
            except Exception as exc:  # API outage, rate limit, bad response → offline reasoner
                llm_error = f"{type(exc).__name__}: {exc}"[:300]
                reasoner = "mock (llm error)"
                self._set_mock_criteria(assessment.situation)
        elif self.llm and assessment.situation == self.situation:
            reasoner = "llm-cached"
        else:
            self._set_mock_criteria(assessment.situation)
            reasoner = "mock" if not self.llm else "mock (situation changed)"

        self._operations()
        # Optimize + validate (unless the LLM already executed a plan)
        if not self.toolbox.executed:
            plan = self.get_plan(self.toolbox.plan_id) if self.toolbox.plan_id else None
            plan = plan or self.make_plan(self.criteria)
            self.execute(plan)
        plan = self.plans[self._outcome["plan_id"]]
        # Explain
        if reasoner != "llm" or not explanation:
            explanation = self.mock.explain(self.situation, assessment.signals, self.criteria, self._outcome)
            if reasoner == "llm-cached" and self.last_llm_text:
                explanation = f"(Criteria from LLM at tick {self.last_llm_tick}.) " + explanation
        decision = Decision(
            tick=st.tick, hour=st.hour, situation=self.situation, signals=assessment.signals, reasoner=reasoner,
            criteria=self.criteria.to_dict(), criteria_reason=self.criteria_reason, plan_source=plan.source,
            attempts=plan.attempts, validation=self._outcome["validation"], action=self._outcome["action"],
            action_summary=self._outcome["first_tick"], approvals=self._outcome.get("approvals", []),
            notes=self._notes, explanation=explanation, kpi=self._outcome["kpi"],
            tool_calls=list(self.toolbox.calls), llm_error=llm_error or self.llm_error,
        )
        self.decisions.append(decision)
        self._log(decision)
        self.plans = {k: v for k, v in self.plans.items() if v.tick >= st.tick}   # drop stale plans
        return decision

    def run_day(self) -> List[Decision]:
        """Step until the simulated day ends."""
        while not self.sim.done:
            self.step()
        return self.decisions

    def _set_mock_criteria(self, situation: str) -> None:
        self.situation = situation
        self.criteria, self.criteria_reason = self.mock.choose_criteria(situation)

    # ------------------------------------------------------------ optimize / validate
    def make_plan(self, criteria: Criteria) -> Plan:
        """Optimize → validate; relax up to ``max_relax`` times; else safe fallback. Never raises."""
        st, fc = self.current_state, self.current_forecast
        base = self._apply_permissions(criteria.normalized())
        attempts: List[Dict[str, Any]] = []
        crit = base
        for attempt in range(self.max_relax + 1):
            res = optimize(st, fc, self.cfg, crit)
            rec = {"attempt": attempt, "status": res.status, "reserve_pct": round(crit.reserve_pct, 3),
                   "allow_shedding": crit.allow_shedding, "allow_unserved": crit.allow_unserved,
                   "solve_s": round(res.solve_time_s, 3)}
            if res.status == "optimal":
                vr = validate(res.action, st, self.cfg)
                rec["validation"] = vr.violations or "passed"
                attempts.append(rec)
                if vr.passed:
                    src = "optimizer" if attempt == 0 else f"optimizer (relaxed x{attempt})"
                    return self._store(Plan("", st.tick, crit, res.action, src, vr, attempts, res.objective))
            else:
                rec["message"] = res.message
                attempts.append(rec)
            crit = self._apply_permissions(base.relaxed(attempt + 1))
        action = safe_action(st, self.cfg, reserve_pct=min(base.reserve_pct, 0.2))
        attempts.append({"attempt": "fallback", "status": "rule-based"})
        return self._store(Plan("", st.tick, base, action, "fallback", validate(action, st, self.cfg), attempts))

    def _store(self, plan: Plan) -> Plan:
        plan.id = f"P{plan.tick}-{len(self.plans) + 1}"
        self.plans[plan.id] = plan
        return plan

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        return self.plans.get(plan_id)

    def _apply_permissions(self, c: Criteria) -> Criteria:
        """Respect recent operator rejections so the agent doesn't keep re-asking."""
        granted, until = self.permissions.get("large_export", (True, -1))
        if not granted and self.sim.tick <= until:
            c = replace(c, max_export_mw=self.export_threshold)
        return c

    # ------------------------------------------------------------ act
    def execute(self, plan: Plan) -> Dict[str, Any]:
        """Approval gate → final validation → apply to the digital twin."""
        st = self.current_state
        action, source, approvals = plan.action.copy(), plan.source, []
        for kind, reason in self._high_impact(action):
            if self._permitted(kind):
                continue
            req = self.request_approval(kind, reason, describe_action(action))
            approvals.append(req.id)
            if req.status == "auto-approved":
                continue
            action, source = self._without(kind, action, plan)
        vr = validate(action, st, self.cfg)
        if not vr.passed and source != "fallback":
            self._notes.append(f"final validation failed ({'; '.join(vr.violations)}) → fallback")
            action, source = safe_action(st, self.cfg), "fallback"
            vr = validate(action, st, self.cfg)
        if not vr.passed:
            self._notes.append("EMERGENCY: supply cannot cover critical load; fallback minimises unserved energy")
        kpi = self.sim.step(action)
        first = describe_action(action)
        self._outcome = {"plan_id": plan.id, "source": source, "validation": vr.to_dict(), "action": action.to_dict(),
                         "first_tick": first, "approvals": approvals, "kpi": kpi.to_dict(),
                         "fallback": source == "fallback", "summary": summarize(first)}
        if source != plan.source:
            plan.source = source
        return {"executed": True, "source": source, "validation": vr.to_dict(), "first_tick": first,
                "approvals": approvals, "violations": kpi.violations}

    def _high_impact(self, a: Action) -> List[tuple]:
        out = []
        if a.shed_mw.sum() > 1e-3:
            out.append(("load_shedding", f"Plan sheds {a.shed_mw.sum():.1f} MW of flexible load"))
        if a.grid_export_mw > self.export_threshold + 1e-6:
            out.append(("large_export", f"Plan exports {a.grid_export_mw:.1f} MW (> {self.export_threshold:.0f} MW)"))
        return out

    def _without(self, kind: str, action: Action, plan: Plan) -> tuple:
        """Re-plan without an unapproved high-impact action; emergency shedding if physically required."""
        st = self.current_state
        if kind == "large_export":
            c = replace(plan.criteria, max_export_mw=self.export_threshold)
        else:
            c = replace(plan.criteria, allow_shedding=False, allow_unserved=False)
        res = optimize(st, self.current_forecast, self.cfg, c)
        if res.status == "optimal" and validate(res.action, st, self.cfg).passed:
            self._notes.append(f"{kind} held pending approval; re-optimised without it")
            return res.action, plan.source + " (approval-constrained)"
        if kind == "large_export":
            capped = cap_export(action, st, self.export_threshold)
            self._notes.append("large export held pending approval; surplus curtailed instead")
            return capped, plan.source + " (export capped)"
        self._notes.append("EMERGENCY load shedding executed while approval pending: supply is physically short")
        return action, plan.source

    # ------------------------------------------------------------ approvals
    def request_approval(self, kind: str, reason: str, details: Dict[str, Any]) -> ApprovalRequest:
        """Queue (or auto-approve) a human approval request; duplicates of a pending kind are merged."""
        for r in self.approvals:
            if r.kind == kind and r.status == "pending" and kind != "delay_maintenance":
                return r
        req = ApprovalRequest(f"A{len(self.approvals) + 1}", self.sim.tick, kind, reason, jsonable(details))
        self.approvals.append(req)
        if self.auto_approve:
            self._resolve(req, True, auto=True)
        return req

    def resolve_approval(self, req_id: str, approved: bool) -> ApprovalRequest:
        """Operator decision from the UI."""
        req = next(r for r in self.approvals if r.id == req_id)
        if req.status == "pending":
            self._resolve(req, approved)
        return req

    def _resolve(self, req: ApprovalRequest, approved: bool, auto: bool = False) -> None:
        req.status = ("auto-approved" if auto else "approved") if approved else "rejected"
        if req.kind in ("load_shedding", "large_export"):
            self.permissions[req.kind] = (approved, self.sim.tick + self.permission_ticks)
        if req.kind == "delay_maintenance":
            m = next((m for m in self.sim.maintenance if m.id == req.details.get("request_id")), None)
            if m and m.scheduled_start is None:
                start = int(req.details["start"]) if approved else m.requested_start
                self.sim.schedule_maintenance(m.id, max(start, self.sim.tick))

    def _permitted(self, kind: str) -> bool:
        granted, until = self.permissions.get(kind, (False, -1))
        return bool(granted) and self.sim.tick <= until

    def pending_approvals(self) -> List[ApprovalRequest]:
        return [r for r in self.approvals if r.status == "pending"]

    # ------------------------------------------------------------ operations
    def _operations(self) -> None:
        """Deterministic housekeeping: schedule maintenance requests, send crews to faults."""
        sim = self.sim
        pending_ids = {r.details.get("request_id") for r in self.pending_approvals()}
        for m in sim.maintenance:
            if m.scheduled_start is None and m.id not in pending_ids:
                self.schedule_maintenance(m.id, None)
        for b, ok in enumerate(self.current_state.battery_available):
            key = (self.cfg.batteries[b].id, next((e.start_tick for e in sim.events
                                                  if e.kind == "battery_unavailable" and e.active(sim.tick)), -1))
            if not ok and key not in self._inspected and sim.maintenance_active(key[0]) is False:
                self._inspected.add(key)
                self._notes.append(sim.dispatch_inspection(key[0]))

    def schedule_maintenance(self, request_id: str, start_tick: Optional[int]) -> Dict[str, Any]:
        """Choose (or accept) a start; delays beyond the requested start need approval."""
        sim = self.sim
        m = next((x for x in sim.maintenance if x.id == request_id), None)
        if m is None:
            return {"error": f"unknown maintenance request {request_id}"}
        if m.scheduled_start is not None:
            return {"status": "already scheduled", "start_tick": m.scheduled_start}
        lo, hi = max(sim.tick, m.requested_start - m.window // 2), m.requested_start + m.window
        start = int(np.clip(start_tick, lo, hi)) if start_tick is not None else self._best_window(m, lo, hi)
        if start > m.requested_start:
            req = self.request_approval("delay_maintenance",
                                        f"Move {m.id} on {m.asset_id} from tick {m.requested_start} to {start} "
                                        f"(lower lost output)", {"request_id": m.id, "start": start})
            if req.status == "pending":
                return {"status": "pending approval", "approval_id": req.id, "proposed_start": start}
            return {"status": "scheduled", "start_tick": m.scheduled_start, "approval_id": req.id}
        sim.schedule_maintenance(m.id, start)
        self._notes.append(f"maintenance {m.id} scheduled at tick {start}")
        return {"status": "scheduled", "start_tick": start}

    def _best_window(self, m: Any, lo: int, hi: int) -> int:
        """Start that minimises forecast lost energy × price for the asset."""
        fc = self.forecaster.forecast(hi - self.sim.tick + m.duration + 1)
        kind, idx = self.sim.asset_index(m.asset_id)
        price = fc.price["p50"]
        out = {"solar": fc.solar["p50"][:, idx] if kind == "solar" else None,
               "wind": fc.wind["p50"][:, idx] if kind == "wind" else None}.get(kind)
        out = out if out is not None else np.full(len(price), 10.0)    # battery: lost arbitrage ∝ price
        best, best_cost = m.requested_start, np.inf
        for s in range(lo, hi + 1):
            k0 = s - self.sim.tick
            if k0 < 0 or k0 + m.duration > len(price):
                continue
            cost = float((out[k0:k0 + m.duration] * price[k0:k0 + m.duration]).sum())
            if cost < best_cost - 1e-6 or (abs(cost - best_cost) <= 1e-6 and s == m.requested_start):
                best, best_cost = s, cost
        return best

    # ------------------------------------------------------------ logging
    def _log(self, d: Decision) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(d.to_dict()) + "\n")


# ---------------------------------------------------------------- helpers
def describe_action(a: Action) -> Dict[str, float]:
    """Aggregate, human-readable view of a dispatch."""
    return {"import_mw": round(a.grid_import_mw, 2), "export_mw": round(a.grid_export_mw, 2),
            "battery_charge_mw": round(float(a.charge_mw.sum()), 2),
            "battery_discharge_mw": round(float(a.discharge_mw.sum()), 2),
            "curtail_mw": round(float(a.curtail_solar_mw.sum() + a.curtail_wind_mw.sum()), 2),
            "dr_mw": round(float(a.dr_mw.sum()), 2), "shed_mw": round(float(a.shed_mw.sum()), 2),
            "unserved_critical_mw": round(float(a.unserved_critical_mw.sum()), 2)}


def summarize(d: Dict[str, float]) -> str:
    """One clause describing the executed dispatch."""
    parts = []
    for key, verb in (("battery_charge_mw", "charged batteries"), ("battery_discharge_mw", "discharged batteries"),
                      ("import_mw", "imported"), ("export_mw", "exported"), ("curtail_mw", "curtailed"),
                      ("dr_mw", "called demand response for"), ("shed_mw", "shed")):
        if d.get(key, 0) > 0.05:
            parts.append(f"{verb} {d[key]:.1f} MW")
    return ", ".join(parts) or "held a balanced position with no grid trades"


def cap_export(a: Action, st: GridState, limit: float) -> Action:
    """Reduce export to ``limit`` by curtailing renewables pro-rata (keeps power balance)."""
    out = a.copy()
    excess = max(0.0, out.grid_export_mw - limit)
    if excess <= 0:
        return out
    out.grid_export_mw = limit
    room_s, room_w = st.solar_mw - out.curtail_solar_mw, st.wind_mw - out.curtail_wind_mw
    total = float(room_s.sum() + room_w.sum())
    if total > 0:
        out.curtail_solar_mw = out.curtail_solar_mw + room_s * min(1.0, excess / total)
        out.curtail_wind_mw = out.curtail_wind_mw + room_w * min(1.0, excess / total)
    return out
