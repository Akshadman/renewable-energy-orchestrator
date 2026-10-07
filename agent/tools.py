"""Tool definitions (Anthropic tool-use schemas) and their implementations.

The LLM only ever passes *criteria*, ids and short text. Dispatch numbers come from the
optimizer and every executed action passes the validator — the LLM cannot inject MW values.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List

from core.models import jsonable
from core.optimizer import SITUATIONS, Criteria

if TYPE_CHECKING:  # pragma: no cover
    from agent.agent import OrchestratorAgent

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {"name": "get_state", "description": "Current tick: generation, demand, prices, battery SoC, line limits, "
                                         "frequency and active events.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_forecast", "description": "P10/P50/P90 forecast totals for solar, wind, demand and price, "
                                            "sampled hourly over the horizon.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_alerts", "description": "Active and upcoming alerts: storms (weather codes / wind), faults, "
                                          "congestion, price spikes and pending maintenance requests.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "run_optimizer",
     "description": "Solve the rolling-horizon MILP with your criteria. Auto-relaxes up to 2x and falls back "
                    "to a safe controller if needed. Returns a plan_id.",
     "input_schema": {"type": "object", "properties": {
         "situation": {"type": "string", "enum": SITUATIONS},
         "w_cost": {"type": "number", "minimum": 0, "maximum": 1},
         "w_clean": {"type": "number", "minimum": 0, "maximum": 1},
         "w_reliability": {"type": "number", "minimum": 0, "maximum": 1},
         "reserve_pct": {"type": "number", "minimum": 0, "maximum": 0.9},
         "uncertainty_mode": {"type": "string", "enum": ["p50", "p10"]},
         "reasoning": {"type": "string", "description": "One sentence: why these criteria."}},
         "required": ["situation", "w_cost", "w_clean", "w_reliability", "reserve_pct", "uncertainty_mode"]}},
    {"name": "validate_actions", "description": "Run the physics validator on a plan.",
     "input_schema": {"type": "object", "properties": {"plan_id": {"type": "string"}}, "required": ["plan_id"]}},
    {"name": "execute_actions", "description": "Execute a validated plan for this tick (human approval is "
                                               "requested automatically for high-impact actions).",
     "input_schema": {"type": "object", "properties": {"plan_id": {"type": "string"}}, "required": ["plan_id"]}},
    {"name": "request_human_approval", "description": "Escalate a decision to the human operator.",
     "input_schema": {"type": "object", "properties": {
         "kind": {"type": "string"}, "reason": {"type": "string"}}, "required": ["kind", "reason"]}},
    {"name": "schedule_maintenance",
     "description": "Schedule a pending maintenance request. Omit start_tick to pick the lowest-impact start "
                    "inside its window. Delaying past the requested start needs human approval.",
     "input_schema": {"type": "object", "properties": {
         "request_id": {"type": "string"}, "start_tick": {"type": "integer"}}, "required": ["request_id"]}},
    {"name": "dispatch_inspection", "description": "Send a field crew to a faulted asset (e.g. B1, W2).",
     "input_schema": {"type": "object", "properties": {
         "asset_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["asset_id"]}},
]


class ToolBox:
    """Binds tool names to agent operations and tracks what the LLM did this tick."""

    def __init__(self, agent: "OrchestratorAgent") -> None:
        self.agent = agent
        self.reset()

    def reset(self) -> None:
        """Clear per-tick bookkeeping."""
        self.calls: List[Dict[str, Any]] = []
        self.situation: str = ""
        self.criteria: Criteria = None  # type: ignore[assignment]
        self.reasoning: str = ""
        self.plan_id: str = ""
        self.executed = False

    @property
    def schemas(self) -> List[Dict[str, Any]]:
        return TOOL_SCHEMAS

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a tool call; errors are returned to the model, never raised."""
        fn: Callable[..., Dict[str, Any]] = getattr(self, f"_t_{name}", None)  # type: ignore[assignment]
        try:
            out = fn(**(args or {})) if fn else {"error": f"unknown tool {name}"}
        except Exception as exc:  # bad arguments etc.
            out = {"error": f"{type(exc).__name__}: {exc}"}
        self.calls.append({"tool": name, "args": jsonable(args), "ok": "error" not in out})
        return jsonable(out)

    # ------------------------------------------------------------ perception
    def _t_get_state(self) -> Dict[str, Any]:
        st, cfg = self.agent.sim.state(), self.agent.cfg
        return {
            "tick": st.tick, "hour": round(st.hour, 2), "solar_mw": round(float(st.solar_mw.sum()), 1),
            "wind_mw": round(float(st.wind_mw.sum()), 1), "demand_mw": round(st.total_demand_mw, 1),
            "critical_mw": round(float(st.critical_mw.sum()), 1), "price_buy": round(st.price_buy),
            "price_sell": round(st.price_sell),
            "batteries": [{"id": b.id, "soc_pct": round(float(st.soc_mwh[i] / b.energy_mwh * 100), 1),
                           "available": bool(st.battery_available[i]), "power_mw": b.power_mw}
                          for i, b in enumerate(cfg.batteries)],
            "import_limit_mw": round(st.import_limit_mw, 1), "export_limit_mw": round(st.export_limit_mw, 1),
            "frequency_hz": round(st.frequency_hz, 3), "active_events": st.active_events,
        }

    def _t_get_forecast(self) -> Dict[str, Any]:
        return self.agent.current_forecast.summary()

    def _t_get_alerts(self) -> Dict[str, Any]:
        return {"alerts": [{"kind": a.kind, "severity": a.severity, "message": a.message, "eta_min": a.eta_ticks * 15}
                           for a in self.agent.current_alerts]}

    # ------------------------------------------------------------ planning & action
    def _t_run_optimizer(self, situation: str, w_cost: float, w_clean: float, w_reliability: float,
                         reserve_pct: float, uncertainty_mode: str, reasoning: str = "") -> Dict[str, Any]:
        crit = Criteria(float(w_cost), float(w_clean), float(w_reliability), float(reserve_pct),
                        str(uncertainty_mode)).normalized()
        self.situation = situation if situation in SITUATIONS else "normal"
        self.criteria, self.reasoning = crit, reasoning
        plan = self.agent.make_plan(crit)
        self.plan_id = plan.id
        return plan.summary()

    def _t_validate_actions(self, plan_id: str) -> Dict[str, Any]:
        plan = self.agent.get_plan(plan_id)
        return plan.validation.to_dict() if plan else {"error": f"unknown plan {plan_id}"}

    def _t_execute_actions(self, plan_id: str) -> Dict[str, Any]:
        if self.executed:
            return {"error": "already executed this tick"}
        plan = self.agent.get_plan(plan_id)
        if not plan:
            return {"error": f"unknown plan {plan_id}"}
        outcome = self.agent.execute(plan)
        self.executed = True
        return outcome

    def _t_request_human_approval(self, kind: str, reason: str) -> Dict[str, Any]:
        req = self.agent.request_approval(kind, reason, {})
        return {"request_id": req.id, "status": req.status}

    def _t_schedule_maintenance(self, request_id: str, start_tick: int = None) -> Dict[str, Any]:
        return self.agent.schedule_maintenance(request_id, start_tick)

    def _t_dispatch_inspection(self, asset_id: str, reason: str = "") -> Dict[str, Any]:
        return {"result": self.agent.sim.dispatch_inspection(asset_id)}
