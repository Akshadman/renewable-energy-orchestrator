"""Autonomous orchestrator.

Every 15 minutes (and immediately after any shock), with a day-ahead plan as the backbone:

  PERCEIVE    state, learned-corrected P10/P50/P90 forecasts, prices, alerts (with probabilities)
  DIAGNOSE    continuous risk signals, all measured together (no situation labels)
  OPTIONS     the MILP optimizer produces a spread of candidate plans (cheapest … safest)
  STRESS-TEST each plan is replayed across ~100 sampled futures → avg ₹, worst-case ₹, P(shortfall)
  CHOOSE      priority ladder with thresholds: Safety > Reliability (≤ tolerance) > Cost vs Clean
              (weighted by industry targets) > Battery life; optional LLM may pick among eligible plans
  VERIFY      validator; if nothing passes → repair (relax) → retry → safe fallback
  ACT+EXPLAIN execute autonomously, log, 2–3 sentence reason
  LEARN       compare forecasts with outcomes, correct bias & band width, persist at end of day
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent.narrator import diagnosis_text, explain
from agent.prompts import SYSTEM_PROMPT, tick_brief
from agent.signals import Signals, diagnose
from agent.stress_test import StressResult, sample_futures, stress_test
from agent.tools import SUBMIT_DECISION, build_briefing
from core.config import ROOT, resolve_path
from core.fallback import safe_action
from core.forecaster import Forecaster
from core.learning import ForecastLearner
from core.models import Action, Config, Forecast, GridState, jsonable
from core.optimizer import Criteria, OptimizerResult, optimize
from core.simulator import Simulator, clock
from core.validator import ValidationResult, validate

DEFAULT_LOG = ROOT / "logs" / "decisions.jsonl"
DEFAULT_MODEL = "claude-sonnet-5"

# Candidate plans: a fine ladder of protection levels (battery reserve 10% → 85%) with weights that
# shift smoothly from cost-lean to reliability-lean, plus one green-lean plan. The stress test scores
# every rung against the *probabilities* of the futures, so the chosen protection rises gradually as
# risk rises instead of jumping between fixed cases.
def _rung(reserve: float) -> Criteria:
    x = (reserve - 0.10) / 0.75                      # 0 at the cheapest rung, 1 at the safest
    return Criteria(0.65 - 0.50 * x, 0.20 - 0.10 * x, 0.15 + 0.60 * x, reserve, "p10" if reserve >= 0.5 else "p50")


OPTION_GRID: List[Tuple[str, Criteria]] = [
    ("Max savings", _rung(0.10)),
    ("Lean", _rung(0.20)),
    ("Balanced", _rung(0.30)),
    ("Green-lean", Criteria(0.30, 0.55, 0.15, 0.30, "p50")),
    ("Steady", _rung(0.40)),
    ("Cautious", _rung(0.50)),
    ("Protective", _rung(0.62)),
    ("Strong reserve", _rung(0.74)),
    ("Fortress", _rung(0.85)),
]


# ---------------------------------------------------------------- records
@dataclass
class Option:
    id: str
    name: str
    criteria: Criteria
    result: OptimizerResult
    validation: Optional[ValidationResult] = None
    stress: Optional[StressResult] = None
    status: str = "invalid"           # eligible | too risky | invalid
    reason: str = ""
    score: float = float("inf")

    @property
    def action(self) -> Optional[Action]:
        return self.result.action

    def row(self) -> Dict[str, Any]:
        s = self.stress
        return jsonable({
            "id": self.id, "name": self.name, "reserve_pct": self.criteria.reserve_pct,
            "forecast": self.criteria.uncertainty_mode, "status": self.status, "reason": self.reason,
            "cost_avg_rs": s.cost_avg if s else None, "cost_p95_rs": s.cost_p95 if s else None,
            "p_shortfall": s.p_shortfall if s else None, "co2_t": s.co2_t if s else None,
            "curtailed_mwh": s.curtailed_mwh if s else None, "battery_wear_mwh": s.battery_throughput_mwh if s else None,
            "score": None if not np.isfinite(self.score) else self.score,
            "first_tick": describe_action(self.action) if self.action is not None else None,
        })


@dataclass
class Decision:
    """Everything that happened in one tick — logged as one JSONL line."""

    tick: int
    hour: float
    time: str
    trigger: str                       # scheduled | shock
    signals: Dict[str, Any]
    headline: List[List[str]]
    diagnosis: str
    options: List[Dict[str, Any]]
    chosen: str
    chosen_name: str
    criteria: Dict[str, Any]
    plan_source: str
    reasoner: str
    validation: Dict[str, Any]
    action: Dict[str, Any]
    action_summary: Dict[str, Any]
    notes: List[str]
    explanation: str
    kpi: Dict[str, Any]
    llm_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return jsonable(asdict(self))


# ---------------------------------------------------------------- LLM
class LLMReasoner:
    """Claude picks among eligible, stress-tested plans and explains (one forced tool call)."""

    name = "llm"

    def __init__(self, api_key: str, model: Optional[str] = None) -> None:
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL

    def decide(self, briefing: str) -> Dict[str, str]:
        """Return {'option_id', 'diagnosis', 'explanation'}."""
        resp = self.client.messages.create(
            model=self.model, max_tokens=700, system=SYSTEM_PROMPT, tools=[SUBMIT_DECISION],
            tool_choice={"type": "tool", "name": "submit_decision"},
            messages=[{"role": "user", "content": tick_brief(briefing)}])
        for block in resp.content:
            if getattr(block, "type", "") == "tool_use" and block.name == "submit_decision":
                return dict(block.input)
        raise ValueError("model did not call submit_decision")


# ---------------------------------------------------------------- agent
class OrchestratorAgent:
    """Runs one fully autonomous decision per tick against a :class:`Simulator`."""

    def __init__(self, sim: Simulator, config: Config, use_llm: Optional[bool] = None,
                 log_path: Optional[Path] = DEFAULT_LOG, llm_every_n: Optional[int] = None,
                 llm: Optional[LLMReasoner] = None, persist_learning: bool = False,
                 n_futures: Optional[int] = None, parallel: bool = True) -> None:
        self.sim, self.cfg = sim, config
        self.parallel = parallel
        a = config.agent
        self.horizon = int(a.get("horizon", 16))
        self.n_futures = int(n_futures or a.get("scenarios", 100))
        self.llm_every_n = int(llm_every_n or a.get("llm_every_n_ticks", 8))
        self.max_relax = int(a.get("max_relax_attempts", 2))
        path = resolve_path(config.data.get("learning_file", "data/learning.json")) if persist_learning else None
        self.learner = ForecastLearner(lead=4, rate=float(a.get("learning_rate", 0.15)), path=path)
        self.forecaster = Forecaster(sim, learner=self.learner)
        self.log_path = Path(log_path) if log_path else None
        self.llm: Optional[LLMReasoner] = llm
        self.llm_error = ""
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if self.llm is None and use_llm is not False and key:
            try:
                self.llm = LLMReasoner(key)
            except Exception as exc:  # SDK missing/misconfigured → offline mode
                self.llm_error = f"LLM unavailable: {exc}"
        self.ref_price = float(np.median(sim.profile.mcp)) + config.market.buy_adder
        self.decisions: List[Decision] = []
        self.current_forecast: Optional[Forecast] = None
        self.current_alerts: list = []
        self.current_signals = Signals()
        self._da_soc: Dict[int, float] = {}
        self._da_renew: Dict[int, float] = {}
        self._signature: Optional[tuple] = None
        self._last_llm_tick = -10 ** 6
        self._last_headline: List[str] = []
        self._inspected: set = set()
        self._notes: List[str] = []

    @property
    def mode(self) -> str:
        return "llm" if self.llm else "offline"

    # ------------------------------------------------------------ main loop
    def step(self) -> Decision:
        """One full autonomous cycle; advances the simulator one tick."""
        sim, cfg, H = self.sim, self.cfg, self.horizon
        self._notes = []
        self._operations()                                              # maintenance & crews (autonomous)
        # PERCEIVE (+ LEARN from the forecast that targeted this tick)
        st = sim.state()
        raw = self.forecaster.raw(H)
        fc = self.current_forecast = self.learner.correct(raw)
        self.learner.observe(raw, fc, st)
        alerts = self.current_alerts = sim.alerts(H)
        signature = (len(sim.events), len(sim.outages), tuple(sorted({a.kind for a in alerts})))
        trigger = "shock" if self._signature is not None and signature != self._signature else "scheduled"
        self._signature = signature
        if not self._da_soc or trigger == "shock":
            self._day_ahead(st)
        # DIAGNOSE
        sig = self.current_signals = self._diagnose(st, fc, alerts)
        headline = sig.headline()
        # OPTIONS + STRESS-TEST
        futures = sample_futures(st, fc, cfg, sig.storms, self.n_futures, seed=sim.seed * 7919 + st.tick)
        target = self._da_soc.get(st.tick + H - 1)
        crits = [self._tune(base, sig) for _, base in OPTION_GRID]
        solve = lambda c: optimize(st, fc, cfg, c, soc_target_mwh=target)  # noqa: E731
        if self.parallel:   # each CBC solve is a separate process → threads give real parallelism
            with ThreadPoolExecutor(max_workers=min(len(crits), os.cpu_count() or 1)) as pool:
                results = list(pool.map(solve, crits))
        else:
            results = [solve(c) for c in crits]
        options = []
        for k, ((name, _), crit, res) in enumerate(zip(OPTION_GRID, crits, results)):
            opt = Option(f"O{k + 1}", name, crit, res)
            if res.status == "optimal":
                opt.validation = validate(res.action, st, cfg)
                if opt.validation.passed:
                    opt.stress = stress_test(res.action, res.plan, st, fc, cfg, futures)
                else:
                    opt.reason = "fails physics check: " + "; ".join(opt.validation.violations)
            else:
                opt.reason = f"optimizer: {res.status}"
            options.append(opt)
        # CHOOSE (priority ladder)
        chosen, eligible = self._ladder(options, sig)
        reasoner, llm_error, diag, expl = "offline", "", diagnosis_text(headline), ""
        if chosen is not None and self.llm and eligible and self._should_consult(trigger, headline, st.tick):
            try:
                pick = self.llm.decide(build_briefing(clock(st.tick), sig.to_dict(), [t for t, _ in headline],
                                                      [o.row() for o in options], chosen.id,
                                                      [o.id for o in eligible], cfg.targets))
                self._last_llm_tick = st.tick
                reasoner = "llm"
                diag, expl = pick.get("diagnosis") or diag, pick.get("explanation", "")
                by_id = {o.id: o for o in eligible}
                if pick.get("option_id") in by_id:
                    if pick["option_id"] != chosen.id:
                        self._notes.append(f"LLM overrode ladder pick {chosen.id} → {pick['option_id']}")
                    chosen = by_id[pick["option_id"]]
                else:
                    llm_error = f"LLM picked non-eligible '{pick.get('option_id')}', kept {chosen.id}"
                    expl = ""
            except Exception as exc:  # API outage etc. → ladder pick stands
                llm_error = f"{type(exc).__name__}: {exc}"[:300]
        # VERIFY → repair → fallback
        action, source, crit = (chosen.action, "optimizer", chosen.criteria) if chosen else self._repair(st, fc, options)
        vr = validate(action, st, cfg)
        if not vr.passed and source != "fallback":
            self._notes.append("final check failed → safe fallback")
            action, source = safe_action(st, cfg), "fallback"
            vr = validate(action, st, cfg)
        if not vr.passed:
            self._notes.append("EMERGENCY: supply physically short of critical load; fallback minimises it")
        # ACT
        self._flag_high_impact(action, st)
        kpi = sim.step(action)
        summary = describe_action(action)
        rows = [o.row() for o in options]
        chosen_row = next((r for r in rows if chosen and r["id"] == chosen.id), None)
        if not expl:
            expl = explain(chosen_row or {}, rows, self._tolerance(), _summarize(summary), self.n_futures,
                           fallback=chosen_row is None, source=source)
        d = Decision(
            tick=st.tick, hour=st.hour, time=clock(st.tick), trigger=trigger, signals=sig.to_dict(),
            headline=[list(h) for h in headline], diagnosis=diag, options=rows,
            chosen=chosen.id if chosen else source, chosen_name=chosen.name if chosen else source,
            criteria=crit.to_dict(), plan_source=source, reasoner=reasoner, validation=vr.to_dict(),
            action=action.to_dict(), action_summary=summary, notes=self._notes, explanation=expl,
            kpi=kpi.to_dict(), llm_error=llm_error or self.llm_error)
        self.decisions.append(d)
        self._log(d)
        if sim.done:
            self.learner.end_of_day()
        return d

    def run_day(self) -> List[Decision]:
        """Step until the simulated day ends."""
        while not self.sim.done:
            self.step()
        return self.decisions

    # ------------------------------------------------------------ diagnose & choose
    def _diagnose(self, st: GridState, fc: Forecast, alerts: list) -> Signals:
        sim, cfg = self.sim, self.cfg
        t = st.tick % sim.T
        cap_s = np.array([f.capacity_mw for f in cfg.solar])
        cap_w = np.array([f.capacity_mw for f in cfg.wind])
        total = cap_s.sum() + cap_w.sum()
        offline = ((cap_s * (1 - sim.solar_avail[t])).sum() + (cap_w * (1 - sim.wind_avail[t])).sum()) / total
        k = sim.kpis
        return diagnose(st, fc, alerts, cfg, self.ref_price, float(sim.profile.demand[t].sum()),
                        self._da_renew.get(st.tick), float(offline), sum(x.co2_t for x in k),
                        sum(x.cost_rs for x in k), sum(x.served_mwh for x in k), len(k))

    def _tune(self, base: Criteria, sig: Signals) -> Criteria:
        """Industry targets shift every option's weights continuously (more CO₂ pressure → greener)."""
        clean = base.w_clean * float(np.clip(sig.carbon_pressure, 0.5, 3.0))
        return Criteria(base.w_cost, clean, base.w_reliability, base.reserve_pct, base.uncertainty_mode).normalized()

    def _tolerance(self) -> float:
        return float(self.cfg.targets.get("max_shortfall_prob", 0.02))

    def _ladder(self, options: List[Option], sig: Signals) -> Tuple[Optional[Option], List[Option]]:
        """1 Safety > 2 Reliability (≤ tolerance) > 3 Cost vs Clean > 4 Battery life."""
        m, tol = self.cfg.market, self._tolerance()
        ra = float(self.cfg.targets.get("risk_aversion", 0.3))
        clean_w = float(np.clip(sig.carbon_pressure, 0.5, 3.0))
        tested = [o for o in options if o.stress is not None]
        for o in tested:
            s = o.stress
            o.score = (s.cost_avg + ra * (s.cost_p95 - s.cost_avg)
                       + clean_w * (s.co2_t * m.carbon_price + s.curtailed_mwh * m.curtailment_penalty))
            o.status = "eligible" if s.p_shortfall <= tol + 1e-9 else "too risky"
            if o.status == "too risky":
                o.reason = f"shortfall risk {s.p_shortfall:.0%} > {tol:.0%}"
        if not tested:
            return None, []
        eligible = [o for o in tested if o.status == "eligible"]
        if not eligible:  # nothing meets the tolerance: minimise risk, then score
            best = min(tested, key=lambda o: (round(o.stress.p_shortfall, 3), o.stress.unserved_critical_mwh, o.score))
            self._notes.append(f"no plan meets the {tol:.0%} shortfall tolerance; chose the least risky")
            return best, [best]
        best_score = min(o.score for o in eligible)
        near = [o for o in eligible if o.score <= best_score + max(300.0, 0.003 * abs(best_score))]
        chosen = min(near, key=lambda o: (o.stress.battery_throughput_mwh, o.score))   # battery-life tiebreak
        for o in eligible:
            if o is not chosen:
                o.reason = f"₹{o.score - chosen.score:,.0f} worse on cost/clean score" if o.score > chosen.score \
                    else "similar score, more battery wear"
        chosen.reason = "chosen"
        return chosen, eligible

    def _should_consult(self, trigger: str, headline: list, tick: int) -> bool:
        texts = [t for t, _ in headline]
        changed = texts != self._last_headline
        self._last_headline = texts
        return trigger == "shock" or changed or tick - self._last_llm_tick >= self.llm_every_n

    def _repair(self, st: GridState, fc: Forecast, options: List[Option]) -> Tuple[Action, str, Criteria]:
        """No option passed: relax constraints (shedding, then unserved) and retry; else safe fallback."""
        base = OPTION_GRID[-2][1]
        for attempt in range(1, self.max_relax + 1):
            crit = base.relaxed(attempt)
            res = optimize(st, fc, self.cfg, crit)
            if res.status == "optimal" and validate(res.action, st, self.cfg).passed:
                self._notes.append(f"repaired by relaxing constraints (attempt {attempt})")
                return res.action, f"optimizer (relaxed x{attempt})", crit
        self._notes.append("repair failed → safe fallback controller")
        return safe_action(st, self.cfg, reserve_pct=0.2), "fallback", base

    # ------------------------------------------------------------ day-ahead backbone
    def _day_ahead(self, st: GridState) -> None:
        """24 h LP plan (to end of day) giving a target battery trajectory; re-based after shocks."""
        H = self.sim.T - st.tick
        if H <= self.horizon:
            return
        fc = self.forecaster.forecast(H)
        if not self._da_renew:
            self._da_renew = {st.tick + k: float(fc.solar["p50"][k].sum() + fc.wind["p50"][k].sum()) for k in range(H)}
        res = optimize(st, fc, self.cfg, Criteria(0.45, 0.30, 0.25, 0.25, "p50"), time_limit_s=20,
                       relax_binaries=True)
        if res.status == "optimal":
            self._da_soc = {st.tick + k: v for k, v in enumerate(res.plan["soc_mwh"])}

    # ------------------------------------------------------------ autonomous operations
    def _operations(self) -> None:
        """Schedule maintenance into the lowest-impact slot and send crews to breakdowns — no approval."""
        sim = self.sim
        for m in sim.maintenance:
            if m.scheduled_start is None and sim.tick >= m.requested_start - self.horizon:
                start = self._best_window(m)
                sim.schedule_maintenance(m.id, start)
                moved = "" if start == m.requested_start else f" (moved from {clock(m.requested_start)})"
                self._notes.append(f"Scheduled maintenance {m.id} on {m.asset_id} at {clock(start)}{moved}")
        for o in sim.outages:
            key = (o["asset"], o["start"])
            if o["cause"] == "breakdown" and o["start"] <= sim.tick < o["end"] and key not in self._inspected:
                self._inspected.add(key)
                self._notes.append(sim.dispatch_inspection(o["asset"]))

    def _best_window(self, m: Any) -> int:
        """Start inside the allowed window that minimises forecast lost energy × price."""
        sim = self.sim
        lo, hi = max(sim.tick, m.requested_start - m.window), min(sim.T - m.duration, m.requested_start + m.window)
        if hi <= lo:
            return max(sim.tick, m.requested_start)
        fc = self.forecaster.forecast(hi - sim.tick + m.duration + 1)
        kind, idx = sim.asset_index(m.asset_id)
        price = fc.price["p50"]
        out = {"solar": fc.solar["p50"], "wind": fc.wind["p50"]}.get(kind)
        out = out[:, idx] if out is not None else np.full(len(price), 10.0)
        best, best_cost = max(lo, m.requested_start), np.inf
        for s in range(lo, hi + 1):
            k0 = s - sim.tick
            cost = float((out[k0:k0 + m.duration] * price[k0:k0 + m.duration]).sum())
            if cost < best_cost - 1e-6 or (abs(cost - best_cost) <= 1e-6 and s == m.requested_start):
                best, best_cost = s, cost
        return best

    def _flag_high_impact(self, a: Action, st: GridState) -> None:
        if a.shed_mw.sum() > 1e-3:
            self._notes.append(f"Autonomously shed {a.shed_mw.sum():.1f} MW of flexible load")
        if a.grid_export_mw > 0.8 * st.export_limit_mw > 0:
            self._notes.append(f"Large export {a.grid_export_mw:.1f} MW (near line limit)")

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


def _summarize(d: Dict[str, float]) -> str:
    parts = []
    for key, verb in (("battery_charge_mw", "charged batteries"), ("battery_discharge_mw", "discharged batteries"),
                      ("import_mw", "imported"), ("export_mw", "exported"), ("curtail_mw", "curtailed"),
                      ("dr_mw", "paid demand response for"), ("shed_mw", "shed")):
        if d.get(key, 0) > 0.05:
            parts.append(f"{verb} {d[key]:.1f} MW")
    return ", ".join(parts) or "held a balanced position with no grid trades"
