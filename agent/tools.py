"""LLM tools and the briefing it receives.

The LLM can (1) propose its own plan *settings*, which the agent optimises and stress-tests like any
other, and (2) choose an option id among eligible plans and explain. It can never inject a megawatt
value, and choices outside the eligible set are rejected by code.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

SUBMIT_DECISION = {
    "name": "submit_decision",
    "description": "Commit this tick's plan. option_id must be one of the ELIGIBLE options.",
    "input_schema": {
        "type": "object",
        "properties": {
            "option_id": {"type": "string", "description": "Id of the chosen eligible option, e.g. 'O3'."},
            "diagnosis": {"type": "string", "description": "One sentence: what is happening, all signals considered."},
            "explanation": {"type": "string",
                            "description": "2–3 plain-English sentences for the operator: what you chose and why, "
                                           "including the main trade-off you accepted."},
        },
        "required": ["option_id", "diagnosis", "explanation"],
    },
}

PROPOSE_PLAN = {
    "name": "propose_plan",
    "description": "Test your own plan settings. The agent optimises it, stress-tests it in the same ~100 futures, "
                   "checks it against the ladder and returns its numbers and option id.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reserve_pct": {"type": "number", "description": "battery reserve floor, 0–0.9 of capacity"},
            "w_cost": {"type": "number", "description": "weight on cost, 0–1"},
            "w_clean": {"type": "number", "description": "weight on clean energy, 0–1"},
            "w_reliability": {"type": "number", "description": "weight on reliability, 0–1"},
            "caution": {"type": "number", "description": "0 = plan on expected forecast, 1 = pessimistic (P10)"},
            "reason": {"type": "string", "description": "why this setting could beat the recommendation"},
        },
        "required": ["reserve_pct", "w_cost", "w_clean", "w_reliability", "caution", "reason"],
    },
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [PROPOSE_PLAN, SUBMIT_DECISION]


def build_briefing(tick_label: str, signals: Dict[str, Any], headline: List[str], options: List[Dict[str, Any]],
                   recommended: str, eligible: List[str], targets: Dict[str, Any], evidence: List[str] = ()) -> str:
    """Compact JSON briefing for the LLM."""
    return json.dumps({
        "time": tick_label, "signals": signals, "signals_summary": headline,
        "multimodal_evidence": list(evidence),
        "industry_targets": targets,
        "options": [{k: o[k] for k in ("id", "name", "reserve_pct", "first_tick", "cost_avg_rs", "cost_p95_rs",
                                        "p_shortfall", "co2_t", "curtailed_mwh", "battery_wear_mwh", "status")}
                    for o in options],
        "eligible": eligible, "recommended_by_ladder": recommended,
    }, default=str)
