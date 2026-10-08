"""LLM tool definition and the briefing it receives.

The LLM can only choose an option id among stress-tested, validated plans and write text — it can
never inject a megawatt value, and choices outside the eligible set are rejected by code.
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

TOOL_SCHEMAS: List[Dict[str, Any]] = [SUBMIT_DECISION]


def build_briefing(tick_label: str, signals: Dict[str, Any], headline: List[str], options: List[Dict[str, Any]],
                   recommended: str, eligible: List[str], targets: Dict[str, Any]) -> str:
    """Compact JSON briefing for the LLM."""
    return json.dumps({
        "time": tick_label, "signals": signals, "signals_summary": headline,
        "industry_targets": targets,
        "options": [{k: o[k] for k in ("id", "name", "reserve_pct", "first_tick", "cost_avg_rs", "cost_p95_rs",
                                        "p_shortfall", "co2_t", "curtailed_mwh", "battery_wear_mwh", "status")}
                    for o in options],
        "eligible": eligible, "recommended_by_ladder": recommended,
    }, default=str)
