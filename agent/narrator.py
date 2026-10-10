"""Offline narrator: plain-English diagnosis and explanation without an LLM (always available)."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def diagnosis_text(headline: List[Tuple[str, str]]) -> str:
    """One sentence that lists every notable signal (not a single category)."""
    items = [t for t, lvl in headline if lvl != "ok"]
    if not items:
        return "All signals are within normal ranges."
    return "Watching: " + "; ".join(items) + "."


def explain(chosen: Dict[str, Any], options: List[Dict[str, Any]], tolerance: float, summary: str,
            n_futures: int, fallback: bool = False, source: str = "") -> str:
    """2–3 sentences: what was chosen, the trade-off accepted, what was done this tick."""
    if fallback and source.startswith("optimizer (relaxed"):
        return ("Available supply (renewables + batteries + grid line) cannot cover all demand over the next "
                "4 hours, so no fully-served plan exists. I re-planned allowing controlled reduction of flexible "
                f"load while protecting critical load first. This tick I {summary}.")
    if fallback:
        return ("Supply is physically short of demand right now and no optimised plan passed the safety checks, "
                f"so the safe fallback controller acted, protecting critical load first. This tick I {summary}.")
    recipes = len({o["name"] for o in options})
    first = (f"I tested {len(options)} combinations from {recipes} recipe bands across {n_futures} possible "
             f"futures and chose a '{chosen['name']}' plan (keep ≥{chosen['reserve_pct']:.0%} battery reserve): expected cost "
             f"₹{chosen['cost_avg_rs']:,.0f}, worst case ₹{chosen['cost_p95_rs']:,.0f}, "
             f"shortfall risk {chosen['p_shortfall']:.0%}.")
    cheaper = [o for o in options if o.get("cost_avg_rs") is not None
               and o["cost_avg_rs"] < chosen["cost_avg_rs"] - 1 and o["status"] != "invalid"]
    if cheaper:
        best = min(cheaper, key=lambda o: o["cost_avg_rs"])
        if best["p_shortfall"] > tolerance:
            why = (f"'{best['name']}' would save ₹{chosen['cost_avg_rs'] - best['cost_avg_rs']:,.0f} on average but "
                   f"risks a critical shortfall in {best['p_shortfall']:.0%} of futures (limit {tolerance:.0%}).")
        else:
            why = (f"'{best['name']}' is ₹{chosen['cost_avg_rs'] - best['cost_avg_rs']:,.0f} cheaper on average but "
                   f"scores worse once worst-case cost, CO₂ or battery wear are counted.")
    else:
        why = "It is also the cheapest safe option."
    return f"{first} {why} This tick I {summary}."
