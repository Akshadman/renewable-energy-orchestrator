"""Documents & notices (PDF / text) → structured operational findings.

Understands four notice types written in free text:
  maintenance   — work orders / planned outages (optionally movable by ±N hours)
  breakdown     — trips, faults, forced outages (optionally partial: "50% of capacity")
  line_limit    — grid-operator (SLDC) congestion / transmission derates
  demand_change — extra shifts, production batches, load increases

Two extractors share one schema and one validator:
  * a deterministic rule-based parser (always available, used in all tests & offline), and
  * an optional LLM extractor (Claude / Groq / Gemini … when a key is set), validated identically.
Confidence < APPLY_THRESHOLD → quarantined (shown to operators, never acted on).
"""
from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional, Tuple

from core.multimodal.evidence import APPLY_THRESHOLD, Evidence

KEYWORDS = {
    "maintenance": ["maintenance", "work order", "servicing", "overhaul", "planned outage", "preventive", "inspection",
                    "cleaning", "shutdown for"],
    "breakdown": ["breakdown", "tripped", "trip ", "failure", "failed", "fault", "forced outage", "out of service",
                  "damaged"],
    "line_limit": ["sldc", "congestion", "transmission", "evacuation", "interconnect", "feeder", "grid operator",
                   "line capacity", "import limit", "export limit"],
    "demand_change": ["extra shift", "additional shift", "production", "additional load", "load increase", "batch",
                      "demand increase", "overtime"],
}
HEDGES = ["tentative", "to be confirmed", "tbc", "may be", "possibly", "might", "subject to", "if required",
          "provisional", "unconfirmed"]
TIME = r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|hrs|hours|h\b)?"


# ---------------------------------------------------------------- reading
def read_document(name: str, data: bytes) -> str:
    """Text from a PDF / TXT / MD upload."""
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------- rule-based extraction
def _to_tick(h: int, m: int, ampm: Optional[str]) -> Optional[int]:
    if ampm in ("pm",) and h < 12:
        h += 12
    if ampm == "am" and h == 12:
        h = 0
    if not (0 <= h <= 24 and 0 <= m < 60):
        return None
    return min(96, (h * 60 + m) // 15)


def _times(text: str) -> List[int]:
    out = []
    for m in re.finditer(r"\b(\d{1,2})(?:[:.](\d{2}))\s*(am|pm|hrs|hours|h)?\b|\b(\d{1,2})\s*(am|pm)\b", text):
        if m.group(1):
            t = _to_tick(int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower() or None)
        else:
            t = _to_tick(int(m.group(4)), 0, m.group(5).lower())
        if t is not None:
            out.append(t)
    if re.search(r"\bnoon\b", text):
        out.append(48)
    return out


def _duration_ticks(text: str) -> Optional[int]:
    m = re.search(r"(?:for|duration[:\s]*|lasting|approx\.?|about)\s*(\d+(?:\.\d+)?)\s*(hours|hrs|hr|h)\b", text)
    if m:
        return max(1, round(float(m.group(1)) * 4))
    m = re.search(r"(\d+)\s*(minutes|mins|min)\b", text)
    return max(1, round(int(m.group(1)) / 15)) if m else None


def _flex_ticks(text: str) -> int:
    m = re.search(r"(?:±|\+/-|plus or minus|flexible by|can be (?:moved|shifted) by|movable by)\s*(\d+(?:\.\d+)?)\s*"
                  r"(hours|hrs|hr|h)", text)
    return round(float(m.group(1)) * 4) if m else 0


def _assets(text: str, cfg: Any) -> List[str]:
    """Asset / load ids mentioned. Explicit ids ("B1") win; otherwise names, ignoring a name that is only
    part of a longer matched name ("bhadla" inside "bess-bhadla")."""
    items = (*cfg.solar, *cfg.wind, *cfg.batteries, *cfg.consumers)
    explicit = [a.id for a in items if re.search(rf"\b{re.escape(a.id.lower())}\b", text)]
    if explicit:
        return list(dict.fromkeys(explicit))
    hits = []
    for a in items:
        for n in {a.name.lower(), a.name.split("(")[0].strip().lower()}:
            if n and re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", text):
                hits.append((a.id, n))
    names = [n for _, n in hits]
    keep = [aid for aid, n in hits if not any(n != m and n in m for m in names)]
    return list(dict.fromkeys(keep))


def _classify(text: str) -> Tuple[str, float]:
    scores = {k: sum(text.count(w) for w in ws) for k, ws in KEYWORDS.items()}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if ranked[0][1] == 0:
        return "none", 0.0
    margin = (ranked[0][1] - ranked[1][1]) / ranked[0][1]
    return ranked[0][0], margin


def extract_rules(text: str, cfg: Any, now_tick: int = 0) -> Dict[str, Any]:
    """Deterministic extraction into the common schema."""
    t = " ".join(text.lower().split())
    t = re.sub(r"\b\d{1,4}[./-]\d{1,2}[./-]\d{2,4}\b", " ", t)      # drop dates so they are not read as times
    kind, margin = _classify(t)
    times = _times(t)
    rec: Dict[str, Any] = {"kind": kind, "assets": _assets(t, cfg), "start_tick": None, "duration_ticks": None,
                           "value": None, "flex_ticks": _flex_ticks(t), "hedged": any(h in t for h in HEDGES),
                           "class_margin": margin}
    if re.search(r"\b(immediately|with immediate effect|right now|since this morning)\b", t):
        rec["start_tick"] = now_tick
    if times:
        rec["start_tick"] = times[0] if rec["start_tick"] is None else rec["start_tick"]
        if len(times) >= 2 and times[1] > times[0]:
            rec["duration_ticks"] = times[1] - times[0]
    rec["duration_ticks"] = rec["duration_ticks"] or _duration_ticks(t)
    pct = [int(x) for x in re.findall(r"(\d{1,3})\s*%", t) if 0 < int(x) <= 100]
    mw = re.search(r"(?:to|at|of)\s*(\d+(?:\.\d+)?)\s*mw", t)
    add_mw = re.search(r"(?:additional|extra|\+|increase(?:d)? by)\s*(\d+(?:\.\d+)?)\s*mw", t)
    if kind == "breakdown":
        rec["value"] = (pct[0] / 100) if pct else 1.0
    elif kind == "line_limit":
        if pct:
            p = pct[0] / 100
            rec["value"] = 1 - p if re.search(r"reduced by|cut by|lower by", t) else p
        elif mw:
            rec["value"] = float(mw.group(1)) / max(cfg.grid.import_limit_mw, 1)
    elif kind == "demand_change":
        if add_mw:
            rec["value_mw"] = float(add_mw.group(1))
        elif pct:
            rec["value"] = 1 + pct[0] / 100
    return rec


# ---------------------------------------------------------------- optional LLM extraction
EXTRACT_TOOL = {
    "name": "record_notice",
    "description": "Record the single operational notice in this document.",
    "input_schema": {"type": "object", "properties": {
        "kind": {"type": "string", "enum": ["maintenance", "breakdown", "line_limit", "demand_change", "none"]},
        "assets": {"type": "array", "items": {"type": "string"}, "description": "asset or load ids from the list"},
        "start_time": {"type": "string", "description": "HH:MM 24h, or 'now'"},
        "duration_hours": {"type": "number"},
        "flexible_hours": {"type": "number"},
        "capacity_or_limit_fraction": {"type": "number", "description": "breakdown: share lost; line: share available"},
        "extra_mw": {"type": "number"}, "demand_factor": {"type": "number"},
        "hedged": {"type": "boolean", "description": "true if tentative / unconfirmed"}},
        "required": ["kind", "assets", "hedged"]},
}


def extract_llm(text: str, cfg: Any, now_tick: int, llm: Any) -> Dict[str, Any]:
    """An LLM reads the document; output is mapped to the same schema (validation happens later)."""
    ids = [f"{a.id} = {a.name}" for a in (*cfg.solar, *cfg.wind, *cfg.batteries, *cfg.consumers)]
    d = llm.call_tool("You extract operational notices for an energy control room. Use only asset ids from the "
                      "list. Mark hedged=true if the notice is tentative or unconfirmed.",
                      f"Assets: {'; '.join(ids)}\n\nDocument:\n{text[:6000]}", EXTRACT_TOOL)
    start = None
    st = str(d.get("start_time") or "")
    if st.lower() == "now":
        start = now_tick
    elif re.match(r"^\d{1,2}:\d{2}$", st):
        h, m = st.split(":")
        start = _to_tick(int(h), int(m), None)
    rec = {"kind": d.get("kind", "none"), "assets": [a for a in d.get("assets", []) if a],
           "start_tick": start, "duration_ticks": round(d["duration_hours"] * 4) if d.get("duration_hours") else None,
           "flex_ticks": round((d.get("flexible_hours") or 0) * 4), "hedged": bool(d.get("hedged")),
           "class_margin": 1.0, "value": d.get("capacity_or_limit_fraction") or d.get("demand_factor")}
    if d.get("extra_mw"):
        rec["value_mw"] = float(d["extra_mw"])
    return rec


# ---------------------------------------------------------------- validation → Evidence
REQUIRED = {"maintenance": ("assets", "start_tick", "duration_ticks"), "breakdown": ("assets", "start_tick"),
            "line_limit": ("start_tick", "duration_ticks", "value"),
            "demand_change": ("assets", "start_tick", "duration_ticks", "value")}


def to_evidence(rec: Dict[str, Any], name: str, cfg: Any, extractor: str) -> Evidence:
    """Validate an extraction and score its confidence."""
    kind = rec["kind"]
    if kind == "none":
        return Evidence("document", name, "none", f"{name}: no operational notice recognised", 0.0, status="info",
                        details={"extractor": extractor})
    if kind == "demand_change" and rec.get("value") is None and rec.get("value_mw") and rec["assets"]:
        c = next((c for c in cfg.consumers if c.id == rec["assets"][0]), None)
        rec["value"] = 1 + rec["value_mw"] / c.base_mw if c else None
    if kind == "breakdown" and rec.get("duration_ticks") is None:
        rec["duration_ticks"], rec["defaulted_duration"] = 16, True         # assume 4 h until repaired
    valid_ids = {a.id for a in (*cfg.solar, *cfg.wind, *cfg.batteries)} if kind in ("maintenance", "breakdown") \
        else {c.id for c in cfg.consumers}
    assets = [a for a in rec["assets"] if a in valid_ids]
    req = REQUIRED[kind]
    have = sum(1 for f in req if (assets if f == "assets" else rec.get(f)) not in (None, [], 0) or
               (f == "start_tick" and rec.get(f) == 0))
    conf = 0.45 + 0.55 * have / len(req)
    reasons = []
    if len(assets) > 1:
        conf -= 0.3
        reasons.append(f"mentions several assets ({', '.join(assets)})")
    if rec.get("hedged"):
        conf -= 0.35
        reasons.append("notice is tentative / unconfirmed")
    if rec.get("class_margin", 1) < 0.34:
        conf -= 0.15
        reasons.append("document type ambiguous")
    if rec.get("defaulted_duration"):
        conf -= 0.05
    v = rec.get("value")
    if v is not None and (kind == "line_limit" and not 0.05 <= v <= 1 or kind == "breakdown" and not 0 < v <= 1
                          or kind == "demand_change" and not 0.5 <= v <= 2.5):
        conf -= 0.4
        reasons.append(f"implausible value {v}")
    missing = [f for f in req if (assets if f == "assets" else rec.get(f)) in (None, [])]
    if missing:
        reasons.append("missing " + ", ".join(missing))
    conf = round(max(0.0, min(1.0, conf)), 2)
    asset = assets[0] if assets else None
    start, dur = rec.get("start_tick"), rec.get("duration_ticks")
    when = (f"{start // 4:02d}:{start % 4 * 15:02d}" if start is not None else "?") + \
           (f" for {dur / 4:g} h" if dur else "")
    text = {"maintenance": f"maintenance on {asset} at {when}" + (f" (movable ±{rec['flex_ticks'] / 4:g} h)"
                                                                   if rec.get("flex_ticks") else ""),
            "breakdown": f"breakdown of {asset} from {when}, {(v or 1):.0%} of capacity lost",
            "line_limit": f"grid line limited to {(v or 0):.0%} from {when}",
            "demand_change": f"demand of {asset} ×{(v or 1):.2f} from {when}"}[kind]
    ev = Evidence("document", name, kind, f"{name}: {text}", conf, asset, start, dur, v,
                  {"extractor": extractor, "flex_ticks": rec.get("flex_ticks", 0), "fields_found": have,
                   "fields_required": len(req)})
    if conf < APPLY_THRESHOLD:
        ev.status, ev.reason = "quarantined", "; ".join(reasons) or "low confidence"
    return ev


def analyse(name: str, data: bytes, cfg: Any, now_tick: int = 0, llm: Any = None) -> Evidence:
    """Read → extract (LLM if configured, else rules) → validate. LLM failures fall back to rules."""
    text = read_document(name, data)
    rec, extractor = None, "rules"
    if llm is not None:
        try:
            rec, extractor = extract_llm(text, cfg, now_tick, llm), f"llm ({llm.label})"
        except Exception:  # API failure → deterministic parser
            rec = None
    rec = rec or extract_rules(text, cfg, now_tick)
    ev = to_evidence(rec, name, cfg, extractor)
    ev.details["chars"] = len(text)
    return ev
