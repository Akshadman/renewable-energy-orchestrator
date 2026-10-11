"""Turn multimodal files into Evidence and apply accepted findings to the digital twin(s)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

from PIL import Image

from core.models import Alert, Event
from core.multimodal import documents, thermal
from core.multimodal.evidence import APPLY_THRESHOLD, Evidence

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
DOC_EXT = (".pdf", ".txt", ".md")


def process(files: Iterable[Tuple[str, bytes]], cfg: Any, now_tick: int = 0,
            llm: Any = None) -> List[Evidence]:
    """Analyse uploaded images & documents. Unsupported or unreadable files become 'rejected' evidence."""
    out: List[Evidence] = []
    for name, data in files:
        low = name.lower()
        try:
            if low.endswith(IMAGE_EXT):
                import io
                ev = thermal.analyse(Image.open(io.BytesIO(data)), name)
            elif low.endswith(DOC_EXT):
                ev = documents.analyse(name, data, cfg, now_tick, llm)
            else:
                ev = Evidence("unknown", name, "none", f"{name}: unsupported file type", 0.0, status="rejected",
                              reason="supported: PNG/JPG thermal images, PDF/TXT/MD documents")
        except Exception as exc:  # corrupt file etc. — never crash the control loop
            ev = Evidence("unknown", name, "none", f"{name}: could not be read", 0.0, status="rejected",
                          reason=f"{type(exc).__name__}: {exc}"[:200])
        if ev.status == "pending" and ev.confidence < APPLY_THRESHOLD:
            ev.status = "quarantined"
            ev.reason = ev.reason or "low confidence"
        out.append(ev)
    return out


def apply(ev: Evidence, sim: Any, now_tick: int) -> Evidence:
    """Apply one accepted finding to a simulator (idempotent per simulator via ``sim.evidence_ids``)."""
    if ev.status not in ("pending", "applied") or ev.confidence < APPLY_THRESHOLD:
        return ev
    from core.simulator import MaintenanceRequest
    T = sim.T
    start = max(now_tick, ev.start_tick if ev.start_tick is not None else now_tick)
    dur = ev.duration_ticks or (T - start)
    sl = slice(min(T, start), min(T, start + dur))
    kind = ev.kind
    try:
        if kind == "cloud_bank":
            _, i = sim.asset_index(ev.asset)
            sim.solar_mult[sl, i] *= 1 - float(ev.value)
        elif kind == "clearer_sky":
            _, i = sim.asset_index(ev.asset)
            cap = sim.cfg.solar[i].capacity_mw
            base = sim.profile.solar[sl, i] * sim.solar_mult[sl, i]
            up = 1 + float(ev.value)
            sim.solar_mult[sl, i] *= [min(up, cap / b) if b > 1e-6 else 1.0 for b in base]
        elif kind == "panel_defects":
            sim.add_outage(ev.asset, now_tick, T - now_tick, "degradation", float(ev.value))
            if ev.details.get("needs_inspection"):
                sim.maintenance.append(MaintenanceRequest(f"M{len(sim.maintenance) + 1}", ev.asset,
                                                          min(T - 9, now_tick + 8), 8, 24))
        elif kind == "maintenance":
            sim.maintenance.append(MaintenanceRequest(f"M{len(sim.maintenance) + 1}", ev.asset, start, dur,
                                                      int(ev.details.get("flex_ticks", 0))))
        elif kind == "breakdown":
            sim.add_outage(ev.asset, start, dur, "breakdown", float(ev.value or 1.0))
        elif kind == "line_limit":
            sim.line_mult[sl] *= float(ev.value)
        elif kind == "demand_change":
            ci = next(k for k, c in enumerate(sim.cfg.consumers) if c.id == ev.asset)
            sim.demand_mult[sl, ci] *= float(ev.value)
        else:
            return ev
    except (KeyError, StopIteration, ValueError) as exc:
        ev.status, ev.reason = "rejected", f"could not apply: {exc}"
        return ev
    sim.events.append(Event(f"evidence:{kind}", start, dur, {"source": ev.source}, ev.summary))
    sim.event_alerts.append(Alert(f"evidence_{kind}", now_tick, f"[{ev.modality}] {ev.summary}",
                                  "critical" if kind in ("breakdown", "line_limit") else "warning",
                                  max(0, start - now_tick), 1.0, dur))
    ev.status = "applied"
    return ev


def apply_all(evidence: List[Evidence], sims: List[Any], now_tick: int) -> List[Evidence]:
    """Apply every accepted finding to each simulator (AI twin and baseline twin)."""
    for ev in evidence:
        if ev.status == "pending" and ev.confidence >= APPLY_THRESHOLD:
            for sim in sims:
                ev.status = "pending" if ev.status == "applied" else ev.status
                apply(ev, sim, now_tick)
    return evidence


def load_folder(folder: Path) -> List[Tuple[str, bytes]]:
    """All supported files in a folder (sample pack)."""
    return [(p.name, p.read_bytes()) for p in sorted(folder.iterdir())
            if p.suffix.lower() in IMAGE_EXT + DOC_EXT]
