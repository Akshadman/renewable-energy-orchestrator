"""Reliability of the multimodal intake on labelled data (the D3 evidence).

  python -m eval.multimodal_eval

1. Thermal images  — synthetic radiometric PV images in 3 difficulty tiers with known defects
2. Documents       — notices generated from many phrasings, 24h/12h times, dates, filler text,
                     plus deliberately ambiguous / tentative ones that must be quarantined
3. Satellite cloud — real cloud-free NASA scenes of our farm sites + synthetic clouds of known cover
Key safety metric: share of *applied* findings that are wrong (must stay ~0).
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import ROOT, load_config  # noqa: E402
from core.multimodal import documents, thermal  # noqa: E402
from core.multimodal.evidence import APPLY_THRESHOLD  # noqa: E402
from core.multimodal.satellite import cloud_fraction  # noqa: E402

OUT = ROOT / "eval" / "results" / "multimodal.json"
TIERS = {"easy": dict(noise=0.8, delta=(15, 30), radius=(3, 6)),
         "medium": dict(noise=1.5, delta=(10, 18), radius=(2, 5)),
         "hard": dict(noise=2.2, delta=(8, 13), radius=(2, 4), glare=True)}


def eval_thermal(n_per_tier: int = 50) -> dict:
    res = {}
    for tier, kw in TIERS.items():
        rng = np.random.default_rng(len(tier))
        exact, abs_err, derate_err, flag_ok, applied, wrong_applied = 0, [], [], 0, 0, 0
        for i in range(n_per_tier):
            n = int(rng.integers(0, 16))
            img, truth = thermal.make_thermal("S2", n, seed=1000 * len(tier) + i, **kw)
            buf = io.BytesIO()
            thermal.save_thermal(img, buf)
            buf.seek(0)
            ev = thermal.analyse(Image.open(buf), "t.png")
            got = ev.details["defective_modules"]
            exact += got == n
            abs_err.append(abs(got - n))
            true_derate = n / 120 * thermal.LOSS_PER_DEFECT
            derate_err.append(abs(ev.value - true_derate) * 100)
            flag_ok += ev.details["needs_inspection"] == (n / 120 >= thermal.ACTION_SHARE)
            if ev.kind == "panel_defects" and ev.confidence >= APPLY_THRESHOLD:
                applied += 1
                wrong_applied += abs(ev.value - true_derate) > 0.02
        res[tier] = {"images": n_per_tier, "exact_count_pct": round(100 * exact / n_per_tier, 1),
                     "mae_modules": round(float(np.mean(abs_err)), 2),
                     "mae_derate_pts": round(float(np.mean(derate_err)), 2),
                     "inspection_flag_accuracy_pct": round(100 * flag_ok / n_per_tier, 1),
                     "applied": applied, "wrong_applied": wrong_applied}
    return res


def _doc_cases(cfg, n: int, rng: np.random.Generator):
    """(text, truth) pairs. truth=None → must be quarantined or ignored."""
    gen = [a for a in (*cfg.solar, *cfg.wind)]
    fmt = lambda t: rng.choice([f"{t // 4:02d}:{t % 4 * 15:02d}", f"{t // 4:02d}.{t % 4 * 15:02d} hrs",  # noqa: E731
                                f"{(t // 4 - 1) % 12 + 1}:{t % 4 * 15:02d} {'pm' if t >= 48 else 'am'}"])
    date = lambda: rng.choice(["", "Date: 11.10.2026\n", "Issued 10/10/2026. ", "Ref date 2026-10-11 - "])  # noqa: E731
    filler = lambda: rng.choice(["", " Please plan accordingly.", " Contact the control room for queries.",  # noqa: E731
                                 " This supersedes earlier notices."])
    cases = []
    for i in range(n):
        kind = rng.choice(["maintenance", "breakdown", "line_limit", "demand_change", "ambiguous"],
                          p=[0.22, 0.22, 0.2, 0.18, 0.18])
        s = int(rng.integers(8, 72)) // 2 * 2
        d = int(rng.choice([4, 8, 12, 16]))
        if kind == "maintenance":
            a = gen[int(rng.integers(len(gen)))]
            name = rng.choice([a.id, f"{a.name.split('(')[0].strip()} ({a.id})", a.name.split("(")[0].strip()])
            flex = int(rng.choice([0, 2, 3]))
            body = rng.choice([f"Work order: planned maintenance of {name} from {fmt(s)} to {fmt(s + d)}.",
                               f"Preventive maintenance scheduled for {name} at {fmt(s)} for {d / 4:g} hours.",
                               f"Servicing of {name}: planned outage starting {fmt(s)}, duration {d / 4:g} hrs."])
            body += f" Window flexible by ±{flex} hours." if flex else ""
            truth = {"kind": kind, "asset": a.id, "start": s, "dur": d}
        elif kind == "breakdown":
            a = gen[int(rng.integers(len(gen)))]
            pct = int(rng.choice([100, 50, 25]))
            body = rng.choice([f"Fault report: {a.id} tripped at {fmt(s)}; expected out of service for {d / 4:g} hours.",
                               f"Forced outage of {a.name.split('(')[0].strip()} ({a.id}) since {fmt(s)} due to inverter "
                               f"failure, about {d / 4:g} hours to repair."])
            body += f" {pct}% of capacity affected." if pct < 100 else ""
            truth = {"kind": kind, "asset": a.id, "start": s, "dur": d, "value": pct / 100}
        elif kind == "line_limit":
            pct = int(rng.choice([40, 50, 60, 70, 80]))
            body = rng.choice([f"SLDC notice: due to transmission congestion the import limit is reduced to {pct}% from "
                               f"{fmt(s)} for {d / 4:g} hours.",
                               f"Grid operator: evacuation corridor congestion; interconnect capacity at {pct}% between "
                               f"{fmt(s)} and {fmt(s + d)}."])
            truth = {"kind": kind, "start": s, "dur": d, "value": pct / 100}
        elif kind == "demand_change":
            c = cfg.consumers[int(rng.integers(3))]
            mw = int(rng.choice([5, 10, 15]))
            body = rng.choice([f"Production notice: {c.name} ({c.id}) will run an extra shift from {fmt(s)} to "
                               f"{fmt(s + d)} with an additional {mw} MW load.",
                               f"{c.name} ({c.id}) overtime batch {fmt(s)}–{fmt(s + d)}, additional {mw} MW."])
            truth = {"kind": kind, "asset": c.id, "start": s, "dur": d, "value": 1 + mw / c.base_mw}
        else:
            a, b = gen[int(rng.integers(len(gen)))], gen[int(rng.integers(len(gen)))]
            body = rng.choice([f"Tentative: we may take {a.id} or {b.id} down sometime this afternoon, to be confirmed.",
                               f"Possibly a maintenance on {a.name.split('(')[0].strip()} later, subject to approval.",
                               "FYI there might be some grid issues later today, details TBC."])
            truth = None
        cases.append((date() + body + filler(), truth))
    return cases


def eval_documents(n: int = 200) -> dict:
    cfg = load_config()
    rng = np.random.default_rng(11)
    tot = {"docs": n, "clear": 0, "ambiguous": 0, "applied": 0, "applied_correct": 0, "wrong_applied": 0,
           "clear_quarantined": 0, "ambiguous_quarantined_or_ignored": 0, "kind_correct": 0}
    for text, truth in _doc_cases(cfg, n, rng):
        ev = documents.analyse("doc.txt", text.encode(), cfg, now_tick=0)
        applied = ev.status == "pending" and ev.confidence >= APPLY_THRESHOLD
        if truth is None:
            tot["ambiguous"] += 1
            tot["ambiguous_quarantined_or_ignored"] += not applied
            tot["wrong_applied"] += applied
            continue
        tot["clear"] += 1
        tot["kind_correct"] += ev.kind == truth["kind"]
        if not applied:
            tot["clear_quarantined"] += 1
            continue
        tot["applied"] += 1
        ok = ev.kind == truth["kind"] and ev.start_tick == truth["start"] and ev.duration_ticks == truth["dur"] \
            and (truth.get("asset") is None or ev.asset == truth["asset"]) \
            and (truth.get("value") is None or abs((ev.value or 0) - truth["value"]) < 0.02)
        tot["applied_correct"] += ok
        tot["wrong_applied"] += not ok
    tot["kind_accuracy_pct"] = round(100 * tot["kind_correct"] / max(1, tot["clear"]), 1)
    tot["applied_precision_pct"] = round(100 * tot["applied_correct"] / max(1, tot["applied"]), 1)
    tot["clear_coverage_pct"] = round(100 * tot["applied"] / max(1, tot["clear"]), 1)
    tot["ambiguous_caught_pct"] = round(100 * tot["ambiguous_quarantined_or_ignored"] / max(1, tot["ambiguous"]), 1)
    return tot


def eval_satellite(n: int = 60) -> dict:
    scenes = sorted((ROOT / "data/cache/satellite").glob("gibs_*.png"))
    clear = [np.asarray(Image.open(p).convert("RGB")) for p in scenes]
    clear = [s for s in clear if cloud_fraction(s)[0] < 3 and cloud_fraction(s)[1] > 0.9]
    if not clear:
        return {"scenes": 0, "note": "no cached real scenes — run the dashboard with live weather once"}
    rng = np.random.default_rng(5)
    errs = []
    for i in range(n):
        bg = clear[i % len(clear)].astype(float)
        h, w, _ = bg.shape
        yy, xx = np.mgrid[0:h, 0:w]
        field = np.zeros((h, w))
        for _ in range(int(rng.integers(0, 6))):
            cy, cx, r = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(8, 40)
            field += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2)))
        mask = field > 0.5
        img = np.where(mask[..., None], 235 + rng.normal(0, 6, (h, w, 1)), bg)
        truth = 100 * mask.mean()
        got, _ = cloud_fraction(np.clip(img, 0, 255).astype(np.uint8))
        errs.append(abs(got - truth))
    return {"real_background_scenes": len(clear), "synthetic_cloud_tests": n,
            "mae_cloud_pct_pts": round(float(np.mean(errs)), 2), "max_error_pts": round(float(np.max(errs)), 1),
            "within_10pts_pct": round(100 * float(np.mean(np.array(errs) <= 10)), 1)}


def main() -> None:
    out = {"thermal": eval_thermal(), "documents": eval_documents(), "satellite": eval_satellite()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
