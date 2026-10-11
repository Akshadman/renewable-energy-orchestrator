"""Sample multimodal pack for demos (thermal images are synthetic with known ground truth)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from core.multimodal.thermal import make_thermal, save_thermal

DOCS = {
    "WO-2231_W2_gearbox_maintenance.pdf": (
        "WORK ORDER WO-2231\nSite: Muppandal wind farm (W2), Tamil Nadu\nDate: 11.10.2026\n"
        "Planned maintenance: gearbox servicing on turbines T4-T9.\n"
        "Schedule: from 15:00 to 18:00.\nThe window is flexible by ±3 hours if grid conditions require.\n"
        "Approved by: Plant O&M head"),
    "SLDC_congestion_notice.pdf": (
        "STATE LOAD DESPATCH CENTRE - GRID OPERATOR NOTICE\nRef: SLDC/OPS/2026/1187\n"
        "Due to transmission congestion on the 220 kV evacuation corridor, the interconnection import limit "
        "for the consumer will be reduced to 60% from 17:30 for 3 hours.\nAll users are requested to comply."),
    "fault_report_BESS_Bhadla.txt": (
        "FAULT REPORT\nAsset: BESS-Bhadla (B1)\nThe power conversion system tripped at 13:45 on an over-temperature "
        "alarm. Battery is out of service with immediate effect.\nExpected restoration: approx 4 hours.\n"),
    "production_notice_steel_mill.txt": (
        "Production planning notice: the Steel mill (C1) will run an extra shift from 19:00 to 23:00 for an export "
        "order, with an additional 15 MW load on the main feeder."),
    "tentative_maintenance_email.txt": (
        "Hi team, tentative plan - we may possibly take Kutch (W1) or Dwarka (W3) down for blade inspection "
        "sometime in the afternoon, to be confirmed tomorrow. Thanks"),
}


def _pdf(text: str, path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path), pagesize=A4)
    y = 800
    for line in text.split("\n"):
        c.drawString(60, y, line)
        y -= 18
    c.save()


def _overexposed(farm: str, path: Path) -> None:
    img, _ = make_thermal(farm, 2, seed=99)
    a = np.asarray(img).astype(int)
    a[: a.shape[0] // 2, : a.shape[1] // 2] += 70                      # large saturated reflection
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), mode="L")
    meta = PngImagePlugin.PngInfo()
    for k, v in {"farm": farm, "tmin": "20", "tmax": "90", "cols": "12", "rows": "10"}.items():
        meta.add_text(k, v)
    out.save(path, format="PNG", pnginfo=meta)


def build_sample_pack(folder: Path) -> Path:
    """Write the demo files to ``folder`` and return it."""
    folder.mkdir(parents=True, exist_ok=True)
    img, _ = make_thermal("S2", 12, seed=7)
    save_thermal(img, folder / "thermal_S2_block3_drone.png")
    img, _ = make_thermal("S1", 0, seed=8)
    save_thermal(img, folder / "thermal_S1_block1_drone.png")
    _overexposed("S4", folder / "thermal_S4_block2_overexposed.png")
    for name, text in DOCS.items():
        if name.endswith(".pdf"):
            _pdf(text, folder / name)
        else:
            (folder / name).write_text(text)
    return folder
