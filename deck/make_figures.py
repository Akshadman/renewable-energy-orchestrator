"""Figures for the deck: a real NASA scene with the detected cloud mask, and a thermal image with detections."""
import io
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.config import load_config  # noqa: E402
from core.multimodal import thermal  # noqa: E402
from core.multimodal.satellite import cloud_fraction, fetch_scene  # noqa: E402

OUT = ROOT / "deck" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config()

# 1 · a real satellite scene that contains some cloud, with the detected cloud mask outlined
best = None
for farm in cfg.solar:
    for k in range(0, 60, 3):
        d = date(2026, 9, 1) - timedelta(days=k)
        try:
            img = fetch_scene(farm.lat, farm.lon, d, ROOT / "data/cache/satellite_fig", size=320)
        except Exception:
            continue
        frac, valid = cloud_fraction(img)
        if valid > 0.9 and 15 < frac < 55:
            best = (img, farm, d, frac)
            break
    if best:
        break
if best:
    img, farm, d, frac = best
    rgb = img[..., :3].astype(float) / 255
    mx, mn = rgb.max(-1), rgb.min(-1)
    mask = (mn > 0.70) & (((mx - mn) / np.maximum(mx, 1e-6)) < 0.15)
    over = img.copy()
    edge = mask ^ np.roll(mask, 1, 0) | mask ^ np.roll(mask, 1, 1)
    over[edge] = [235, 104, 52]
    pair = Image.new("RGB", (660, 320), "white")
    pair.paste(Image.fromarray(img[..., :3]), (0, 0))
    pair.paste(Image.fromarray(over[..., :3]), (340, 0))
    pair.save(OUT / "satellite.png")
    (OUT / "satellite.txt").write_text(f"{farm.name} · NASA MODIS Terra · {d.isoformat()} · detected cloud {frac:.0f}%")
    print("satellite", farm.id, d, round(frac))

# 2 · thermal image (false colour) with detected hot-spot modules boxed
timg, truth = thermal.make_thermal("S2", 12, seed=7)
buf = io.BytesIO()
thermal.save_thermal(timg, buf)
buf.seek(0)
ev = thermal.analyse(Image.open(buf), "thermal_S2_block3_drone.png")
g = np.asarray(timg, dtype=float) / 255
lut = np.stack([np.clip(1.5 * g - 0.2, 0, 1), np.clip(2.2 * g - 0.9, 0, 1), np.clip(0.9 - 1.6 * abs(g - 0.35), 0, 1)], -1)
col = Image.fromarray((lut * 255).astype(np.uint8)).resize((672, 560))
dr = ImageDraw.Draw(col)
cw, ch = 672 / 12, 560 / 10
for c, r in ev.details["module_cells"]:
    dr.rectangle([c * cw + 2, r * ch + 2, (c + 1) * cw - 2, (r + 1) * ch - 2], outline=(255, 255, 255), width=3)
col.save(OUT / "thermal.png")
(OUT / "thermal.txt").write_text(ev.summary)
print("thermal", ev.summary)
