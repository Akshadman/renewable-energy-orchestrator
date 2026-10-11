"""Thermal / drone inspection images of PV blocks → hot-spot detection → farm derate.

Images are radiometric grayscale PNGs (pixel 0–255 ↔ ``tmin``–``tmax`` °C, stored as PNG text
metadata together with ``farm`` and the module grid). Detection: robust z-score of temperature vs
the block median (MAD), connected hot regions, mapped to modules. Defective-module share × 0.5
(typical bypass-diode loss) gives the capacity derate.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, PngImagePlugin

from core.multimodal.evidence import Evidence

LOSS_PER_DEFECT = 0.5        # share of a module's output lost to a hot-spot (bypass diode)
ACTION_SHARE = 0.04          # ≥4% defective modules → schedule inspection / cleaning


def make_thermal(farm: str, defects: int, seed: int = 0, grid: Tuple[int, int] = (12, 10), cell: int = 28,
                 noise: float = 0.8, glare: bool = False, delta: Tuple[float, float] = (15, 30),
                 radius: Tuple[int, int] = (3, 6)) -> Tuple[Image.Image, List[Tuple[int, int]]]:
    """Synthetic radiometric image of a PV block with ``defects`` hot-spot modules. Returns (image, truth)."""
    rng = np.random.default_rng(seed)
    cols, rows = grid
    h, w = rows * cell, cols * cell
    yy, xx = np.mgrid[0:h, 0:w]
    temp = 44 + 3 * (xx / w) + 1.5 * np.sin(yy / h * np.pi) + rng.normal(0, noise, (h, w))
    gap = ((xx % cell) < 3) | ((yy % cell) < 3)
    temp = np.where(gap, 34 + rng.normal(0, noise, (h, w)), temp)          # frames / gaps are cooler
    idx = rng.choice(cols * rows, size=defects, replace=False) if defects else []
    truth = [(int(k) % cols, int(k) // cols) for k in idx]
    for c, r in truth:
        cy, cx = r * cell + rng.integers(8, cell - 8), c * cell + rng.integers(8, cell - 8)
        rad = rng.integers(*radius)
        spot = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2
        temp = np.where(spot, temp + rng.uniform(*delta), temp)
    if glare:                                                                # sun glare: broad warm smear
        temp += 6 * np.exp(-(((xx - w * 0.7) / (w * 0.15)) ** 2 + ((yy - h * 0.3) / (h * 0.2)) ** 2))
    tmin, tmax = 20.0, 90.0
    px = np.clip((temp - tmin) / (tmax - tmin) * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(px, mode="L")
    meta = PngImagePlugin.PngInfo()
    for k, v in {"farm": farm, "tmin": tmin, "tmax": tmax, "cols": cols, "rows": rows}.items():
        meta.add_text(k, str(v))
    img.info["pnginfo"] = meta
    img.text_meta = {"farm": farm, "tmin": str(tmin), "tmax": str(tmax), "cols": str(cols), "rows": str(rows)}
    return img, sorted(truth)


def save_thermal(img: Image.Image, path: Any) -> None:
    img.save(path, format="PNG", pnginfo=img.info.get("pnginfo"))


def _components(mask: np.ndarray, min_px: int = 4) -> List[Tuple[float, float, int]]:
    """Connected regions (4-neighbour) → (cy, cx, size)."""
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack, pts = [(y, x)], []
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            pts.append((cy, cx))
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(pts) >= min_px:
            arr = np.array(pts)
            out.append((float(arr[:, 0].mean()), float(arr[:, 1].mean()), len(pts)))
    return out


def analyse(img: Image.Image, name: str = "image", farm: Optional[str] = None) -> Evidence:
    """Detect hot-spot modules and turn them into a farm-derate finding."""
    info: Dict[str, str] = {**{k: v for k, v in img.info.items() if isinstance(v, str)},
                            **(getattr(img, "text_meta", None) or {}), **(getattr(img, "text", None) or {})}
    farm = farm or info.get("farm")
    tmin, tmax = float(info.get("tmin", 20)), float(info.get("tmax", 90))
    cols, rows = int(info.get("cols", 12)), int(info.get("rows", 10))
    temp = np.asarray(img.convert("L"), dtype=float) / 255 * (tmax - tmin) + tmin
    h, w = temp.shape
    # residual vs each module's own median (removes gradients, glare and frame/gap temperatures)
    resid = np.zeros_like(temp)
    ch, cw = h / rows, w / cols
    mod_med = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            y0, y1, x0, x1 = int(r * ch), int((r + 1) * ch), int(c * cw), int((c + 1) * cw)
            block = temp[y0:y1, x0:x1]
            mod_med[r, c] = np.median(block)
            resid[y0:y1, x0:x1] = block - mod_med[r, c]
    warm_modules = float((np.abs(mod_med - np.median(mod_med)) > 10).mean())   # whole modules far off typical
    pad = np.pad(resid, 1, mode="edge")                                   # 3×3 box filter (sensor noise)
    smooth = sum(pad[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)) / 9
    sigma = float(np.median(np.abs(smooth - np.median(smooth)))) * 1.4826 + 1e-6
    thr = max(4.0, 6 * sigma)
    hot = smooth > thr
    blobs = _components(hot, min_px=3)
    peaks = [float(smooth[int(round(cy)), int(round(cx))]) for cy, cx, _ in blobs]
    med = float(np.median(temp))
    modules = sorted({(min(cols - 1, int(cx / w * cols)), min(rows - 1, int(cy / h * rows))) for cy, cx, _ in blobs})
    share = len(modules) / (cols * rows)
    derate = share * LOSS_PER_DEFECT
    conf = 0.95 if "tmin" in info and farm else 0.7
    why = ""
    if hot.mean() > 0.05 or warm_modules > 0.1:
        conf, why = 0.3, "large warm region across many modules (reflection / exposure problem?)"
    elif sigma > 1.2:
        conf, why = min(conf, 0.6), f"noisy image (σ={sigma:.1f} °C) — detections unreliable"
    elif peaks and min(peaks) < 1.5 * thr:
        conf, why = min(conf, 0.65), "some hot-spots barely above the detection threshold"
    if not farm:
        conf, why = min(conf, 0.5), "farm not identified in image metadata"
    action = share >= ACTION_SHARE
    summary = (f"{farm or '?'}: {len(modules)} of {cols * rows} modules with hot-spots ({share:.1%}) → "
               f"capacity −{derate:.1%}" + ("; inspection crew needed" if action else ""))
    return Evidence("thermal_image", name, "panel_defects" if modules else "none", summary, round(conf, 2), farm,
                    value=round(derate, 4), details={"defective_modules": len(modules), "modules": cols * rows,
                                                     "hotspots": len(blobs), "median_temp_c": round(med, 1),
                                                     "needs_inspection": action, "module_cells": modules},
                    status="info" if not modules else "pending", reason=why)
