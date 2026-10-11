"""Satellite imagery: NASA GIBS MODIS true-colour scenes → measured cloud fraction per solar farm.

The measured cloud is fused with the weather-API cloud cover at the satellite overpass (~10:30 IST).
If the satellite sees substantially more (less) cloud than the forecast, that becomes a 'cloud_bank'
('clearer_sky') finding that derates (uplifts) the farm around the overpass. Scenes from a fallback date are informational only.
"""
from __future__ import annotations

import io
from datetime import date as Date, timedelta
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

from core.multimodal.evidence import Evidence
from core.power_curves import cloud_attenuation

GIBS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"
OVERPASS_TICK = 42            # ~10:30 IST (Terra equator crossing)
HALF_DEG = 0.25               # scene half-width around the farm
DISAGREE_PTS = 25.0           # satellite − forecast cloud (percentage points) that triggers a finding


def cloud_fraction(img: np.ndarray) -> Tuple[float, float]:
    """(cloud %, valid-pixel share) using a white & unsaturated test; black = no-data pixels."""
    rgb = img[..., :3].astype(float) / 255.0
    mx, mn = rgb.max(-1), rgb.min(-1)
    valid = mx > 0.06
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    cloud = (mn > 0.70) & (sat < 0.15) & valid
    v = float(valid.mean())
    return (100.0 * float(cloud.sum()) / max(1, int(valid.sum())), v)


def fetch_scene(lat: float, lon: float, day: Date, cache_dir: Path, http_get: Optional[Callable[..., Any]] = None,
                size: int = 128) -> np.ndarray:
    """Download (or load cached) true-colour scene centred on (lat, lon)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"gibs_{day.isoformat()}_{lat:.2f}_{lon:.2f}.png"
    if path.exists():
        return np.asarray(Image.open(path).convert("RGB"))
    params = {"SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.3.0", "LAYERS": LAYER, "CRS": "EPSG:4326",
              "BBOX": f"{lat - HALF_DEG},{lon - HALF_DEG},{lat + HALF_DEG},{lon + HALF_DEG}", "WIDTH": size,
              "HEIGHT": size, "FORMAT": "image/jpeg", "TIME": day.isoformat()}
    if http_get is None:
        import requests
        http_get = requests.get
    resp = http_get(GIBS_URL, params=params, timeout=20)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img.save(path)
    return np.asarray(img)


def analyse_sites(cfg: Any, weather: Any, day: Date, cache_dir: Path,
                  http_get: Optional[Callable[..., Any]] = None) -> List[Evidence]:
    """One Evidence per solar farm: measured vs forecast cloud at overpass."""
    out: List[Evidence] = []
    for i, farm in enumerate(cfg.solar):
        scene, used, note = None, day, ""
        for d in (day, day - timedelta(days=1)):          # today's scene may not be published yet
            try:
                img = fetch_scene(farm.lat, farm.lon, d, cache_dir, http_get)
                frac, valid = cloud_fraction(img)
                if valid > 0.5:
                    scene, used = (frac, valid), d
                    break
                note = "scene mostly no-data"
            except Exception as exc:   # network / service errors → try fallback date
                note = f"fetch failed: {type(exc).__name__}"
        if scene is None:
            out.append(Evidence("satellite", f"MODIS {day}", "none", f"{farm.id}: no usable satellite scene ({note})",
                                0.0, farm.id, status="rejected", reason=note))
            continue
        sat_cloud, valid = scene
        api_cloud = float(weather.cloud_cover[OVERPASS_TICK, i]) if weather.cloud_cover.shape[1] > i else 0.0
        same_day = used == day
        conf = round(min(1.0, valid) * (1.0 if same_day else 0.6), 2)
        gap = sat_cloud - api_cloud
        src = f"NASA MODIS Terra {used.isoformat()} ({farm.name})"
        if gap >= DISAGREE_PTS:
            f_api = float(cloud_attenuation(np.array([api_cloud]))[0])
            f_sat = float(cloud_attenuation(np.array([min(sat_cloud, 100.0)]))[0])
            derate = float(np.clip(1 - f_sat / max(f_api, 1e-6), 0, 0.9))
            out.append(Evidence("satellite", src, "cloud_bank",
                                f"{farm.id}: satellite sees {sat_cloud:.0f}% cloud vs {api_cloud:.0f}% forecast → "
                                f"solar −{derate:.0%} around overpass", conf, farm.id, OVERPASS_TICK - 4, 16, derate,
                                {"sat_cloud_pct": round(sat_cloud, 1), "api_cloud_pct": round(api_cloud, 1),
                                 "scene_date": used.isoformat()}))
        elif gap <= -DISAGREE_PTS:
            f_api = float(cloud_attenuation(np.array([api_cloud]))[0])
            f_sat = float(cloud_attenuation(np.array([max(sat_cloud, 0.0)]))[0])
            uplift = float(np.clip(f_sat / max(f_api, 1e-6) - 1, 0, 1.5))
            out.append(Evidence("satellite", src, "clearer_sky",
                                f"{farm.id}: satellite sees only {sat_cloud:.0f}% cloud vs {api_cloud:.0f}% forecast → "
                                f"solar +{uplift:.0%} around overpass", conf, farm.id, OVERPASS_TICK - 4, 16, uplift,
                                {"sat_cloud_pct": round(sat_cloud, 1), "api_cloud_pct": round(api_cloud, 1),
                                 "scene_date": used.isoformat()}))
        else:
            out.append(Evidence("satellite", src, "none",
                                f"{farm.id}: satellite {sat_cloud:.0f}% cloud, forecast {api_cloud:.0f}% — consistent",
                                conf, farm.id, details={"sat_cloud_pct": round(sat_cloud, 1),
                                                        "api_cloud_pct": round(api_cloud, 1),
                                                        "scene_date": used.isoformat()}, status="info"))
    return out
