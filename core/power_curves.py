"""Standard conversions from weather variables to farm output."""
from __future__ import annotations

import numpy as np

STC_IRRADIANCE = 1000.0  # W/m² at standard test conditions


def solar_mw(ghi_wm2: np.ndarray, capacity_mw: float, performance_ratio: float = 0.9) -> np.ndarray:
    """PV output from global horizontal irradiance: P = Cap · (GHI / 1000) · PR, clipped to capacity."""
    ghi = np.clip(np.nan_to_num(np.asarray(ghi_wm2, dtype=float)), 0.0, None)
    return np.clip(capacity_mw * ghi / STC_IRRADIANCE * performance_ratio, 0.0, capacity_mw)


def wind_mw(speed_ms: np.ndarray, capacity_mw: float, cut_in: float = 3.0,
            rated: float = 12.0, cut_out: float = 25.0) -> np.ndarray:
    """Piecewise cubic turbine curve: 0 below cut-in, cubic ramp to rated, flat, 0 above cut-out."""
    v = np.clip(np.nan_to_num(np.asarray(speed_ms, dtype=float)), 0.0, None)
    ramp = (v ** 3 - cut_in ** 3) / (rated ** 3 - cut_in ** 3)
    out = np.where(v < cut_in, 0.0, np.where(v < rated, ramp, 1.0))
    out = np.where(v >= cut_out, 0.0, out)
    return capacity_mw * np.clip(out, 0.0, 1.0)


def cloud_attenuation(cloud_pct: np.ndarray) -> np.ndarray:
    """Kasten–Czeplak clear-sky attenuation factor for cloud cover in percent."""
    c = np.clip(np.asarray(cloud_pct, dtype=float), 0, 100) / 100.0
    return 1.0 - 0.75 * c ** 3.4
