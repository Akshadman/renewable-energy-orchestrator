"""Load, build and save industry configurations (``config/assets.yaml`` and ``config/profiles/*.yaml``)."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from core.models import Battery, Config, Consumer, GridConfig, MarketConfig, SolarFarm, WindFarm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "assets.yaml"
PROFILE_DIR = ROOT / "config" / "profiles"

DEFAULT_TARGETS = {
    "annual_carbon_target_t": 250_000.0,   # tCO2 per year the industry wants to stay under
    "target_cost_rs_per_mwh": 4_500.0,     # desired average energy cost
    "max_shortfall_prob": 0.02,            # tolerated chance of a critical-load shortfall in the look-ahead
    "risk_aversion": 0.3,                  # weight on worst-case (P95) vs average cost
}


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Parse a YAML configuration file and return a :class:`Config`."""
    return config_from_dict(yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text()))


def config_from_dict(raw: Dict[str, Any]) -> Config:
    """Build a :class:`Config` from a plain dict (YAML or UI tables)."""
    return Config(
        solar=[SolarFarm(**s) for s in raw.get("solar_farms") or []],
        wind=[WindFarm(**w) for w in raw.get("wind_farms") or []],
        batteries=[Battery(**b) for b in raw.get("batteries") or []],
        consumers=[Consumer(**c) for c in raw["consumers"]],
        grid=GridConfig(**raw["grid"]),
        market=MarketConfig(**raw["market"]),
        sim=raw.get("sim", {}),
        agent=raw.get("agent", {}),
        data=raw.get("data", {}),
        targets={**DEFAULT_TARGETS, **(raw.get("targets") or {})},
        schedules={"maintenance": [], "conservation": [], **(raw.get("schedules") or {})},
        custom_profiles=raw.get("custom_profiles") or {},
        name=raw.get("name", "Default portfolio"),
    )


def config_to_dict(cfg: Config) -> Dict[str, Any]:
    """Inverse of :func:`config_from_dict` (for saving profiles)."""
    return {
        "name": cfg.name,
        "solar_farms": [asdict(s) for s in cfg.solar],
        "wind_farms": [asdict(w) for w in cfg.wind],
        "batteries": [asdict(b) for b in cfg.batteries],
        "consumers": [asdict(c) for c in cfg.consumers],
        "grid": asdict(cfg.grid),
        "market": asdict(cfg.market),
        "sim": cfg.sim, "agent": cfg.agent, "data": cfg.data,
        "targets": cfg.targets, "schedules": cfg.schedules, "custom_profiles": cfg.custom_profiles,
    }


def validate_config(cfg: Config) -> List[str]:
    """Human-readable problems with an industry configuration (empty list = OK)."""
    errs = []
    if not cfg.solar and not cfg.wind:
        errs.append("Add at least one solar or wind farm.")
    if not cfg.consumers:
        errs.append("Add at least one load / consumer.")
    for kind, items, attr in (("Solar", cfg.solar, "capacity_mw"), ("Wind", cfg.wind, "capacity_mw"),
                              ("Load", cfg.consumers, "base_mw"), ("Battery", cfg.batteries, "power_mw")):
        for it in items:
            if getattr(it, attr) <= 0:
                errs.append(f"{kind} '{it.name}' must have a positive {attr.replace('_', ' ')}.")
    for b in cfg.batteries:
        if not 0 <= b.soc_min < b.soc_max <= 1:
            errs.append(f"Battery '{b.name}': SoC limits must satisfy 0 ≤ min < max ≤ 1.")
    for c in cfg.consumers:
        if not 0 <= c.critical_frac <= 1:
            errs.append(f"Load '{c.name}': critical share must be between 0 and 1.")
    ids = [x.id for x in (*cfg.solar, *cfg.wind, *cfg.batteries, *cfg.consumers)]
    if len(ids) != len(set(ids)):
        errs.append("Asset ids must be unique.")
    if cfg.grid.import_limit_mw < 0 or cfg.grid.export_limit_mw < 0:
        errs.append("Grid limits cannot be negative.")
    return errs


def list_profiles() -> Dict[str, Path]:
    """Saved industry profiles: display name -> path (the default config first)."""
    out = {"Default portfolio (Rajasthan / Gujarat / Tamil Nadu)": DEFAULT_CONFIG}
    for p in sorted(PROFILE_DIR.glob("*.yaml")) if PROFILE_DIR.exists() else []:
        out[yaml.safe_load(p.read_text()).get("name", p.stem)] = p
    return out


def save_profile(cfg: Config) -> Path:
    """Write ``cfg`` to ``config/profiles/<slug>.yaml``."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    slug = "".join(ch if ch.isalnum() else "_" for ch in cfg.name.lower()).strip("_") or "profile"
    path = PROFILE_DIR / f"{slug}.yaml"
    path.write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False, allow_unicode=True))
    return path


def resolve_path(rel: str) -> Path:
    """Resolve a config-relative data path against the project root."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p
