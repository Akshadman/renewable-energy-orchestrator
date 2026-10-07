"""Load ``config/assets.yaml`` into typed dataclasses."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import yaml

from core.models import Battery, Config, Consumer, GridConfig, MarketConfig, SolarFarm, WindFarm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "assets.yaml"


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Parse the YAML asset file and return a :class:`Config`."""
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    return Config(
        solar=[SolarFarm(**s) for s in raw["solar_farms"]],
        wind=[WindFarm(**w) for w in raw["wind_farms"]],
        batteries=[Battery(**b) for b in raw["batteries"]],
        consumers=[Consumer(**c) for c in raw["consumers"]],
        grid=GridConfig(**raw["grid"]),
        market=MarketConfig(**raw["market"]),
        sim=raw.get("sim", {}),
        agent=raw.get("agent", {}),
        data=raw.get("data", {}),
    )


def resolve_path(rel: str) -> Path:
    """Resolve a config-relative data path against the project root."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p
