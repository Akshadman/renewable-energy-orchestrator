"""Plan recipes as *bands*: every recipe allows a range for every parameter, and the AI searches
inside the bands for the best combination.

Search per decision (all candidates are optimised, stress-tested and ranked by the same ladder):
  1. EXPLORE — a few combinations spread across every recipe's band (centre + stratified samples)
  2. REFINE  — new combinations around the best one found, still inside its band
So the chosen plan can land anywhere inside a band (e.g. 56% reserve), not just on fixed points.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.models import Config
from core.optimizer import Criteria

PARAMS = ["reserve_pct", "w_cost", "w_clean", "w_reliability", "caution", "max_dr_frac"]
PARAM_LABELS = {
    "reserve_pct": "Battery reserve", "w_cost": "Price sensitivity", "w_clean": "Clean-energy weight",
    "w_reliability": "Reliability weight", "caution": "Forecast caution", "max_dr_frac": "Demand-response use",
}

DEFAULT_RECIPES: List[Dict[str, Any]] = [
    {"name": "Max savings", "reserve_pct": [0.05, 0.30], "w_cost": [0.55, 0.80], "w_clean": [0.10, 0.30],
     "w_reliability": [0.05, 0.20], "caution": [0.0, 0.3], "max_dr_frac": [1.0, 1.0]},
    {"name": "Green-lean", "reserve_pct": [0.15, 0.45], "w_cost": [0.20, 0.40], "w_clean": [0.45, 0.70],
     "w_reliability": [0.10, 0.25], "caution": [0.0, 0.4], "max_dr_frac": [1.0, 1.0]},
    {"name": "Balanced", "reserve_pct": [0.20, 0.50], "w_cost": [0.35, 0.55], "w_clean": [0.20, 0.40],
     "w_reliability": [0.20, 0.35], "caution": [0.2, 0.6], "max_dr_frac": [1.0, 1.0]},
    {"name": "Protective", "reserve_pct": [0.45, 0.75], "w_cost": [0.20, 0.40], "w_clean": [0.10, 0.25],
     "w_reliability": [0.40, 0.65], "caution": [0.5, 0.9], "max_dr_frac": [1.0, 1.0]},
    {"name": "Fortress", "reserve_pct": [0.70, 0.90], "w_cost": [0.10, 0.25], "w_clean": [0.05, 0.15],
     "w_reliability": [0.60, 0.85], "caution": [0.8, 1.0], "max_dr_frac": [1.0, 1.0]},
]


@dataclass
class Recipe:
    name: str
    bands: Dict[str, Tuple[float, float]]

    def centre(self) -> Dict[str, float]:
        return {p: (lo + hi) / 2 for p, (lo, hi) in self.bands.items()}

    def clip(self, point: Dict[str, float]) -> Dict[str, float]:
        return {p: float(np.clip(point[p], *self.bands[p])) for p in PARAMS}


def load_recipes(cfg: Config) -> List[Recipe]:
    """Recipes from ``agent.recipes`` in the industry config, else the defaults. Bad bands are repaired."""
    out = []
    for r in cfg.agent.get("recipes") or DEFAULT_RECIPES:
        bands = {}
        for p in PARAMS:
            lo, hi = (r.get(p) or next(d[p] for d in DEFAULT_RECIPES if d["name"] == "Balanced"))[:2]
            lo, hi = float(np.clip(min(lo, hi), 0, 1)), float(np.clip(max(lo, hi), 0, 1))
            bands[p] = (lo, min(hi, 0.9) if p == "reserve_pct" else hi)
        out.append(Recipe(str(r.get("name", f"Recipe {len(out) + 1}")), bands))
    return out


def to_criteria(point: Dict[str, float]) -> Criteria:
    """A point in parameter space → optimizer criteria (weights get normalised to sum to 1)."""
    return Criteria(w_cost=point["w_cost"], w_clean=point["w_clean"], w_reliability=point["w_reliability"],
                    reserve_pct=point["reserve_pct"], caution=point["caution"],
                    max_dr_frac=point["max_dr_frac"]).normalized()


def explore(recipes: List[Recipe], rng: np.random.Generator, per_recipe: int = 3) -> List[Tuple[Recipe, Dict]]:
    """Centre of every band + samples stratified along the reserve axis (low / high part of the band)."""
    out = []
    for r in recipes:
        out.append((r, r.centre()))
        for k in range(per_recipe - 1):
            pt = {p: rng.uniform(*r.bands[p]) for p in PARAMS}
            lo, hi = r.bands["reserve_pct"]
            seg = (hi - lo) / max(1, per_recipe - 1)
            pt["reserve_pct"] = rng.uniform(lo + k * seg, lo + (k + 1) * seg)
            out.append((r, pt))
    return out


def refine(recipe: Recipe, best: Dict[str, float], rng: np.random.Generator, n: int = 6,
           step: float = 0.25) -> List[Tuple[Recipe, Dict]]:
    """New combinations around ``best`` inside its band: a cheaper and a safer reserve step, then
    random perturbations of every parameter (± ``step`` of the band width)."""
    lo, hi = recipe.bands["reserve_pct"]
    width = hi - lo
    out = [(recipe, recipe.clip({**best, "reserve_pct": best["reserve_pct"] - step * width})),
           (recipe, recipe.clip({**best, "reserve_pct": best["reserve_pct"] + step * width}))]
    for _ in range(max(0, n - 2)):
        pt = {p: best[p] + rng.normal(0, step / 2) * (recipe.bands[p][1] - recipe.bands[p][0]) for p in PARAMS}
        out.append((recipe, recipe.clip(pt)))
    return out


def recipes_to_rows(recipes: List[Recipe]) -> List[Dict[str, Any]]:
    """For the Setup tab editor: one row per recipe, min/max per parameter in %."""
    rows = []
    for r in recipes:
        row: Dict[str, Any] = {"name": r.name}
        for p in PARAMS:
            row[f"{p}_min"], row[f"{p}_max"] = round(r.bands[p][0] * 100), round(r.bands[p][1] * 100)
        rows.append(row)
    return rows


def rows_to_recipes(rows: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Inverse of :func:`recipes_to_rows` (values in %) → config dicts."""
    out = []
    for row in rows:
        if not row.get("name"):
            continue
        out.append({"name": str(row["name"]), **{p: [float(row.get(f"{p}_min", 0) or 0) / 100,
                                                      float(row.get(f"{p}_max", 0) or 0) / 100] for p in PARAMS}})
    return out or None
