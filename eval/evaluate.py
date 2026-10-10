"""Monte Carlo evaluation: agent vs naive baseline vs no-battery over N randomized days.

Usage:
    python -m eval.evaluate --n 100            # offline reasoner (default, free)
    python -m eval.evaluate --n 10 --llm       # consult the LLM (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.baseline import NaiveController  # noqa: E402
from core.config import ROOT, load_config  # noqa: E402
from core.events import random_schedule  # noqa: E402
from core.models import Event  # noqa: E402
from core.simulator import Simulator  # noqa: E402

CONTROLLERS = ["agent", "baseline", "no_battery"]
METRICS = {"cost_rs": "Total cost (₹)", "clean_pct": "Clean energy (%)", "curtailed_mwh": "Curtailment (MWh)",
           "unserved_mwh": "Unserved load (MWh)", "unserved_critical_mwh": "Unserved critical (MWh)",
           "violations": "Constraint violations", "co2_t": "CO₂ (t)"}
OUT_DIR = ROOT / "eval" / "results"


def run_day(seed: int, controller: str, n_events: int = 3, use_llm: bool = False,
            config_path: Optional[str] = None) -> Dict[str, float]:
    """Simulate one randomized day with the given controller and return its totals."""
    cfg = load_config(config_path)
    sim = Simulator(cfg, seed=seed, source="synthetic", use_batteries=controller != "no_battery")
    schedule = random_schedule(np.random.default_rng(10_000 + seed), sim.T, len(cfg.batteries), n_events)
    pending = [Event(e.kind, e.start_tick, e.duration, dict(e.params)) for e in schedule]
    if controller == "agent":
        from agent.agent import OrchestratorAgent
        agent = OrchestratorAgent(sim, cfg, use_llm=use_llm, log_path=None, parallel=False)
        act = lambda: agent.step()  # noqa: E731
    else:
        ctrl = NaiveController(cfg, use_battery=controller == "baseline")
        act = lambda: sim.step(ctrl.decide(sim.state()))  # noqa: E731
    while not sim.done:
        for ev in [e for e in pending if e.start_tick == sim.tick]:   # shocks arrive unannounced
            sim.apply_event(ev)
        act()
    out = {"seed": seed, "controller": controller, "events": "|".join(e.kind for e in schedule)}
    out.update(sim.totals())
    return out


def _job(args: tuple) -> Dict[str, float]:
    return run_day(*args)


def run_monte_carlo(n: int = 100, seed0: int = 0, n_events: int = 3, use_llm: bool = False,
                    workers: Optional[int] = None, progress=None, config_path: Optional[str] = None,
                    controllers: Optional[List[str]] = None) -> pd.DataFrame:
    """Run ``n`` days × 3 controllers (same seeds & events for each) and return one row per run."""
    if not use_llm:
        os.environ.pop("ANTHROPIC_API_KEY", None)  # mock reasoner for speed/cost
    jobs = [(seed0 + i, c, n_events, use_llm, config_path) for i in range(n) for c in (controllers or CONTROLLERS)]
    workers = workers if workers is not None else (1 if use_llm else min(8, os.cpu_count() or 1))
    rows: List[Dict[str, float]] = []
    if workers <= 1:
        for k, job in enumerate(jobs):
            rows.append(_job(job))
            if progress:
                progress((k + 1) / len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for k, row in enumerate(pool.map(_job, jobs, chunksize=2)):
                rows.append(row)
                if progress:
                    progress((k + 1) / len(jobs))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / P5 / P95 per controller and metric."""
    g = df.groupby("controller")[list(METRICS)]
    out = pd.concat({"mean": g.mean(), "p5": g.quantile(0.05), "p95": g.quantile(0.95)}, axis=1)
    return out.reindex(CONTROLLERS)


def save_outputs(df: pd.DataFrame, out_dir: Path = OUT_DIR) -> Dict[str, Path]:
    """Write CSVs and interactive HTML charts."""
    import plotly.express as px
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"runs": out_dir / "results.csv", "summary": out_dir / "summary.csv"}
    df.to_csv(paths["runs"], index=False)
    summarize(df).to_csv(paths["summary"])
    colors = dict(zip(CONTROLLERS, ["#2a78d6", "#eb6834", "#1baf7a"]))
    for metric, label in METRICS.items():
        fig = px.box(df, x="controller", y=metric, color="controller", points="all", category_orders={
            "controller": CONTROLLERS}, color_discrete_map=colors, labels={metric: label, "controller": ""},
            title=f"{label} — {df.seed.nunique()} randomized days")
        fig.update_layout(showlegend=False, template="plotly_white")
        paths[metric] = out_dir / f"{metric}.html"
        fig.write_html(paths[metric], include_plotlyjs="cdn")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="number of randomized days")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--events", type=int, default=3, help="random events per day")
    ap.add_argument("--llm", action="store_true", help="use the LLM reasoner (slow, costs API credits)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--config", default=None, help="alternative YAML config (e.g. different recipe bands)")
    ap.add_argument("--agent-only", action="store_true", help="run only the AI (for A/B comparisons)")
    ap.add_argument("--out", default=None, help="write results CSV here instead of eval/results/")
    a = ap.parse_args()
    df = run_monte_carlo(a.n, a.seed, a.events, a.llm, a.workers,
                         progress=lambda f: print(f"\r{f:6.1%}", end="", flush=True), config_path=a.config,
                         controllers=["agent"] if a.agent_only else None)
    print()
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
        print(df[list(METRICS)].mean().round(2))
        return
    paths = save_outputs(df)
    pd.set_option("display.width", 200)
    print(summarize(df)["mean"].round(2))
    print(f"\nSaved {paths['runs']} and charts in {paths['runs'].parent}")


if __name__ == "__main__":
    main()
