"""Measure the demo facts used in the deck (storm-graded response, example explanation, learning)."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.pop("ANTHROPIC_API_KEY", None)

from agent.agent import OrchestratorAgent  # noqa: E402
from core.config import load_config  # noqa: E402
from core.events import make_event  # noqa: E402
from core.simulator import Simulator  # noqa: E402

cfg = load_config()
grade, example = [], ""
for prob in (0, 10, 50, 90):
    sim = Simulator(cfg, seed=1)
    ag = OrchestratorAgent(sim, cfg, log_path=None)
    for _ in range(72):
        ag.step()
    if prob:
        sim.apply_event(make_event("storm_alert", 72, probability=prob / 100, hits=False))
    d = ag.step()
    ch = next(o for o in d.options if o["id"] == d.chosen)
    grade.append([prob, round(ch["reserve_pct"] * 100)])
    if prob == 50:
        example = d.explanation
    print(prob, d.chosen_name, grade[-1], flush=True)

sim = Simulator(cfg, seed=1)
ag = OrchestratorAgent(sim, cfg, log_path=None)
ag.run_day()
rep = ag.learner.report()
bias = cfg.sim["forecast_bias"]
learn = {"wind_true": round((1 / (1 + bias["wind"]) - 1) * 100, 1), "wind_learned": rep["wind"]["correction_pct"],
         "demand_true": round((1 / (1 + bias["demand"]) - 1) * 100, 1), "demand_learned": rep["demand"]["correction_pct"],
         "wind_mae_before": rep["wind"]["mae_raw_mw"], "wind_mae_after": rep["wind"]["mae_learned_mw"],
         "demand_mae_before": rep["demand"]["mae_raw_mw"], "demand_mae_after": rep["demand"]["mae_learned_mw"]}
facts = {"storm_grade": grade, "explanation_example": example, "learn": learn}
(ROOT / "deck/facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False))
print(json.dumps(facts, indent=2, ensure_ascii=False))
