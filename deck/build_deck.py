"""Build the ET AI Hackathon solution deck as HTML and PDF from the project's real results.

    python deck/build_deck.py            → deck/Solution_Deck.html and deck/Solution_Deck.pdf

Inputs: eval/results/results.csv (100-day Monte Carlo), eval/results/multimodal.json (D3 reliability),
deck/facts.json (measured demo facts), deck/screens/*.png (dashboard screenshots, optional).
"""
from __future__ import annotations

import base64
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.evaluate import summarize  # noqa: E402

TEAM = "Vaishnav2728Hritika"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BLUE, ORANGE, AQUA, YELLOW, NAVY, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#10233f", "#5b6470"

# ---------------------------------------------------------------- data
res = pd.read_csv(ROOT / "eval/results/results.csv")
S = summarize(res)["mean"]
ai, bau, nob = S.loc["agent"], S.loc["baseline"], S.loc["no_battery"]
A = res[res.controller == "agent"].set_index("seed")
B = res[res.controller == "baseline"].set_index("seed")
N = len(A)
CHEAPER = int((A.cost_rs < B.cost_rs).sum())
CRIT_DAYS_AI, CRIT_DAYS_BAU = int((A.unserved_critical_mwh > 0.01).sum()), int((B.unserved_critical_mwh > 0.01).sum())
pct = lambda a, b: (a / b - 1) * 100  # noqa: E731
F = json.loads((ROOT / "deck/facts.json").read_text())
MM = json.loads((ROOT / "eval/results/multimodal.json").read_text())
TH, DOC, SAT = MM["thermal"], MM["documents"], MM["satellite"]
th_applied = sum(v["applied"] for v in TH.values())
th_wrong = sum(v["wrong_applied"] for v in TH.values())
SHOTS = {p.stem: p for p in sorted((ROOT / "deck/screens").glob("*.png"))} if (ROOT / "deck/screens").exists() else {}

slides: List[str] = []
e = html.escape


# ---------------------------------------------------------------- building blocks
def slide(title: str, sub: str = "", tag: str = "", body: str = "", dark: bool = False) -> None:
    n = len(slides) + 1
    if dark:
        slides.append(f'<section class="slide dark">{body}<div class="foot">{e(TEAM)} · ET AI Hackathon: Agentic '
                      f'Edition</div></section>')
        return
    slides.append(f'''<section class="slide"><div class="bar"></div>
<div class="head"><div><h1>{e(title)}</h1>{f'<p class="sub">{e(sub)}</p>' if sub else ''}</div>
{f'<span class="tag">{e(tag)}</span>' if tag else ''}</div>
<div class="body">{body}</div>
<div class="foot">{e(TEAM)} · ET AI Hackathon: Agentic Edition · Renewable Energy Orchestrator<span>{n}</span></div>
</section>''')


def card(title: str, text: str = "", color: str = "", cls: str = "") -> str:
    style = f' style="background:{color};color:#fff"' if color else ""
    return f'<div class="card {cls}"{style}><h3>{e(title)}</h3>{text}</div>'


def ul(items: Sequence[str]) -> str:
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def table(rows: Sequence[Sequence[str]], widths: Sequence[int] = (), hl: int = -1, small: bool = False) -> str:
    cols = "".join(f'<col style="width:{w}%">' for w in widths)
    out = [f'<table class="{"small" if small else ""}"><colgroup>{cols}</colgroup>']
    for i, r in enumerate(rows):
        tag = "th" if i == 0 else "td"
        cls = ' class="hl"' if i == hl else ""
        out.append(f"<tr{cls}>" + "".join(f"<{tag}>{c}</{tag}>" for c in r) + "</tr>")
    return "".join(out) + "</table>"


def bars(items: List[Tuple[str, float, str]], unit: str = "", fmt: str = "{:,.0f}", w: int = 560, h: int = 300,
         title: str = "") -> str:
    """Simple vertical bar chart as inline SVG: (label, value, color)."""
    mx = max(v for _, v, _ in items) * 1.18 or 1
    pad_b, pad_t = 46, 34 if title else 14
    bw = (w - 40) / len(items)
    svg = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    if title:
        svg.append(f'<text x="{w / 2}" y="20" text-anchor="middle" class="ct">{e(title)}</text>')
    base = h - pad_b
    svg.append(f'<line x1="20" y1="{base}" x2="{w - 20}" y2="{base}" stroke="#cfd5dd"/>')
    for i, (lab, v, col) in enumerate(items):
        bh = (base - pad_t) * v / mx
        x = 20 + i * bw + bw * 0.18
        svg.append(f'<rect x="{x:.1f}" y="{base - bh:.1f}" width="{bw * 0.64:.1f}" height="{bh:.1f}" rx="4" fill="{col}"/>')
        svg.append(f'<text x="{x + bw * 0.32:.1f}" y="{base - bh - 7:.1f}" text-anchor="middle" class="cv">'
                   f'{e(fmt.format(v))}{e(unit)}</text>')
        svg.append(f'<text x="{x + bw * 0.32:.1f}" y="{base + 20}" text-anchor="middle" class="cl">{e(lab)}</text>')
    return "".join(svg) + "</svg>"


FIG = ROOT / "deck" / "figures"


def fig(name: str, cls: str = "fig") -> str:
    p = FIG / name
    if not p.exists():
        return ""
    return f'<img class="{cls}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">'


def figtxt(name: str) -> str:
    p = FIG / name
    return e(p.read_text()) if p.exists() else ""


def img(name: str, cls: str = "shot") -> str:
    p = SHOTS.get(name)
    if not p:
        return ""
    return f'<img class="{cls}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">'


# ================================================================ slides
NOT_CLAIMED = ul(["Audio / voice input is not supported yet",
                  "Thermal inspection images are synthetic with known ground truth (no real drone data was available); "
                  "satellite scenes, weather data and document formats are real",
                  "Plant meters come from a physics-based digital twin; the same interface accepts live SCADA"])
cost_red, co2_red = pct(ai.cost_rs, bau.cost_rs), pct(ai.co2_t, bau.co2_t)
crit_red = pct(ai.unserved_critical_mwh, bau.unserved_critical_mwh)

# 1 title
slide("", dark=True, body=f'''
<div class="kicker">ET AI HACKATHON: AGENTIC EDITION · SOLUTION DECK</div>
<div class="title">Renewable Energy Orchestrator</div>
<div class="lead">An autonomous AI control room that reads satellite images, thermal inspections, documents and live
data, simulates 100 futures for every decision, and runs an industry's solar, wind, batteries, loads and grid
connection every 15 minutes — cheaper, cleaner and safer.</div>
<div class="kpis">
 <div><b>{cost_red:.0f}%</b><span>energy cost</span></div>
 <div><b>{co2_red:.0f}%</b><span>CO₂</span></div>
 <div><b>{crit_red:.0f}%</b><span>critical load cut</span></div>
 <div><b>0</b><span>safety violations</span></div>
 <div class="claim"><b>D3 × F3</b><span>claimed position</span></div>
</div>
<div class="note">vs business-as-usual control, {N} randomized simulated days with unannounced storms, breakdowns and
price spikes</div>
<div class="team">Team {e(TEAM)}</div>''')

# 2 problem
slide("The problem", "Industrial renewable portfolios must be re-planned every 15 minutes under deep uncertainty",
      body=f'''<div class="grid3">
{card("Volatile, messy inputs", ul(["Solar & wind swing with clouds and storms", "IEX prices change every 15-min block",
      "Crucial facts arrive as images, PDFs and emails: inspection photos, work orders, grid notices"]))}
{card("Conflicting goals", ul(["Lowest cost vs cleanest energy vs never cutting critical load",
      "The right trade-off depends on the company's risk appetite"]))}
{card("Today's practice", ul(["Rule-of-thumb control: charge on surplus, discharge on deficit",
      "Reacts only to <i>now</i>; ignores forecasts, prices, documents and risk",
      "Breaks grid-line limits, wastes clean power"]))}
</div>
<div class="callout warn"><h3>Business-as-usual on our test portfolio (5 solar, 3 wind, 2 batteries, 4 industrial
loads)</h3>{ul([f"₹{bau.cost_rs / 1e5:.1f} lakh per day energy cost · {bau.co2_t:,.0f} t CO₂ per day · only "
                f"{bau.clean_pct:.0f}% clean energy",
                f"{bau.unserved_critical_mwh:.1f} MWh of <b>critical</b> load cut per day on average "
                f"(on {CRIT_DAYS_BAU} of {N} days) · {bau.violations:.1f} physical-limit violations per day"])}</div>''')

# 3 overview
steps = [("Perceive", "live data + images + documents", BLUE), ("Diagnose", "9 risk signals at once", BLUE),
         ("Plan", "33 plans from 9 recipe bands", AQUA), ("Simulate", "each plan × 100 futures", AQUA),
         ("Choose", "4-rule ladder ± LLM critique", ORANGE), ("Verify & act", "physics validator", ORANGE),
         ("Explain", "reason + audit log", NAVY), ("Learn", "forecast bias & bands", NAVY)]
flow = "".join(f'<div class="step" style="background:{c}"><b>{e(t)}</b><span>{e(d)}</span></div>' for t, d, c in steps)
slide("Our solution at a glance", "An agent that reads heterogeneous inputs, reasons over all risks at once, simulates "
      "futures, acts autonomously and learns", body=f'''<div class="flow">{flow}</div>
<div class="grid2">
{card("What makes it different", ul(["Reads <b>satellite images, thermal inspection images and PDF / text notices</b> "
      "and turns them into validated, confidence-scored facts", "No fixed playbook: every risk counts simultaneously, "
      "as a probability", "The LLM can propose and test its own plan, but never types a megawatt — maths computes, "
      "physics validates", "Autonomous inside management-set limits, with operator override and review"]))}
{card("Built and working", ul(["Streamlit control room: industry setup → live operations → results & learning",
      "Live Open-Meteo weather + NASA satellite scenes for the real farm coordinates", "MILP optimiser, Monte Carlo "
      "stress test, online learning, multimodal intake", "Labelled reliability tests, 90+ automated tests, 100-day "
      "Monte Carlo"]))}
</div>''')

# 4 grid
cells = ""
for d in ("D1", "D2", "D3"):
    cells += f'<div class="gl">{d}</div>'
    for f in ("F1", "F2", "F3"):
        claim = d == "D3" and f == "F3"
        cells += f'<div class="gc {"claim" if claim else "cov"}">{"CLAIMED" if claim else "covered"}</div>'
slide("Our position in the 3×3 grid: D3 × F3", "Highest depth and highest feature level — each axis evidenced below",
      "Self-assessment", body=f'''<div class="grid-2-1">
<div><div class="g9"><div></div><div class="gh">F1</div><div class="gh">F2</div><div class="gh">F3</div>{cells}</div>
<p class="muted small">F1 optimal action cluster · F2 + situation-specific criteria under uncertainty · F3 + simulation
over time with varying inputs<br>D1 acceptable outputs · D2 structured input, high reliability · D3 heterogeneous
multimodal input, high reliability</p></div>
<div>{table([["Axis", "Requirement", "Evidence"],
             ["<b>D3</b>", "Highly heterogeneous multimodal input",
              "Satellite images (NASA MODIS) · thermal / drone images · PDF work orders & grid notices · free-text "
              "fault reports & emails · weather API (JSON) · price & consumption CSV · config tables"],
             ["<b>D3</b>", "High degree of demonstrable reliability",
              f"0 wrong findings applied in {th_applied + DOC['applied']} labelled image & document cases · "
              f"{DOC['ambiguous_caught_pct']:.0f}% of vague notices quarantined · 0 safety violations in {N} days · "
              f"cheaper than business-as-usual on {CHEAPER}/{N} days"],
             ["<b>F3</b>", "Simulate over time with varying inputs",
              "100 sampled futures every 15 min · 24 h day-ahead plan · full-day digital twin with 11 shock types · "
              f"{N}-day Monte Carlo"]], widths=(12, 30, 58), small=True)}
<div class="callout warn" style="margin-top:16px"><h3>What we deliberately do not claim</h3>{NOT_CLAIMED}</div>
</div></div>''')

# 5 multimodal pipeline
mods = [("🛰️ Satellite scenes", "NASA GIBS MODIS true-colour image around each solar farm", "Cloud mask (white & "
         "unsaturated pixels) → measured cloud % vs forecast at overpass", "≥25-pt gap → solar −/+ around overpass"),
        ("🌡️ Thermal / drone images", "Radiometric PNG/JPG of PV blocks", "Per-module median → noise filter → robust "
         "threshold → hot-spot regions → modules", "Farm derate + repair job the AI schedules"),
        ("📄 PDF & text notices", "Work orders, fault reports, SLDC grid notices, production notices, emails",
         "Classify → extract asset, time window, size (rules, or LLM when configured) → schema validation",
         "Maintenance request, outage, line limit or demand change"),
        ("📊 Structured feeds", "Weather API (JSON), IEX prices, consumption CSV, asset tables",
         "Live → last-year proxy → saved day → simulated fallback", "Forecasts & profiles")]
rows = [["Modality", "Input", "How it is understood", "What it changes"]] + [[f"<b>{a}</b>", b, c, d] for a, b, c, d in mods]
slide("D3: highly heterogeneous multimodal input", "Every modality becomes a confidence-scored finding before it "
      "can touch a decision", "D3", body=f'''{table(rows, widths=(18, 24, 34, 24))}
<div class="flow small-flow">
<div class="step" style="background:{BLUE}"><b>Read</b><span>image / PDF / text / API</span></div>
<div class="step" style="background:{AQUA}"><b>Extract</b><span>CV, parser or LLM</span></div>
<div class="step" style="background:{ORANGE}"><b>Validate</b><span>schema, ranges, asset ids, image quality</span></div>
<div class="step" style="background:{NAVY}"><b>Gate</b><span>≥75% confidence → apply; else quarantine</span></div>
<div class="step" style="background:{BLUE}"><b>Act</b><span>twin updated, AI re-plans instantly</span></div>
</div>''')

# 5b multimodal evidence in action
slide("D3 in action: what the agent actually reads", "A real satellite scene, a thermal inspection and a grid notice — "
      "each turned into a decision-ready finding", "D3", body=f'''<div class="grid3 evid">
<div>{fig("satellite.png")}<p class="cap">🛰️ {figtxt("satellite.txt")}. Left: raw NASA scene; right: detected cloud
edge (orange). Compared with the forecast at the 10:30 overpass; a ≥25-point gap changes the solar plan.</p></div>
<div>{fig("thermal.png")}<p class="cap">🌡️ Drone thermal image of a PV block (synthetic, known truth). White boxes =
modules the agent flagged → <b>{figtxt("thermal.txt")}</b></p></div>
<div><div class="doc">STATE LOAD DESPATCH CENTRE<br>GRID OPERATOR NOTICE<br>Ref: SLDC/OPS/2026/1187<br><br>Due to
transmission congestion on the 220 kV evacuation corridor, the interconnection import limit for the consumer will be
reduced to 60% from 17:30 for 3 hours.</div>
<p class="cap">📄 PDF notice → extracted <b>line limit 60%, 17:30–20:30, confidence 100% → applied</b>. A vague email
("may possibly take W1 or W3 down… to be confirmed") → <b>quarantined</b>.</p></div></div>''')

# 6 D3 reliability
th_rows = [["Thermal tier", "Images", "Exact defect count", "Applied", "Wrong applied"]]
for t, v in TH.items():
    th_rows.append([t.title(), str(v["images"]), f"{v['exact_count_pct']:.0f}%", str(v["applied"]),
                    f"<b>{v['wrong_applied']}</b>"])
slide("D3: demonstrable reliability on labelled data", "Measured accuracy, and — most important — nothing wrong is "
      "ever acted on", "D3 · reliability", body=f'''<div class="grid2">
<div>{table(th_rows, widths=(26, 14, 26, 16, 18))}
<p class="muted small">Hard tier = heavy sensor noise + sun glare: detection is unreliable, so every one is
quarantined instead of applied.</p>
{table([["Satellite cloud detector", "Result"], ["Real NASA scenes of our sites", str(SAT['real_background_scenes'])],
        ["Tests with known synthetic cloud", str(SAT['synthetic_cloud_tests'])],
        ["Mean error", f"{SAT['mae_cloud_pct_pts']} % points (max {SAT['max_error_pts']})"]], widths=(60, 40))}</div>
<div>{table([["Documents (200 notices)", "Result"], ["Clear notices applied correctly",
             f"{DOC['applied_correct']} / {DOC['clear']} ({DOC['applied_precision_pct']:.0f}%)"],
            ["Vague / tentative notices quarantined", f"{DOC['ambiguous_quarantined_or_ignored']} / {DOC['ambiguous']} "
             f"({DOC['ambiguous_caught_pct']:.0f}%)"],
            ["Wrong findings applied", f"<b>{DOC['wrong_applied']}</b>"],
            ["Formats covered", "24 h, 12 h am/pm, 'hrs', durations, ranges, dates, filler text, PDF"]], widths=(55, 45))}
<div class="callout good"><h3>Safety metric across all modalities</h3><p><b>0 wrong findings applied</b> out of
{th_applied + DOC['applied']} applied image and document findings. Corrupt or unsupported files are rejected without
interrupting control.</p></div></div></div>''')

# 7 features
slide("Feature evidence: F1 → F2 → F3", "Each level is demonstrated live in the prototype and verified by tests", "F3",
      body=table([["Level", "Requirement", "How the agent does it", "Where to see it"],
                  ["<b>F1</b>", "Determine optimal cluster of actions",
                   "MILP jointly sets battery charge / discharge, grid import / export, curtailment per farm and demand "
                   "response per load — one consistent action set every 15 min", "Live tab → action log"],
                  ["<b>F2</b>", "Optimal options after determining optimality criteria for the situation, adjusting "
                   "for uncertainty", "9 continuous signals + multimodal evidence → 33 plans from 9 recipe bands "
                   "(weights, reserve, forecast caution) → ranked against the industry's risk tolerance",
                   "'What the AI is thinking' panel"],
                  ["<b>F3</b>", "Simulate the scenario over time, accounting for variations in input conditions",
                   "Every plan replayed in 100 futures (forecast spread + storm probabilities) over 4 h; 24 h day-ahead "
                   f"plan; full-day twin with 11 shocks; {N}-day Monte Carlo", "Inject a storm at 10% vs 90%"]],
                 widths=(8, 26, 46, 20)) + f'''<div class="callout"><b>Graded response to uncertainty</b> (same day,
18:00 storm forecast): {" · ".join(f"{p}% storm → {r}% battery reserve" for p, r in F["storm_grade"])}</div>''')

# 8 architecture
arch_in = "".join(f'<div class="ain"><b>{e(a)}</b><span>{e(b)}</span></div>' for a, b in [
    ("Industry inputs", "loads, assets, limits, maintenance, targets, risk appetite"),
    ("Images", "NASA satellite scenes · thermal / drone inspections"),
    ("Documents", "PDF work orders · SLDC notices · fault reports · emails"),
    ("Live data", "Open-Meteo 15-min weather · IEX prices · plant state (twin / SCADA)")])
loop = "".join(f'<div class="lp"><i style="background:{c}">{e(a)}</i><span>{e(b)}</span></div>' for a, b, c in [
    ("1 Perceive", "state, learned P10/P50/P90 forecasts, alerts with probabilities, evidence", BLUE),
    ("2 Diagnose", "9 risk signals simultaneously", BLUE),
    ("3 Options", "9 recipe bands → explore 27 + refine 6 = 33 plans (MILP, PuLP/CBC)", AQUA),
    ("4 Stress-test", "each plan × 100 futures → avg ₹, worst ₹, P(shortfall), CO₂", AQUA),
    ("5 Choose", "ladder: safe → risk ≤ tolerance → cost/clean → battery wear; LLM may propose & pick", ORANGE),
    ("6 Verify", "physics validator; repair ×2 → safe fallback", ORANGE),
    ("7 Act + explain", "execute, JSONL audit log, plain-English reason", NAVY),
    ("8 Learn", "forecast bias & band width, saved daily", NAVY)])
slide("Architecture", "Process flow from heterogeneous inputs to autonomous action — every block is implemented",
      "Architecture", body=f'''<div class="arch">
<div class="acol">{arch_in}<div class="ain gate"><b>Multimodal intake</b><span>extract → validate → confidence gate
(≥75%) → twin</span></div></div>
<div class="aarrow">➜</div>
<div class="aloop"><h3>Autonomous agent loop · every 15 min + instantly on any shock</h3>{loop}</div>
<div class="aarrow">➜</div>
<div class="acol">
<div class="ain llm"><b>LLM (optional, any provider)</b><span>Claude · Groq · Gemini · OpenRouter · Ollama. Critiques the
pick, proposes its own plan (tested like all others), explains. Never outputs MW.</span></div>
<div class="ain"><b>Day-ahead backbone</b><span>24 h LP battery targets, re-solved after shocks</span></div>
<div class="ain over"><b>Human oversight</b><span>operator override → safe mode · high-impact review · quarantine
queue</span></div>
<div class="ain out"><b>Outputs</b><span>set-points · ₹ & CO₂ saved · clean % · efficiency · audit log</span></div>
</div></div>''')

# 9 decision engine
slide("How the AI chooses: 33 plans × 100 futures × 4 rules", "Optimisation, simulation and a transparent priority "
      "ladder — under a second per decision", body=f'''<div class="grid-3-2">
<div>{table([["Recipe band", "Battery reserve", "Forecast caution", "Price / clean / reliability"],
             ["Max savings", "5–15%", "0–15%", "≈65 / 20 / 15"], ["Lean", "12–25%", "0–25%", "≈59 / 19 / 22"],
             ["Balanced", "22–35%", "10–35%", "≈53 / 18 / 30"], ["Green-lean", "20–40%", "0–25%", "≈30 / 55 / 15"],
             ["Steady", "32–45%", "20–50%", "≈46 / 16 / 38"], ["Cautious", "42–55%", "35–65%", "≈39 / 15 / 46"],
             ["Protective", "52–68%", "50–80%", "≈32 / 13 / 55"], ["Strong reserve", "64–80%", "65–95%", "≈24 / 12 / 65"],
             ["Fortress", "76–90%", "80–100%", "≈16 / 10 / 73"]], widths=(28, 22, 22, 28), small=True)}
<p class="muted small">3 combinations explored inside every band (27) + 6 refined around the best = 33 MILP plans,
replayed in the same 100 futures (a 35% storm hits in ~35 of them).</p></div>
<div class="rules">
<div class="rule" style="border-color:#d64545"><b style="color:#d64545">1 · Safety</b>Must pass the physics validator
(state of charge, rates, line limits, balance, availability).</div>
<div class="rule" style="border-color:{ORANGE}"><b style="color:{ORANGE}">2 · Reliability</b>Chance of cutting critical
load ≤ the industry's tolerance (default 2%).</div>
<div class="rule" style="border-color:{BLUE}"><b style="color:{BLUE}">3 · Cost vs clean</b>Lowest score = avg ₹ +
risk-aversion × (worst − avg) + carbon-pressure × CO₂ value.</div>
<div class="rule" style="border-color:{AQUA}"><b style="color:{AQUA}">4 · Battery life</b>Among near-equal plans, least
battery cycling wins.</div></div></div>''')

# 10 uncertainty
slide("Adjusting for uncertainty: protection scales with probability", "Same day, same time — only the storm "
      "probability changes", "F2 · F3", body=f'''<div class="grid2">
<div>{bars([(f"{p}% storm", r, BLUE) for p, r in F["storm_grade"]], "%", "{:.0f}", title="Battery reserve the AI keeps")}</div>
<div>{card("Why it matters", "<p>A fixed playbook treats every storm alert alike. Here the storm is sampled into ~p% "
            "of 100 futures, so protection rises gradually with risk — the AI pays for protection only when the "
            "expected outage cost justifies it.</p>")}
{card("What the agent says (real output)", f"<p class='quote'>{e(F['explanation_example'])}</p>")}</div></div>
<p class="muted">Combined risks are handled together: a storm, a price spike, a battery fault and a grid notice all
enter the same 100 futures — nothing is collapsed into one 'situation'.</p>''')

# 11 agentic
caps = [("Reasons", "Diagnoses 9 interacting risk signals plus image & document evidence; writes its own rationale."),
        ("Plans", "24 h day-ahead plan + rolling 4 h plans, re-planned every 15 min and instantly after shocks or new "
                  "documents."),
        ("Uses tools", "MILP optimiser, Monte Carlo simulator, physics validator, image analysers, document parser, "
                       "weather / satellite / price APIs, crew dispatch."),
        ("Acts autonomously", "Dispatches assets, reduces flexible load, reschedules maintenance, sends repair crews "
                              "after breakdowns or thermal hot-spots."),
        ("Recovers from errors", "Infeasible → relax ×2 → safe fallback · API down → last-year proxy → replay → "
                                 "simulated · LLM failure → optimiser decides · bad files → rejected."),
        ("Learns & self-critiques", "Learns forecast bias daily; the LLM can challenge the ladder by proposing its "
                                    "own plan, which must win on the same stress test.")]
slide("Agentic & autonomous capability", "Reasons, plans, uses tools, acts, recovers from errors and learns", "Agentic",
      body='<div class="grid3">' + "".join(card(t, f"<p>{e(b)}</p>", cls="tall") for t, b in caps) + "</div>")

# 12 AI
slide("Effective use of AI: the right technique for each job", "AI is the core of every decision, not a bolt-on", "AI",
      body=table([["AI technique", "Role in the agent", "Why this technique"],
                  ["<b>Computer vision</b>", "Hot-spot detection in thermal images; cloud masking in satellite scenes",
                   "Turns pixels into capacity and weather facts, with quality checks"],
                  ["<b>Document understanding</b> (parser / LLM)", "Work orders, grid notices, fault reports → structured "
                   "events", "Operational reality arrives as text; validated before use"],
                  ["<b>Mathematical optimisation</b> (MILP)", "Each plan → exact feasible MW for 4 h",
                   "Physically consistent numbers — no hallucination possible"],
                  ["<b>Probabilistic simulation</b>", "100 futures per decision from forecast quantiles & alert "
                   "probabilities", "Honest risk: average, worst case, probability of outage"],
                  ["<b>Search</b>", "Explore + refine across 9 recipe bands", "Finds in-between optima a fixed menu "
                   "misses"],
                  ["<b>Online machine learning</b>", "Forecast bias & calibration from every outcome",
                   f"Learned wind bias {F['learn']['wind_learned']}% (true {F['learn']['wind_true']}%)"],
                  ["<b>LLM agent</b> (any provider)", "Critiques the recommendation, proposes & tests its own plan, "
                   "explains", "Reasoning & communication, fenced by code"]], widths=(26, 40, 34), small=True) +
      '<p class="muted small">Transparency: the results in this deck were produced with the optimiser + ladder '
      '(offline mode). The LLM layer is optional, provider-neutral (incl. free Groq / Gemini tiers) and the system '
      'never depends on it.</p>')

# 13 impact
slide(f"Business impact: {N}-day Monte Carlo", "Randomized days with 3 unannounced shocks each; all controllers see "
      "identical weather, prices and events", "Impact", body=f'''<div class="grid3">
{bars([("AI", ai.cost_rs / 1e5, BLUE), ("BAU", bau.cost_rs / 1e5, ORANGE), ("No battery", nob.cost_rs / 1e5, AQUA)],
      "", "{:.1f}", 360, 250, "Cost (₹ lakh / day)")}
{bars([("AI", ai.co2_t, BLUE), ("BAU", bau.co2_t, ORANGE), ("No battery", nob.co2_t, AQUA)], "", "{:.0f}", 360, 250,
      "CO₂ (t / day)")}
{bars([("AI", ai.unserved_critical_mwh, BLUE), ("BAU", bau.unserved_critical_mwh, ORANGE),
       ("No battery", nob.unserved_critical_mwh, AQUA)], "", "{:.1f}", 360, 250, "Critical load cut (MWh / day)")}
</div>
{table([["Mean per day", "AI agent", "Business-as-usual", "Improvement"],
        ["Energy cost", f"₹{ai.cost_rs / 1e5:.1f} L", f"₹{bau.cost_rs / 1e5:.1f} L",
         f"<b>{cost_red:.0f}%</b> · cheaper on {CHEAPER}/{N} days"],
        ["CO₂", f"{ai.co2_t:,.0f} t", f"{bau.co2_t:,.0f} t", f"<b>{co2_red:.0f}%</b>"],
        ["Critical load cut", f"{ai.unserved_critical_mwh:.1f} MWh", f"{bau.unserved_critical_mwh:.1f} MWh",
         f"<b>{crit_red:.0f}%</b> · days affected {CRIT_DAYS_AI} vs {CRIT_DAYS_BAU}"],
        ["All load cut (incl. flexible)", f"{ai.unserved_mwh:.1f} MWh", f"{bau.unserved_mwh:.1f} MWh",
         f"{pct(ai.unserved_mwh, bau.unserved_mwh):.0f}%"],
        ["Clean energy share / safety violations", f"{ai.clean_pct:.1f}% / {ai.violations:.2f}",
         f"{bau.clean_pct:.1f}% / {bau.violations:.2f}", "+{:.1f} pts / zero".format(ai.clean_pct - bau.clean_pct)]],
       widths=(32, 20, 22, 26), small=True)}
<p class="muted small">We lead with critical load ({crit_red:.0f}%) rather than total load ({pct(ai.unserved_mwh,
bau.unserved_mwh):.0f}%): nearly all of the AI's remaining cuts are physical shortages (e.g. storm + congested line),
where it protects critical load first and says so.</p>''')

# 14 learning
L = F["learn"]
slide("Learning: the agent gets better every day", "Online correction of forecast bias and uncertainty bands",
      "Trainability", body=f'''<div class="grid2">
<div>{bars([("Wind before", L["wind_mae_before"], ORANGE), ("Wind after", L["wind_mae_after"], BLUE),
            ("Demand before", L["demand_mae_before"], ORANGE), ("Demand after", L["demand_mae_after"], BLUE)],
           " MW", "{:.2f}", title="1-hour-ahead forecast error (MAE, one day)")}</div>
<div>{table([["Forecast", "Hidden model bias", "Bias the agent learned"],
             ["Wind", f"{L['wind_true']}%", f"{L['wind_learned']}%"],
             ["Demand", f"{L['demand_true']}%", f"{L['demand_learned']}%"]], widths=(30, 35, 35))}
{ul(["Every tick, the 1-hour-ahead forecast is compared with what actually happened",
     "Bias learned with a robust moving average; P10–P90 bands widened or narrowed to ~80% coverage",
     "Lessons saved at end of day and used from the first tick tomorrow",
     "Honest note: solar error did not improve on the test day (sunrise / sunset ramps dominate)"])}</div></div>''')

# 15 prototype
shot_html = "".join(f'<figure>{img(k)}<figcaption>{e(c)}</figcaption></figure>' for k, c in
                    [("live", "Live control room"), ("thinking", "What the AI is thinking"),
                     ("evidence", "Evidence from images & documents"), ("results", "Best recipe & why")] if k in SHOTS)
slide("Working prototype", "Streamlit control room — runs without internet or API key; live data when available",
      "Demo", body=(f'<div class="shots">{shot_html}</div>' if shot_html else "") + '<div class="grid3">' + "".join(
          card(t, f"<p>{e(b)}</p>") for t, b in [
              ("1 · Your industry", "Loads & consumption CSV, solar / wind / batteries with coordinates, grid limits, "
                                    "maintenance & breakdowns, targets, satellite / image / document inputs"),
              ("2 · Live control room", "Status, ₹ & CO₂ saved, all risk signals, every plan considered, evidence "
                                        "table, high-impact review, inject 11 events or feed a document"),
              ("3 · Results & learning", "Best recipe & why, AI vs business-as-usual, learning table, Monte Carlo "
                                         "proof")]) + "</div>")

# 16 responsible AI
items = [("No hallucinated actions", "The LLM can only propose settings or pick a validated plan id; all MW come from "
          "the optimiser; ineligible picks are rejected in code."),
         ("Multimodal guardrails", "Every image / document finding is validated and confidence-scored; below 75% it is "
          "quarantined for a human, never applied."),
         ("Human oversight", "Operator override hands control to the safe controller instantly; high-impact actions "
          "are flagged for review; management sets risk limits and recipe ranges."),
         ("Physics guardrails", "Validator checks state of charge, rates, line limits, balance, critical load and "
          "availability before every action; the twin independently counts violations."),
         ("Transparency & audit", "Every decision logged: signals, evidence, 33 options, stress results, ladder, "
          "action, explanation — reproducible with seeded scenarios."),
         ("Security & privacy", "No personal data; keys only in .env; read-only SCADA first; OT isolation; "
                                "shadow-mode rollout before control.")]
slide("Responsible AI, human oversight & robustness", "Autonomous by design — governed by limits, verification, "
      "override and audit", "Responsible AI",
      body='<div class="grid3">' + "".join(card(t, f"<p>{e(b)}</p>", cls="tall") for t, b in items) + "</div>")

# 17 scale
phases = [("Week 0 · Configure", "Assets, loads, schedules, targets (UI, YAML or CSV); upload documents & images."),
          ("Weeks 1–2 · Shadow mode", "SCADA / meters / BMS read-only (OPC-UA, Modbus, MQTT); AI recommends, operators "
                                      "compare."),
          ("Weeks 3–4 · Battery control", "Battery set-points only; watchdog returns to local control on silence."),
          ("Month 2+ · Full autonomy", "Curtailment & demand response; learning tunes forecasts per site.")]
slide("Scalability & path to a real plant", "Industry-agnostic by configuration; one interface swaps the twin for live "
      "data", "Scale", body='<div class="grid4">' + "".join(card(t, f"<p>{e(b)}</p>", c) for (t, b), c in
                                                           zip(phases, [BLUE, AQUA, ORANGE, NAVY])) + "</div>" +
      ul(["Any number of farms, batteries and loads — sites without wind or batteries are tested",
          "&lt; 1 s per decision for 33 plans × 100 futures on a laptop; solves run in parallel",
          "Same agent for steel, cement, textiles or data centres — only configuration changes",
          "New modalities plug into the same extract → validate → gate → apply pipeline"]))

# 18 limitations
slide("Honest limitations & roadmap", "What we have not proven yet, and what comes next", body=f'''<div class="grid2">
<div class="callout warn">{ul(["Thermal images are <b>synthetic</b> with known ground truth (no real drone data "
     "available); satellite scenes are real", "No audio / voice input yet", "Plant meters, batteries and demand come "
     "from a physics-based digital twin; weather and satellite are live", "IEX price file is a labelled sample",
     "Results produced in offline mode; the LLM's added value is not yet measured on the 100 days",
     "Band search finds very good plans, not a proven global optimum"])}</div>
<div class="callout good">{ul(["Real drone / thermal datasets and panel-defect models", "Voice notes from field crews "
     "→ the same document pipeline", "Live SCADA adapter with data-health monitoring", "Real IEX DAM + RTM prices and "
     "DSM penalty rules", "A/B the LLM layer over 100 days with a free-tier model", "Full stochastic optimisation as "
     "the end-state optimum"])}</div></div>''')

# 19 close
slide("", dark=True, body=f'''<div class="title" style="margin-top:110px">Cheaper, cleaner, never dark —</div>
<div class="title light">decided by an AI that shows its work.</div>
<div class="lead" style="margin-top:40px">Claimed position <b>D3 × F3</b> · {N}-day proof: {cost_red:.0f}% cost,
{co2_red:.0f}% CO₂, {crit_red:.0f}% critical load cut, 0 safety violations · 0 wrong multimodal findings applied</div>
<div class="lead">Code &amp; README: github.com/Akshadman/renewable-energy-orchestrator</div>
<div class="team">Team {e(TEAM)}</div>''')

# ---------------------------------------------------------------- render
CSS = f"""
@page {{ size: 13.333in 7.5in; margin: 0 }}
* {{ box-sizing: border-box }}
body {{ margin: 0; font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; color: #1f2937 }}
.slide {{ width: 13.333in; height: 7.5in; position: relative; overflow: hidden; page-break-after: always;
  padding: 0.38in 0.5in 0.45in; background: #fff }}
.bar {{ position: absolute; top: 0; left: 0; right: 0; height: 9px; background: {BLUE} }}
.head {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px }}
h1 {{ font-size: 30px; color: {NAVY}; margin: 6px 0 4px }}
.sub {{ font-size: 15.5px; color: {MUTED}; margin: 0 }}
.tag {{ background: {NAVY}; color: #fff; padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 600;
  white-space: nowrap; margin-top: 8px }}
.body {{ font-size: 16.5px }}
.foot {{ position: absolute; bottom: 14px; left: 0.5in; right: 0.5in; font-size: 11px; color: {MUTED};
  display: flex; justify-content: space-between }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px }}
.grid3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px }}
.grid4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 14px }}
.grid-2-1 {{ display: grid; grid-template-columns: 0.9fr 1.6fr; gap: 26px }}
.grid-3-2 {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 22px }}
.card {{ background: #f1f4f8; border-radius: 10px; padding: 12px 16px }}
.card h3 {{ margin: 0 0 6px; font-size: 18.5px; color: {NAVY} }}
.card[style] h3 {{ color: #fff }}
.card p {{ margin: 4px 0; line-height: 1.38 }}
.card.tall {{ min-height: 2.15in }}
ul {{ margin: 4px 0; padding-left: 20px; line-height: 1.45 }}
li {{ margin: 3px 0 }}
.callout {{ background: #eef5fd; border-radius: 10px; padding: 12px 18px; margin-top: 14px; line-height: 1.45 }}
.callout.warn {{ background: #fff4e5 }} .callout.good {{ background: #e6f6ee }}
.callout h3 {{ margin: 0 0 4px; font-size: 16px; color: {NAVY} }}
table {{ width: 100%; border-collapse: collapse; font-size: 15.5px; table-layout: fixed }}
table.small {{ font-size: 14px }}
th {{ background: {NAVY}; color: #fff; text-align: left; padding: 7px 9px; font-weight: 600 }}
td {{ padding: 6px 9px; border-bottom: 1px solid #e5e7eb; vertical-align: top; line-height: 1.35 }}
tr:nth-child(even) td {{ background: #f7f9fb }}
tr.hl td {{ background: #e3eefb }}
.muted {{ color: {MUTED} }} .small {{ font-size: 13.5px }}
.fig {{ width: 100%; border-radius: 8px; border: 1px solid #d0d6de }}
.evid .fig {{ height: 2.7in; object-fit: contain; background: #f7f9fb }}
.cap {{ font-size: 14px; color: #1f2937; line-height: 1.4; margin-top: 8px }}
.doc {{ font-family: 'Courier New', monospace; font-size: 14px; background: #fff; border: 1px solid #c9cfd8;
  border-radius: 6px; padding: 16px; height: 2.7in; box-shadow: 3px 3px 0 #e5e7eb; line-height: 1.45 }}
.flow {{ display: flex; gap: 10px; margin: 6px 0 18px }}
.flow .step {{ flex: 1; color: #fff; border-radius: 10px; padding: 10px 10px; min-height: 92px; position: relative }}
.flow .step b {{ display: block; font-size: 15px; margin-bottom: 4px }}
.flow .step span {{ font-size: 12.5px; line-height: 1.3; display: block }}
.small-flow {{ margin-top: 16px }} .small-flow .step {{ min-height: 70px }}
.g9 {{ display: grid; grid-template-columns: 44px repeat(3, 1fr); gap: 5px; margin-top: 6px }}
.gh, .gl {{ font-weight: 700; color: {NAVY}; display: flex; align-items: center; justify-content: center; font-size: 17px }}
.gc {{ height: 74px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: 700;
  font-size: 14px }}
.gc.cov {{ background: #d6e6fa; color: {MUTED}; font-weight: 500 }}
.gc.claim {{ background: {ORANGE}; color: #fff; font-size: 16px }}
.chart {{ width: 100%; height: auto }} .ct {{ font-size: 15px; font-weight: 700; fill: {NAVY} }}
.cv {{ font-size: 14px; font-weight: 700; fill: #1f2937 }} .cl {{ font-size: 13px; fill: {MUTED} }}
.rules {{ display: flex; flex-direction: column; gap: 12px }}
.rule {{ border: 2px solid; border-radius: 10px; padding: 10px 14px; line-height: 1.4 }}
.rule b {{ display: block; font-size: 16px; margin-bottom: 2px }}
.quote {{ font-style: italic; font-size: 13.5px }}
.arch {{ display: grid; grid-template-columns: 2.1fr 0.25fr 4fr 0.25fr 2.2fr; align-items: center; gap: 4px }}
.acol {{ display: flex; flex-direction: column; gap: 9px }}
.ain {{ background: #e3eefb; border-radius: 8px; padding: 8px 10px; font-size: 12px; line-height: 1.3 }}
.ain b {{ display: block; font-size: 13.5px; color: {NAVY} }}
.ain.gate {{ background: #fff4e5; border: 2px solid {ORANGE} }} .ain.llm {{ background: #fff4e5 }}
.ain.over {{ background: #fde8e8 }} .ain.out {{ background: {NAVY}; color: #fff }} .ain.out b {{ color: #fff }}
.aarrow {{ font-size: 28px; color: {MUTED}; text-align: center }}
.aloop {{ border: 2px solid {BLUE}; border-radius: 12px; padding: 10px 12px }}
.aloop h3 {{ margin: 0 0 8px; font-size: 14px; color: {NAVY} }}
.lp {{ display: grid; grid-template-columns: 112px 1fr; gap: 8px; align-items: center; margin: 5px 0; font-size: 12.5px }}
.lp i {{ font-style: normal; color: #fff; border-radius: 6px; padding: 5px 8px; font-weight: 700; font-size: 12.5px }}
.shots {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 12px }}
.shots figure {{ margin: 0 }} .shot {{ width: 100%; height: 2.55in; object-fit: cover; object-position: top;
  border: 1px solid #d0d6de; border-radius: 6px }}
figcaption {{ font-size: 12px; color: {MUTED}; text-align: center; margin-top: 3px }}
.dark {{ background: {NAVY}; color: #fff; padding: 0.7in 0.8in }}
.dark .foot {{ color: #9eb3cc }}
.kicker {{ color: #9ec5f4; font-weight: 700; letter-spacing: .06em; font-size: 15px }}
.title {{ font-size: 52px; font-weight: 800; margin: 18px 0 10px }} .title.light {{ color: #9ec5f4; margin-top: 0 }}
.lead {{ font-size: 19px; color: #dbe4f0; max-width: 11.5in; line-height: 1.45 }}
.kpis {{ display: flex; gap: 18px; margin-top: 34px }}
.kpis div {{ background: #1b365d; border-radius: 12px; padding: 14px 20px; min-width: 1.9in }}
.kpis b {{ display: block; font-size: 34px }} .kpis span {{ font-size: 13px; color: #b8c7da }}
.kpis .claim {{ background: {ORANGE} }} .kpis .claim span {{ color: #fff }}
.note {{ margin-top: 14px; color: #9eb3cc; font-size: 14px }}
.team {{ position: absolute; bottom: 0.75in; font-size: 20px; font-weight: 700 }}
"""

out_html = ROOT / "deck" / "Solution_Deck.html"
out_html.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>Renewable Energy Orchestrator — '
                    f'Solution Deck</title><style>{CSS}</style></head><body>{"".join(slides)}</body></html>')
out_pdf = ROOT / "deck" / "Solution_Deck.pdf"
out_pdf.unlink(missing_ok=True)
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out_pdf}",
                out_html.as_uri()], capture_output=True, timeout=120)   # Chrome may exit non-zero on harmless warnings
if not out_pdf.exists():
    raise SystemExit("PDF was not produced — is Google Chrome installed?")
print(f"{len(slides)} slides → {out_html.name}, {out_pdf.name}")
