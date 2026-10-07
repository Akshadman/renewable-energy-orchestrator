"""Streamlit dashboard for the Renewable Energy Orchestrator.  Run:  streamlit run app.py"""
from __future__ import annotations

import os
import time
from datetime import date as Date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # optional dependency
    pass

from agent.agent import OrchestratorAgent
from core.config import load_config
from core.data_sources import SOURCES, load_weather_day, today_ist
from core.events import EVENTS, make_event
from core.simulator import Simulator
from eval.evaluate import CONTROLLERS, METRICS, OUT_DIR, run_monte_carlo, save_outputs, summarize

st.set_page_config(page_title="Renewable Energy Orchestrator", page_icon="⚡", layout="wide")

# Reference categorical palette — fixed slot order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#52514e"
SOURCE_LABEL = {"openmeteo": "Open-Meteo live (15-min)", "replay": "Replay (cached CSV)", "synthetic": "Synthetic"}


# ---------------------------------------------------------------- session
def build(source: str, day: Date, seed: int, use_llm: bool, auto_approve: bool) -> None:
    """(Re)create the digital twin and agent."""
    cfg = load_config()
    with st.spinner(f"Loading weather from {SOURCE_LABEL[source]}…"):
        weather = load_weather_day(source, cfg, day, seed)
    sim = Simulator(cfg, seed=seed, weather=weather, day=day)
    st.session_state.update(cfg=cfg, sim=sim, playing=False, requested_source=source,
                            agent=OrchestratorAgent(sim, cfg, use_llm=use_llm, auto_approve=auto_approve))


def sidebar() -> None:
    sb = st.sidebar
    sb.title("⚡ Orchestrator")
    sb.subheader("Data source")
    source = sb.selectbox("Weather data", SOURCES, format_func=SOURCE_LABEL.get, key="source",
                          help="Falls back Open-Meteo → Replay → Synthetic automatically if a source fails.")
    day = sb.date_input("Day", value=today_ist(), key="day")
    seed = int(sb.number_input("Seed", 0, 10_000, 42, key="seed"))
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    use_llm = sb.toggle("Use LLM reasoner", value=has_key, disabled=not has_key, key="use_llm",
                        help="Needs ANTHROPIC_API_KEY in .env; otherwise the rule-based mock reasoner runs.")
    auto = sb.toggle("Auto-approve high-impact actions", value=True, key="auto_approve")
    if sb.button("Reset / apply", type="primary", width="stretch") or "sim" not in st.session_state:
        build(source, day, seed, use_llm, auto)
    st.session_state.agent.auto_approve = auto

    sb.subheader("Simulation")
    c1, c2 = sb.columns(2)
    playing = st.session_state.playing
    if c1.button("⏸ Pause" if playing else "▶ Play", width="stretch"):
        st.session_state.playing = not playing
        st.rerun()
    if c2.button("⏭ Step", width="stretch", disabled=st.session_state.sim.done):
        st.session_state.agent.step()
    sb.slider("Speed (ticks per refresh)", 1, 12, 2, key="speed")
    sb.slider("Refresh delay (s)", 0.0, 3.0, 0.5, 0.1, key="delay")
    sb.caption(f"Reasoner: **{st.session_state.agent.mode}**" +
               ("" if has_key else " — no API key, running offline"))


# ---------------------------------------------------------------- live view
def source_banner() -> None:
    sim = st.session_state.sim
    w = sim.weather
    requested = st.session_state.requested_source
    text = f"**Active data source:** {SOURCE_LABEL[w.source]} · weather date {w.date} · prices: {sim.profile.price_source}"
    if w.source != requested:
        st.warning(text + "  \n" + " · ".join(w.notes), icon="⚠️")
    else:
        st.info(text + (("  \n" + " · ".join(w.notes)) if w.notes else ""), icon="🛰️")


def kpi_cards() -> None:
    t = st.session_state.sim.totals()
    cols = st.columns(6)
    cols[0].metric("Cost", f"₹{t['cost_rs'] / 1e5:,.2f} L", border=True)
    cols[1].metric("Clean energy", f"{t['clean_pct']:.1f}%", border=True)
    cols[2].metric("Curtailed", f"{t['curtailed_mwh']:.1f} MWh", border=True)
    cols[3].metric("Unserved load", f"{t['unserved_mwh']:.1f} MWh", border=True,
                   help=f"of which critical: {t['unserved_critical_mwh']:.2f} MWh")
    cols[4].metric("Violations", f"{t['violations']:d}", border=True)
    cols[5].metric("CO₂", f"{t['co2_t']:.1f} t", border=True)


def _fig(title: str, y_title: str, height: int = 280) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=height, hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2),
                      yaxis_title=y_title, xaxis=dict(title="Hour (IST)", range=[0, 24], dtick=3))
    return fig


def charts() -> None:
    h = pd.DataFrame(st.session_state.sim.history)
    if h.empty:
        st.caption("Press ▶ Play or ⏭ Step to start the day.")
        return
    x = h.hour
    left, right = st.columns(2)
    gen = _fig("Generation by source vs demand", "MW")
    for i, (col, name) in enumerate((("solar_mw", "Solar"), ("wind_mw", "Wind"),
                                     ("discharge_mw", "Battery discharge"), ("import_mw", "Grid import"))):
        gen.add_trace(go.Scatter(x=x, y=h[col], name=name, stackgroup="gen", line=dict(width=0.5, color=SERIES[i]),
                                 hovertemplate="%{y:.1f} MW"))
    gen.add_trace(go.Scatter(x=x, y=h.demand_mw, name="Demand", line=dict(color=INK, width=2),
                             hovertemplate="%{y:.1f} MW"))
    left.plotly_chart(gen, key="gen")

    soc = _fig("Battery state of charge", "%")
    for i, bat in enumerate(st.session_state.cfg.batteries):
        soc.add_trace(go.Scatter(x=x, y=h[f"soc_{bat.id}"], name=bat.id, line=dict(width=2, color=SERIES[i]),
                                 hovertemplate="%{y:.1f}%"))
    soc.add_hrect(y0=0, y1=10, fillcolor=INK, opacity=0.08, line_width=0)
    soc.update_yaxes(range=[0, 100])
    right.plotly_chart(soc, key="soc")

    price = _fig("Market price", "₹/MWh")
    price.add_trace(go.Scatter(x=x, y=h.price_buy, name="Buy", line=dict(width=2, color=SERIES[0]),
                               hovertemplate="₹%{y:,.0f}"))
    price.add_trace(go.Scatter(x=x, y=h.price_sell, name="Sell", line=dict(width=2, color=SERIES[1]),
                               hovertemplate="₹%{y:,.0f}"))
    left.plotly_chart(price, key="price")

    grid = _fig("Grid interconnection (+ import / − export)", "MW")
    grid.add_trace(go.Bar(x=x, y=h.import_mw - h.export_mw, name="Net import", marker_color=SERIES[0],
                          hovertemplate="%{y:.1f} MW"))
    grid.add_trace(go.Scatter(x=x, y=h.import_limit_mw, name="Import limit", line=dict(color=INK, dash="dot", width=1.5)))
    grid.add_trace(go.Scatter(x=x, y=-h.export_limit_mw, name="Export limit", line=dict(color=INK, dash="dash", width=1.5)))
    right.plotly_chart(grid, key="grid")


def agent_panel() -> None:
    agent = st.session_state.agent
    st.subheader("🧠 Agent")
    if not agent.decisions:
        st.caption("No decisions yet.")
        return
    d = agent.decisions[-1]
    hh, mm = int(d.hour), int(round(d.hour % 1 * 60))
    c = d.criteria
    st.markdown(f"**Tick {d.tick} · {hh:02d}:{mm:02d}** — situation: `{d.situation}` · reasoner: `{d.reasoner}` "
                f"· plan: `{d.plan_source}`")
    cols = st.columns(5)
    cols[0].metric("w_cost", f"{c['w_cost']:.2f}")
    cols[1].metric("w_clean", f"{c['w_clean']:.2f}")
    cols[2].metric("w_reliability", f"{c['w_reliability']:.2f}")
    cols[3].metric("Reserve", f"{c['reserve_pct']:.0%}")
    cols[4].metric("Forecast", c["uncertainty_mode"].upper())
    st.success(d.explanation)
    for n in d.notes:
        st.caption(f"• {n}")
    if d.llm_error:
        st.caption(f"LLM note: {d.llm_error}")
    alerts = agent.current_alerts
    if alerts:
        with st.expander(f"Alerts ({len(alerts)})", expanded=True):
            for a in alerts:
                icon = {"critical": "🔴", "warning": "🟠"}.get(a.severity, "🔵")
                st.markdown(f"{icon} **{a.kind}** — {a.message}")


def event_buttons() -> None:
    st.subheader("💥 Inject event")
    sim = st.session_state.sim
    cols = st.columns(5)
    for i, kind in enumerate(EVENTS):
        if cols[i % 5].button(kind.replace("_", " "), key=f"ev_{kind}", width="stretch", disabled=sim.done):
            ev = sim.apply_event(make_event(kind, sim.tick))
            st.toast(ev.description, icon="💥")


def approvals() -> None:
    agent = st.session_state.agent
    st.subheader("🙋 Human approval queue")
    pending = agent.pending_approvals()
    if not pending:
        st.caption("Nothing waiting." + (" Auto-approve is on." if agent.auto_approve else ""))
    for r in pending:
        c1, c2, c3 = st.columns([6, 1, 1])
        c1.markdown(f"**{r.id}** · tick {r.tick} · `{r.kind}` — {r.reason}")
        if c2.button("Approve", key=f"ok_{r.id}", type="primary"):
            agent.resolve_approval(r.id, True)
            st.rerun()
        if c3.button("Reject", key=f"no_{r.id}"):
            agent.resolve_approval(r.id, False)
            st.rerun()
    done = [r for r in agent.approvals if r.status != "pending"][-5:]
    if done:
        st.caption("Recent: " + " · ".join(f"{r.id} {r.kind} → {r.status}" for r in reversed(done)))


def action_log() -> None:
    decisions = st.session_state.agent.decisions
    st.subheader("📜 Action log")
    if not decisions:
        return
    rows = [{"tick": d.tick, "time": f"{int(d.hour):02d}:{int(round(d.hour % 1 * 60)):02d}", "situation": d.situation,
             "reasoner": d.reasoner, "plan": d.plan_source, "valid": d.validation["passed"],
             **{k: v for k, v in d.action_summary.items() if k != "unserved_critical_mw"},
             "cost ₹": round(d.kpi["cost_rs"]), "violations": d.kpi["violations"]} for d in reversed(decisions[-96:])]
    st.dataframe(pd.DataFrame(rows), hide_index=True, height=300)
    st.caption("Full audit trail (criteria, attempts, validation, tool calls): logs/decisions.jsonl")


# ---------------------------------------------------------------- results
def results_tab() -> None:
    st.subheader("Monte Carlo: agent vs baselines")
    st.caption("Each run is one randomized synthetic day with 3 unannounced shocks. All three controllers see "
               "identical weather, demand, prices and events. The agent uses the mock reasoner.")
    c1, c2 = st.columns([1, 3])
    n = c1.number_input("Days", 5, 500, 20, step=5)
    if c1.button("Run Monte Carlo", type="primary"):
        bar = c2.progress(0.0, text="Simulating…")
        df = run_monte_carlo(int(n), progress=lambda f: bar.progress(f, text=f"Simulating… {f:.0%}"))
        save_outputs(df)
        bar.empty()
    path = OUT_DIR / "results.csv"
    if not path.exists():
        st.info("No results yet. Click **Run Monte Carlo**, or run `python -m eval.evaluate --n 100`.")
        return
    df = pd.read_csv(path)
    st.caption(f"{df.seed.nunique()} days loaded from {path.relative_to(OUT_DIR.parent.parent)}")
    summary = summarize(df)["mean"]
    agent_row, base_row = summary.loc["agent"], summary.loc["baseline"]
    cols = st.columns(4)
    cols[0].metric("Cost vs baseline", f"{(agent_row.cost_rs / base_row.cost_rs - 1) * 100:+.1f}%",
                   delta_color="inverse", border=True)
    cols[1].metric("Clean energy", f"{agent_row.clean_pct:.1f}%", f"{agent_row.clean_pct - base_row.clean_pct:+.1f} pts",
                   border=True)
    cols[2].metric("Unserved load / day", f"{agent_row.unserved_mwh:.1f} MWh",
                   f"{agent_row.unserved_mwh - base_row.unserved_mwh:+.1f} MWh", delta_color="inverse", border=True)
    cols[3].metric("Violations / day", f"{agent_row.violations:.2f}",
                   f"{agent_row.violations - base_row.violations:+.1f}", delta_color="inverse", border=True)
    metric = st.radio("Metric", list(METRICS), format_func=METRICS.get, horizontal=True)
    fig = go.Figure()
    for i, c in enumerate(CONTROLLERS):
        fig.add_trace(go.Box(y=df[df.controller == c][metric], name=c, marker_color=SERIES[i], boxpoints="all",
                             jitter=0.4, pointpos=0, marker=dict(size=4, opacity=0.5), line=dict(width=1.5)))
    fig.update_layout(title=f"{METRICS[metric]} per day — distribution over {df.seed.nunique()} days", height=420,
                      showlegend=False, margin=dict(l=10, r=10, t=40, b=10), yaxis_title=METRICS[metric])
    st.plotly_chart(fig, key="mc")
    st.dataframe(summarize(df).round(2))


# ---------------------------------------------------------------- main
def main() -> None:
    sidebar()
    live, results = st.tabs(["Live control room", "Results"])
    with live:
        source_banner()
        kpi_cards()
        charts()
        a, b = st.columns([3, 2])
        with a:
            agent_panel()
        with b:
            event_buttons()
            approvals()
        action_log()
    with results:
        results_tab()
    sim = st.session_state.sim
    if st.session_state.playing and not sim.done:
        for _ in range(st.session_state.speed):
            if sim.done:
                break
            st.session_state.agent.step()
        time.sleep(st.session_state.delay)
        st.rerun()
    elif sim.done and st.session_state.playing:
        st.session_state.playing = False
        st.toast("Day complete ✅")


main()
