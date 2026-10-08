"""Streamlit control room for the Renewable Energy Orchestrator.  Run:  streamlit run app.py"""
from __future__ import annotations

import copy
import os
import time
from datetime import date as Date
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # optional dependency
    pass

from agent.runner import Operation
from core.config import config_from_dict, config_to_dict, list_profiles, load_config, save_profile, validate_config
from core.data_sources import SOURCE_LABELS, SOURCES, load_weather_day, today_ist
from core.events import make_event
from core.simulator import Simulator, clock
from eval.evaluate import CONTROLLERS, METRICS, OUT_DIR, run_monte_carlo, save_outputs, summarize

st.set_page_config(page_title="Renewable Energy Orchestrator", page_icon="⚡", layout="wide")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical, fixed order
INK = "#52514e"
STATUS = {"ok": "#2e9e5b", "info": "#2a78d6", "warning": "#d98a00", "critical": "#d64545"}
ICON = {"ok": "🟢", "info": "🔵", "warning": "🟠", "critical": "🔴"}

EVENT_UI = {
    "storm_alert": "🌩️ Storm forecast", "cloud_cover": "☁️ Cloud cover", "wind_drop": "🍃 Wind drop",
    "wind_surge": "💨 Wind surge", "price_spike": "💸 Price spike", "demand_surge": "🏭 Demand surge",
    "line_congestion": "🔌 Grid line congestion", "generator_breakdown": "🛠️ Generator breakdown",
    "battery_unavailable": "🔋 Battery fault", "maintenance_window": "🧰 Maintenance request",
    "forecast_update": "📡 Forecast revision",
}


# ================================================================ session
def ss() -> Any:
    return st.session_state


def _has_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def start_day(raw: Dict[str, Any], source: str, day: Date, seed: int, use_llm: bool) -> None:
    """Build the AI twin, the baseline twin and the agent for one operating day."""
    cfg = config_from_dict(raw)
    with st.spinner(f"Loading weather ({SOURCE_LABELS[source]})…"):
        weather = load_weather_day(source, cfg, day, seed)
    sim = Simulator(cfg, seed=seed, weather=weather, day=day)
    ss().op = Operation(cfg, sim, use_llm=use_llm, persist_learning=True)
    ss().update(raw=raw, playing=False, requested_source=source)


def init() -> None:
    if "raw" not in ss():
        ss().raw = config_to_dict(load_config())
        ss().uploaded = {}
        ss().run_opts = {"source": "openmeteo", "day": today_ist(), "seed": 42, "use_llm": _has_key()}
    if "op" not in ss():
        o = ss().run_opts
        start_day(ss().raw, o["source"], o["day"], o["seed"], o["use_llm"])


# ================================================================ sidebar
def sidebar() -> None:
    op, sb = ss().op, st.sidebar
    sb.title("⚡ Orchestrator")
    sb.caption(f"**{op.cfg.name}**")
    t = op.sim.tick
    sb.markdown(f"### 🕒 {clock(min(t, 95))}{' · day complete' if op.done else ''}")
    sb.progress(min(t, 96) / 96)
    c1, c2 = sb.columns(2)
    if c1.button("⏸ Pause" if ss().playing else "▶ Play", width="stretch", type="primary", disabled=op.done):
        ss().playing = not ss().playing
        st.rerun()
    if c2.button("⏭ Step", width="stretch", disabled=op.done):
        op.step()
    sb.select_slider("Speed", ["slow", "normal", "fast", "very fast"], value="normal", key="speed")
    if sb.button("↺ Restart day", width="stretch"):
        o = ss().run_opts
        start_day(ss().raw, o["source"], o["day"], o["seed"], o["use_llm"])
        st.rerun()
    w = op.sim.weather
    sb.divider()
    sb.markdown(f"**Weather:** {SOURCE_LABELS[w.source]}  \n**Date:** {w.date}  \n"
                f"**Prices:** {op.sim.profile.price_source}")
    if w.source != ss().requested_source:
        sb.warning("Fell back: " + " · ".join(w.notes))
    sb.markdown(f"**AI reasoning:** {'Claude + optimizer' if op.agent.mode == 'llm' else 'Optimizer + rules (offline)'}")
    sb.caption("Fully autonomous — no human approval needed. Every action is logged.")


# ================================================================ tab 1 — industry setup
def setup_tab() -> None:
    raw = ss().raw
    st.markdown("Describe **your** site. The AI adapts to whatever you enter here. "
                "Press **Apply & start day** at the bottom when you're done.")
    profiles = list_profiles()
    c1, c2 = st.columns([3, 1])
    pick = c1.selectbox("Start from a saved profile", list(profiles), index=None, placeholder="Choose a profile…")
    if c2.button("Load profile", disabled=pick is None, width="stretch"):
        ss().raw, ss().uploaded = config_to_dict(load_config(profiles[pick])), {}
        _clear_editors()
        st.rerun()

    with st.form("setup"):
        name = st.text_input("Site / industry name", raw.get("name", "My site"))

        st.subheader("1 · Weather & data")
        o = ss().run_opts
        c = st.columns([2, 1, 1, 1])
        source = c[0].selectbox("Weather data", SOURCES, index=SOURCES.index(o["source"]), format_func=SOURCE_LABELS.get,
                                help="If live data fails it falls back automatically: same day last year → "
                                     "saved day → simulated weather.")
        day = c[1].date_input("Operating day", o["day"])
        seed = int(c[2].number_input("Random seed", 0, 10_000, o["seed"]))
        use_llm = c[3].toggle("Use Claude", o["use_llm"] and _has_key(), disabled=not _has_key(),
                              help="Needs ANTHROPIC_API_KEY in .env. Without it the optimizer + rules decide.")

        st.subheader("2 · Demand (your loads)")
        st.caption("Critical % is never cut. Flexible load can be reduced through paid demand response.")
        loads = st.data_editor(
            pd.DataFrame([{**c, "critical_pct": round(c["critical_frac"] * 100)} for c in raw["consumers"]])
            [["id", "name", "base_mw", "critical_pct", "dr_eligible", "profile"]],
            num_rows="dynamic", key="ed_loads", width="stretch",
            column_config={"id": "ID", "name": "Name",
                           "base_mw": st.column_config.NumberColumn("Typical MW", min_value=0),
                           "critical_pct": st.column_config.NumberColumn("Critical %", min_value=0, max_value=100),
                           "dr_eligible": st.column_config.CheckboxColumn("Demand response?"),
                           "profile": st.column_config.SelectboxColumn(
                               "Daily shape", options=["industrial", "commercial", "flat", "custom"])})
        up = st.file_uploader("Upload utility consumption data (CSV: one column per load ID, values in MW, "
                              "any time step)", type="csv")
        if up is not None:
            df = pd.read_csv(up)
            ids = [r["id"] for r in raw["consumers"]]
            matched = {col: df[col].astype(float).tolist() for col in df.columns if col in ids}
            if not matched and len(df.select_dtypes("number").columns):
                col = df.select_dtypes("number").columns[-1]
                matched = {ids[0]: df[col].astype(float).tolist()}
            ss().uploaded.update(matched)
            st.success(f"Consumption profiles loaded for: {', '.join(matched) or 'nothing (no matching columns)'}")

        st.subheader("3 · Generation resources & limits")
        cs, cw = st.columns(2)
        cs.caption("☀️ Solar farms")
        solar = cs.data_editor(pd.DataFrame(raw["solar_farms"] or [], columns=["id", "name", "capacity_mw", "lat", "lon"])
                               [["id", "name", "capacity_mw", "lat", "lon"]], num_rows="dynamic", key="ed_solar",
                               width="stretch", column_config={"capacity_mw": "Capacity MW"})
        cw.caption("🌬️ Wind farms")
        wind = cw.data_editor(pd.DataFrame(raw["wind_farms"] or [], columns=["id", "name", "capacity_mw", "lat", "lon"])
                              [["id", "name", "capacity_mw", "lat", "lon"]], num_rows="dynamic", key="ed_wind",
                              width="stretch", column_config={"capacity_mw": "Capacity MW"})
        st.caption("🔋 Batteries")
        bats = st.data_editor(
            pd.DataFrame(raw["batteries"] or [], columns=["id", "name", "energy_mwh", "power_mw", "soc_min", "soc_max"])
            [["id", "name", "energy_mwh", "power_mw", "soc_min", "soc_max"]], num_rows="dynamic", key="ed_bat",
            width="stretch", column_config={"energy_mwh": "Energy MWh", "power_mw": "Power MW",
                                            "soc_min": "Min charge (0–1)", "soc_max": "Max charge (0–1)"})
        g = st.columns(2)
        imp = g[0].number_input("Grid import limit (MW)", 0.0, 10_000.0, float(raw["grid"]["import_limit_mw"]))
        exp = g[1].number_input("Grid export limit (MW)", 0.0, 10_000.0, float(raw["grid"]["export_limit_mw"]))

        st.subheader("4 · Maintenance & breakdown schedule")
        st.caption("Maintenance with flexible hours may be moved by the AI to a cheaper slot. Breakdowns are fixed.")
        assets = [r["id"] for r in (raw["solar_farms"] or []) + (raw["wind_farms"] or []) + (raw["batteries"] or [])]
        maint = st.data_editor(
            pd.DataFrame(raw["schedules"].get("maintenance") or [],
                         columns=["asset", "start", "hours", "kind", "flexible_hours", "severity"]),
            num_rows="dynamic", key="ed_maint", width="stretch",
            column_config={"asset": st.column_config.SelectboxColumn("Asset", options=assets),
                           "start": st.column_config.TextColumn("Start (HH:MM)"),
                           "hours": st.column_config.NumberColumn("Hours", min_value=0.25, step=0.25),
                           "kind": st.column_config.SelectboxColumn("Type", options=["maintenance", "breakdown"]),
                           "flexible_hours": st.column_config.NumberColumn("Can move ± hours", min_value=0),
                           "severity": st.column_config.NumberColumn("Capacity lost (0–1)", min_value=0, max_value=1)})

        st.subheader("5 · Energy conservation schedule")
        cons = st.data_editor(pd.DataFrame(raw["schedules"].get("conservation") or [],
                                           columns=["start", "end", "reduction_pct"]),
                              num_rows="dynamic", key="ed_cons", width="stretch",
                              column_config={"start": "From (HH:MM)", "end": "To (HH:MM)",
                                             "reduction_pct": st.column_config.NumberColumn("Reduce demand %",
                                                                                            min_value=0, max_value=100)})

        st.subheader("6 · Targets")
        t = raw.get("targets", {})
        tc = st.columns(4)
        carbon = tc[0].number_input("Annual CO₂ target (t)", 0.0, 1e8, float(t.get("annual_carbon_target_t", 250000)),
                                    step=10_000.0)
        cost = tc[1].number_input("Target energy cost (₹/MWh)", 0.0, 50_000.0,
                                  float(t.get("target_cost_rs_per_mwh", 4500)), step=100.0)
        risk = tc[2].slider("Acceptable risk of cutting critical load (%)", 0.0, 20.0,
                            float(t.get("max_shortfall_prob", 0.02)) * 100, 0.5,
                            help="Lower = the AI buys more protection (battery reserve) against storms & failures.")
        aversion = tc[3].slider("Worst-case aversion", 0.0, 1.0, float(t.get("risk_aversion", 0.3)), 0.05,
                                help="0 = only average cost matters · 1 = worst case matters as much")

        b1, b2 = st.columns(2)
        apply = b1.form_submit_button("✅ Apply & start day", type="primary", width="stretch")
        save = b2.form_submit_button("💾 Save as profile", width="stretch")

    if apply or save:
        new = _build_raw(raw, name, loads, solar, wind, bats, imp, exp, maint, cons, carbon, cost, risk, aversion)
        errors = validate_config(config_from_dict(new))
        if errors:
            for e in errors:
                st.error(e)
            return
        if save:
            st.success(f"Saved profile: {save_profile(config_from_dict(new)).name}")
        if apply:
            ss().run_opts = {"source": source, "day": day, "seed": seed, "use_llm": use_llm}
            start_day(new, source, day, seed, use_llm)
            _clear_editors()
            st.success("Applied — open **⚡ 2 · Live control room** and press ▶ Play.")


def _clear_editors() -> None:
    for k in ("ed_loads", "ed_solar", "ed_wind", "ed_bat", "ed_maint", "ed_cons"):
        ss().pop(k, None)


def _clean(r: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in r.items() if v is not None and not (isinstance(v, float) and pd.isna(v))}


def _records(df: pd.DataFrame, prefix: str) -> List[Dict[str, Any]]:
    rows = []
    for i, r in enumerate(df.dropna(how="all").to_dict("records")):
        r = _clean(r)
        r["id"] = str(r.get("id") or f"{prefix}{i + 1}")
        r["name"] = str(r.get("name") or r["id"])
        rows.append(r)
    return rows


def _build_raw(raw, name, loads, solar, wind, bats, imp, exp, maint, cons, carbon, cost, risk, aversion) -> Dict:
    """Turn the edited tables back into a configuration dict."""
    new = copy.deepcopy(raw)
    new["name"] = name or "My site"
    old = {r["id"]: r for r in (raw["solar_farms"] or []) + (raw["wind_farms"] or []) + (raw["batteries"] or [])}
    merge = lambda rows: [{**old.get(r["id"], {}), **r} for r in rows]  # noqa: E731 — keep hidden params
    new["solar_farms"] = merge(_records(solar, "S"))
    new["wind_farms"] = merge(_records(wind, "W"))
    new["batteries"] = merge(_records(bats, "B"))
    new["consumers"] = [{"id": r["id"], "name": r["name"], "base_mw": float(r.get("base_mw", 0)),
                         "critical_frac": float(r.get("critical_pct", 50)) / 100,
                         "dr_eligible": bool(r.get("dr_eligible", False)), "profile": r.get("profile") or "industrial"}
                        for r in _records(loads, "C")]
    new["custom_profiles"] = {**(raw.get("custom_profiles") or {}), **ss().uploaded}
    for c in new["consumers"]:
        if c["id"] in new["custom_profiles"]:
            c["profile"] = "custom"
    new["grid"] = {**raw["grid"], "import_limit_mw": imp, "export_limit_mw": exp}
    new["schedules"] = {"maintenance": [_clean(r) for r in maint.dropna(subset=["asset", "start"]).to_dict("records")],
                        "conservation": [_clean(r) for r in cons.dropna(subset=["start", "end"]).to_dict("records")]}
    new["targets"] = {"annual_carbon_target_t": carbon, "target_cost_rs_per_mwh": cost,
                      "max_shortfall_prob": risk / 100, "risk_aversion": aversion}
    return new


# ================================================================ tab 2 — live control room
def live_tab() -> None:
    op = ss().op
    status_banner(op)
    output_cards(op)
    if not op.sim.history:
        st.info("Press **▶ Play** (or **⏭ Step**) in the sidebar to start the day.")
    else:
        charts(op)
    st.divider()
    thinking(op)
    st.divider()
    inject_panel(op)
    with st.expander("📜 Action log (every decision, newest first)"):
        action_log(op)


def status_banner(op: Operation) -> None:
    d = op.agent.decisions[-1] if op.agent.decisions else None
    if d is None:
        st.info("🟢 Ready. The AI decides every 15 minutes and immediately after any shock.")
        return
    worst = max((lvl for _, lvl in d.headline), key=["ok", "info", "warning", "critical"].index)
    tags = " · ".join(t for t, _ in d.headline)
    msg = (f"{ICON[worst]} **{d.time}** — {tags}  \n**AI decision:** {d.chosen_name} "
           f"(keep ≥{d.criteria['reserve_pct']:.0%} battery reserve)")
    {"ok": st.success, "info": st.info, "warning": st.warning, "critical": st.error}[worst](msg)


def output_cards(op: Operation) -> None:
    o = op.outputs()
    c = st.columns(6)
    c[0].metric("⚡ Energy produced", f"{o['energy_produced_mwh']:,.0f} MWh", border=True,
                help=f"Solar {o['solar_mwh']:,.0f} · Wind {o['wind_mwh']:,.0f} MWh")
    c[1].metric("💰 Money saved", f"₹{o['money_saved_rs'] / 1e5:,.2f} L", border=True,
                help="vs. a business-as-usual controller running the same day with the same events")
    c[2].metric("🌱 Carbon saved", f"{o['carbon_saved_t']:,.1f} t CO₂", border=True,
                help=f"AI {o['co2_t']:,.0f} t vs business-as-usual {o['baseline_co2_t']:,.0f} t")
    c[3].metric("♻️ Clean energy", f"{o['clean_pct']:.0f}%", f"{o['clean_pct'] - o['baseline_clean_pct']:+.0f} pts",
                border=True, help="Share of load served without grid (fossil) power")
    c[4].metric("⚙️ Generation efficiency", f"{o['generation_efficiency_pct']:.0f}%", border=True,
                help="Renewable energy used ÷ renewable energy available (100% = nothing wasted)")
    c[5].metric("🛡️ Load cut", f"{o['unserved_mwh']:.1f} MWh", f"{o['violations']} safety violations",
                delta_color="off", border=True,
                help=f"Business-as-usual: {o['baseline_unserved_mwh']:.1f} MWh cut, {o['baseline_violations']} violations")


def _fig(title: str, y: str, height: int = 260) -> go.Figure:
    f = go.Figure()
    f.update_layout(title=dict(text=title, font=dict(size=14)), height=height, hovermode="x unified",
                    margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25), yaxis_title=y,
                    xaxis=dict(range=[0, 24], dtick=3, ticksuffix=":00"))
    return f


def charts(op: Operation) -> None:
    h = pd.DataFrame(op.sim.history)
    x = h.hour
    a, b = st.columns(2)
    gen = _fig("Where the power came from", "MW")
    for i, (col, name) in enumerate((("solar_mw", "Solar"), ("wind_mw", "Wind"), ("discharge_mw", "Battery"),
                                     ("import_mw", "Grid (fossil)"))):
        gen.add_trace(go.Scatter(x=x, y=h[col], name=name, stackgroup="g", line=dict(width=0.5, color=SERIES[i]),
                                 hovertemplate="%{y:.0f} MW"))
    gen.add_trace(go.Scatter(x=x, y=h.demand_mw, name="Demand", line=dict(color=INK, width=2),
                             hovertemplate="%{y:.0f} MW"))
    a.plotly_chart(gen, key="gen")

    soc = _fig("Battery charge & the AI's chosen reserve", "%")
    for i, bat in enumerate(op.cfg.batteries):
        soc.add_trace(go.Scatter(x=x, y=h[f"soc_{bat.id}"], name=bat.id, line=dict(width=2, color=SERIES[i % 4]),
                                 hovertemplate="%{y:.0f}%"))
    res = [d.criteria["reserve_pct"] * 100 for d in op.agent.decisions][: len(h)]
    soc.add_trace(go.Scatter(x=x[: len(res)], y=res, name="Reserve target", line=dict(color=INK, dash="dot", width=1.5),
                             line_shape="hv", hovertemplate="%{y:.0f}%"))
    soc.update_yaxes(range=[0, 100])
    b.plotly_chart(soc, key="soc")

    price = _fig("Electricity price", "₹/MWh")
    price.add_trace(go.Scatter(x=x, y=h.price_buy, name="Buy", line=dict(width=2, color=SERIES[0]),
                               hovertemplate="₹%{y:,.0f}"))
    price.add_trace(go.Scatter(x=x, y=h.price_sell, name="Sell", line=dict(width=2, color=SERIES[1]),
                               hovertemplate="₹%{y:,.0f}"))
    a.plotly_chart(price, key="price")

    grid = _fig("Grid connection (+ buying / − selling)", "MW")
    grid.add_trace(go.Bar(x=x, y=h.import_mw - h.export_mw, name="Net import", marker_color=SERIES[0],
                          hovertemplate="%{y:.0f} MW"))
    grid.add_trace(go.Scatter(x=x, y=h.import_limit_mw, name="Line limit", line=dict(color=INK, dash="dot", width=1.5)))
    grid.add_trace(go.Scatter(x=x, y=-h.export_limit_mw, showlegend=False, line=dict(color=INK, dash="dot", width=1.5)))
    b.plotly_chart(grid, key="grid")


def _risk_rows(s: Dict[str, Any]) -> List[tuple]:
    """(label, 0–1 pressure, value text) for every signal — shown together, never collapsed into one label."""
    eta = f" in {s['storm_eta_min']} min" if s.get("storm_eta_min") is not None else ""
    return [
        ("Storm risk", s["storm_prob"], f"{s['storm_prob']:.0%}{eta}"),
        ("Price pressure", min(1, max(0, (s["price_ratio"] - 1) / 1.5)), f"{s['price_ratio']:.1f}× normal"),
        ("Grid line limited", max(0, 1 - s["line_headroom"]), f"{s['line_headroom']:.0%} available"),
        ("Batteries out of service", 1 - s["battery_available"], f"{s['battery_available']:.0%} in service"),
        ("Generation offline", s["generation_offline"], f"{s['generation_offline']:.0%} of capacity"),
        ("Demand surprise", min(1, abs(s["demand_deviation"]) / 0.25), f"{s['demand_deviation']:+.0%} vs expected"),
        ("Renewable shortfall", min(1, max(0, -s["renewable_deviation"]) / 0.5),
         f"{s['renewable_deviation']:+.0%} vs plan"),
        ("Low battery charge", 1 - s["battery_soc"], f"{s['battery_soc']:.0%} charged"),
        ("CO₂ over budget pace", min(1, max(0, s["carbon_pressure"] - 1)), f"{s['carbon_pressure']:.0%} of pace"),
    ]


def thinking(op: Operation) -> None:
    st.subheader("🧠 What the AI is thinking right now")
    if not op.agent.decisions:
        st.caption("No decision yet.")
        return
    d = op.agent.decisions[-1]
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**1 · All signals, measured together**")
        rows = _risk_rows(d.signals)
        lvl = lambda v: "critical" if v >= 0.6 else "warning" if v >= 0.25 else "ok"  # noqa: E731
        fig = go.Figure(go.Bar(
            x=[r[1] for r in rows], y=[r[0] for r in rows], orientation="h",
            marker_color=[STATUS[lvl(r[1])] for r in rows], text=[r[2] for r in rows], textposition="outside",
            hovertemplate="%{y}: %{text}<extra></extra>"))
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 1.5], visible=False),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, key="signals")
    with right:
        st.markdown(f"**2 · Diagnosis** — {d.diagnosis}")
        st.markdown(f"**3 · Decision** ({'Claude' if d.reasoner == 'llm' else 'optimizer + priority ladder'}, "
                    f"{'⚡ triggered by a shock' if d.trigger == 'shock' else 'scheduled 15-min check'})")
        st.success(d.explanation)
        for n in d.notes:
            st.caption(f"• {n}")
        if d.llm_error:
            st.caption(f"LLM note: {d.llm_error}")
    st.markdown("**4 · Every plan the AI considered** — each tested against 100 possible futures")
    rows = [{"": "✅" if o["id"] == d.chosen else ("⚠️" if o["status"] == "too risky" else
                                                  "❌" if o["status"] == "invalid" else ""),
             "Plan": o["name"], "Battery reserve": f"{o['reserve_pct']:.0%}",
             "Avg cost ₹": _num(o["cost_avg_rs"]), "Worst case ₹": _num(o["cost_p95_rs"]),
             "Risk of cutting critical load": "—" if o["p_shortfall"] is None else f"{o['p_shortfall']:.0%}",
             "CO₂ t": "—" if o["co2_t"] is None else f"{o['co2_t']:.1f}",
             "Why / why not": o["reason"]} for o in d.options]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Priority ladder: 1 physically safe → 2 risk within your tolerance → 3 best cost-vs-clean score "
               "(weighted by your carbon target & worst-case aversion) → 4 least battery wear.")


def _num(v: Any) -> str:
    return "—" if v is None else f"{v:,.0f}"


def inject_panel(op: Operation) -> None:
    st.subheader("🎛️ Make something happen")
    st.caption("Test the AI: inject a real-world event and watch it re-plan immediately.")
    c = st.columns([2, 2, 2, 1])
    kind = c[0].selectbox("Event", list(EVENT_UI), format_func=EVENT_UI.get, key="ev_kind")
    params: Dict[str, Any] = {}
    hours = c[1].slider("Lasts (hours)", 0.5, 6.0, 2.0, 0.5, key="ev_hours")
    farms = [f.id for f in (*op.cfg.solar, *op.cfg.wind)]
    if kind == "storm_alert":
        params["probability"] = c[2].slider("Chance it hits", 0.05, 1.0, 0.5, 0.05, key="ev_p")
        params["lead"] = int(c[2].slider("Arrives in (hours)", 0.5, 4.0, 2.0, 0.5, key="ev_lead") * 4)
    elif kind == "generator_breakdown":
        params["asset"] = c[2].selectbox("Which farm", farms, key="ev_asset")
        params["severity"] = c[2].slider("Capacity lost", 0.1, 1.0, 1.0, 0.1, key="ev_sev")
    elif kind == "battery_unavailable" and op.cfg.batteries:
        ids = [b.id for b in op.cfg.batteries]
        params["battery"] = ids.index(c[2].selectbox("Which battery", ids, key="ev_bat"))
    elif kind == "maintenance_window":
        params["asset"] = c[2].selectbox("Which asset", farms + [b.id for b in op.cfg.batteries], key="ev_m")
    elif kind in ("price_spike", "demand_surge"):
        params["factor"] = c[2].slider("Size (× normal)", 1.1, 4.0, 2.5 if kind == "price_spike" else 1.25, 0.05,
                                       key=f"ev_f_{kind}")
    if c[3].button("Inject", type="primary", width="stretch", disabled=op.done):
        ev = op.inject(make_event(kind, op.sim.tick, duration=int(hours * 4), **params))
        st.toast(ev.description, icon="💥")
        op.step()                       # re-plan immediately on the shock
        st.rerun()


def action_log(op: Operation) -> None:
    rows = [{"Time": d.time, "Trigger": d.trigger, "Plan": d.chosen_name, "Reserve": f"{d.criteria['reserve_pct']:.0%}",
             "Signals": " · ".join(t for t, lvl in d.headline if lvl != "ok") or "normal",
             "Battery MW (+out/−in)": round(d.action_summary["battery_discharge_mw"]
                                             - d.action_summary["battery_charge_mw"], 1),
             "Grid MW (+buy/−sell)": round(d.action_summary["import_mw"] - d.action_summary["export_mw"], 1),
             "Demand response MW": d.action_summary["dr_mw"], "Cost ₹": round(d.kpi["cost_rs"]),
             "Notes": " | ".join(d.notes)} for d in reversed(op.agent.decisions)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, height=320, width="stretch")
    st.caption("Full audit trail incl. every plan's stress-test numbers: logs/decisions.jsonl")


# ================================================================ tab 3 — results & learning
def results_tab() -> None:
    op = ss().op
    o = op.outputs()
    st.subheader("Today: AI vs business-as-usual (same weather, same events)")
    if op.sim.history:
        a, b = st.columns(2)
        comp = go.Figure()
        labels = ["Cost (₹ lakh)", "CO₂ (t ÷ 10)", "Load cut (MWh)"]
        comp.add_trace(go.Bar(name="AI", x=labels, marker_color=SERIES[0],
                              y=[o["cost_rs"] / 1e5, o["co2_t"] / 10, o["unserved_mwh"]]))
        comp.add_trace(go.Bar(name="Business as usual", x=labels, marker_color=SERIES[1],
                              y=[o["baseline_cost_rs"] / 1e5, o["baseline_co2_t"] / 10, o["baseline_unserved_mwh"]]))
        comp.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=30, b=10), title="Lower is better",
                           legend=dict(orientation="h", y=-0.15))
        a.plotly_chart(comp, key="cmp")
        vals = (o["solar_mwh"], o["wind_mwh"], o["battery_mwh"], o["grid_mwh"])
        mix = go.Figure(go.Bar(x=["Solar", "Wind", "Battery", "Grid (fossil)"], y=vals, marker_color=SERIES,
                               text=[f"{v:,.0f}" for v in vals], textposition="outside"))
        mix.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="MWh",
                          title=f"Clean vs traditional energy — {o['clean_pct']:.0f}% clean")
        b.plotly_chart(mix, key="mix")
    else:
        st.caption("Run the day to see the comparison.")

    st.subheader("🧪 What the AI has learned")
    learner = op.agent.learner
    st.caption("The AI compares every forecast with what actually happened and corrects itself; lessons are "
               f"saved at the end of each day. Days of experience: {learner.days_trained}.")
    st.dataframe(pd.DataFrame([{
        "Forecast": k.title(), "Learned correction": f"{v['correction_pct']:+.1f}%",
        "Error before learning (MW)": v["mae_raw_mw"], "Error after learning (MW)": v["mae_learned_mw"],
        "Outcomes inside P10–P90 band": "—" if v["coverage_pct"] is None else f"{v['coverage_pct']:.0f}%",
        "Samples": v["samples"]} for k, v in learner.report().items()]), hide_index=True, width="stretch")

    st.subheader("📊 Proof over many days (Monte Carlo)")
    st.caption("Each run is one randomized day with 3 unannounced shocks. All controllers see identical conditions.")
    c1, c2 = st.columns([1, 3])
    n = c1.number_input("Days", 5, 500, 10, step=5)
    if c1.button("Run Monte Carlo", type="primary"):
        bar = c2.progress(0.0, text="Simulating…")
        df = run_monte_carlo(int(n), progress=lambda f: bar.progress(f, text=f"Simulating… {f:.0%}"))
        save_outputs(df)
        bar.empty()
    path = OUT_DIR / "results.csv"
    if not path.exists():
        st.info("No results yet. Click **Run Monte Carlo** or run `python -m eval.evaluate --n 100`.")
        return
    df = pd.read_csv(path)
    summary = summarize(df)["mean"]
    ai, base = summary.loc["agent"], summary.loc["baseline"]
    m = st.columns(4)
    m[0].metric("Cost vs business-as-usual", f"{(ai.cost_rs / base.cost_rs - 1) * 100:+.1f}%", border=True)
    m[1].metric("Clean energy", f"{ai.clean_pct:.1f}%", f"{ai.clean_pct - base.clean_pct:+.1f} pts", border=True)
    m[2].metric("Load cut / day", f"{ai.unserved_mwh:.1f} MWh", f"{ai.unserved_mwh - base.unserved_mwh:+.1f} MWh",
                delta_color="inverse", border=True)
    m[3].metric("Safety violations / day", f"{ai.violations:.2f}", f"{ai.violations - base.violations:+.1f}",
                delta_color="inverse", border=True)
    metric = st.radio("Metric", list(METRICS), format_func=METRICS.get, horizontal=True)
    fig = go.Figure()
    for i, c in enumerate(CONTROLLERS):
        fig.add_trace(go.Box(y=df[df.controller == c][metric], name=c, marker_color=SERIES[i], boxpoints="all",
                             jitter=0.4, pointpos=0, marker=dict(size=4, opacity=0.5)))
    fig.update_layout(height=380, showlegend=False, margin=dict(l=10, r=10, t=30, b=10),
                      title=f"{METRICS[metric]} per day over {df.seed.nunique()} days")
    st.plotly_chart(fig, key="mc")


# ================================================================ main
def main() -> None:
    init()
    sidebar()
    t1, t2, t3 = st.tabs(["🏭 1 · Your industry", "⚡ 2 · Live control room", "📊 3 · Results & learning"])
    with t1:
        setup_tab()
    with t2:
        live_tab()
    with t3:
        results_tab()
    op = ss().op
    if ss().playing and not op.done:
        steps, delay = {"slow": (1, 1.0), "normal": (1, 0.2), "fast": (3, 0.1), "very fast": (8, 0.0)}[ss().speed]
        for _ in range(steps):
            if not op.done:
                op.step()
        time.sleep(delay)
        st.rerun()
    elif op.done and ss().playing:
        ss().playing = False
        st.toast("Day complete ✅ — see 📊 Results & learning")


main()
