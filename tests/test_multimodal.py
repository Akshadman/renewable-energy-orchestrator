import io
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agent.llm import LLMClient, from_env
from core.data_sources import SyntheticProvider
from core.multimodal import documents, intake, satellite, thermal
from core.multimodal.samples import DOCS, build_sample_pack
from core.simulator import Simulator


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "LLM_BASE_URL",
              "LLM_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)


def _png(img):
    buf = io.BytesIO()
    thermal.save_thermal(img, buf)
    return buf.getvalue()


# ---------------------------------------------------------------- thermal
@pytest.mark.parametrize("n", [0, 3, 12])
def test_thermal_detects_known_defects(n):
    img, truth = thermal.make_thermal("S3", n, seed=n + 1)
    ev = thermal.analyse(Image.open(io.BytesIO(_png(img))), "x.png")
    assert ev.details["defective_modules"] == n and ev.asset == "S3"
    assert ev.value == pytest.approx(n / 120 * thermal.LOSS_PER_DEFECT)
    assert ev.details["needs_inspection"] == (n / 120 >= thermal.ACTION_SHARE)


def test_thermal_noisy_image_is_not_trusted():
    img, _ = thermal.make_thermal("S3", 5, seed=3, noise=2.2, delta=(8, 13), radius=(2, 4), glare=True)
    ev = thermal.analyse(Image.open(io.BytesIO(_png(img))), "x.png")
    assert ev.confidence < 0.75


# ---------------------------------------------------------------- documents
def test_sample_documents_extracted(cfg):
    got = {name: documents.analyse(name, text.encode(), cfg, 40) for name, text in DOCS.items()
           if not name.endswith(".pdf")}
    bd = got["fault_report_BESS_Bhadla.txt"]
    assert bd.kind == "breakdown" and bd.asset == "B1" and bd.duration_ticks == 16 and bd.confidence >= 0.75
    dm = got["production_notice_steel_mill.txt"]
    assert dm.kind == "demand_change" and dm.asset == "C1" and dm.start_tick == 76 and dm.duration_ticks == 16
    assert dm.value == pytest.approx(1 + 15 / 60)
    assert got["tentative_maintenance_email.txt"].status == "quarantined"


def test_pdf_documents_are_read(cfg, tmp_path):
    build_sample_pack(tmp_path)
    ev = documents.analyse("SLDC_congestion_notice.pdf", (tmp_path / "SLDC_congestion_notice.pdf").read_bytes(), cfg)
    assert ev.kind == "line_limit" and ev.value == pytest.approx(0.6) and ev.start_tick == 70 and ev.duration_ticks == 12
    wo = documents.analyse("wo.pdf", (tmp_path / "WO-2231_W2_gearbox_maintenance.pdf").read_bytes(), cfg)
    assert wo.kind == "maintenance" and wo.asset == "W2" and wo.details["flex_ticks"] == 12


@pytest.mark.parametrize("text, start", [("Planned maintenance of W1 from 2 pm to 4 pm.", 56),
                                         ("Planned maintenance of W1 from 14.00 hrs for 2 hours.", 56),
                                         ("Date 11.10.2026: planned maintenance of W1 at 14:00 for 2 hours.", 56)])
def test_time_formats_and_dates(cfg, text, start):
    ev = documents.analyse("n.txt", text.encode(), cfg)
    assert ev.kind == "maintenance" and ev.start_tick == start and ev.duration_ticks == 8 and ev.status == "pending"


def test_llm_extraction_is_validated_and_falls_back(cfg):
    class Good:
        label = "fake:x"

        def call_tool(self, system, user, tool):
            return {"kind": "breakdown", "assets": ["W2"], "start_time": "13:00", "duration_hours": 3, "hedged": False}

    class Broken(Good):
        def call_tool(self, *a):
            raise RuntimeError("down")
    ev = documents.analyse("n.txt", b"anything", cfg, 0, Good())
    assert ev.kind == "breakdown" and ev.asset == "W2" and ev.start_tick == 52 and ev.details["extractor"].startswith("llm")
    fb = documents.analyse("n.txt", b"Fault: W3 tripped at 10:00, out of service for 2 hours.", cfg, 0, Broken())
    assert fb.details["extractor"] == "rules" and fb.asset == "W3"


# ---------------------------------------------------------------- satellite
def test_cloud_fraction():
    img = np.full((50, 50, 3), [190, 175, 140], dtype=np.uint8)       # desert sand
    assert satellite.cloud_fraction(img)[0] == 0
    img[:25] = 240                                                     # top half white cloud
    assert satellite.cloud_fraction(img)[0] == pytest.approx(50, abs=1)


def test_satellite_finding_with_mocked_service(cfg, tmp_path):
    wd = SyntheticProvider().get_day(cfg, date(2026, 10, 11), seed=1)
    wd.cloud_cover[:] = 10.0

    class Resp:
        def __init__(self, cloudy):
            arr = np.full((64, 64, 3), [190, 175, 140], dtype=np.uint8)
            if cloudy:
                arr[:48] = 245
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG")
            self.content = buf.getvalue()

        def raise_for_status(self):
            pass

    calls = []

    def get(url, params=None, timeout=None):
        calls.append(params)
        return Resp(cloudy=float(params["BBOX"].split(",")[0]) > 26.0)   # cloud over the northern farms
    found = satellite.analyse_sites(cfg, wd, date(2026, 10, 11), tmp_path, get)
    kinds = {e.asset: e.kind for e in found}
    assert kinds["S1"] == "cloud_bank" and kinds["S5"] == "none" and len(calls) == len(cfg.solar)
    satellite.analyse_sites(cfg, wd, date(2026, 10, 11), tmp_path, get)
    assert len(calls) == len(cfg.solar)                                  # cached


# ---------------------------------------------------------------- intake → twin
def test_intake_applies_to_twin_and_quarantines(cfg, tmp_path):
    folder = build_sample_pack(tmp_path)
    sim = Simulator(cfg, seed=1)
    ev = intake.process(intake.load_folder(folder), cfg, 40)
    intake.apply_all(ev, [sim], 40)
    status = {e.source: e.status for e in ev}
    assert status["tentative_maintenance_email.txt"] == "quarantined"
    assert status["thermal_S4_block2_overexposed.png"] == "quarantined"
    assert status["SLDC_congestion_notice.pdf"] == "applied"
    assert sim.import_limit_at(72) == pytest.approx(cfg.grid.import_limit_mw * 0.6)
    assert sim.demand_mult[78, 0] == pytest.approx(1.25)
    assert not sim.battery_available_at(41)[0]
    assert any(m.asset_id == "W2" and m.window == 12 for m in sim.maintenance)
    assert sim.solar_avail[60, 1] == pytest.approx(0.95)                 # S2 derated 5% by thermal finding
    assert any(a.kind.startswith("evidence_") for a in sim.event_alerts)


def test_bad_files_never_crash(cfg):
    ev = intake.process([("broken.png", b"not an image"), ("x.docx", b"..."), ("e.txt", b"")], cfg)
    assert [e.status for e in ev] == ["rejected", "rejected", "info"]


# ---------------------------------------------------------------- free LLM providers
def test_provider_selection(monkeypatch):
    assert from_env() is None
    monkeypatch.setenv("GROQ_API_KEY", "x")
    c = from_env()
    assert c.provider == "groq" and "groq.com" in c.base_url
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    assert from_env().provider == "gemini"


def test_openai_compatible_tool_loop(monkeypatch):
    sent = []

    class R:
        status_code = 200

        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    def post(url, json=None, headers=None, timeout=None):
        import copy
        sent.append(copy.deepcopy(json))
        if len(sent) == 1:
            return R({"choices": [{"message": {"tool_calls": [{"id": "1", "function": {
                "name": "propose_plan", "arguments": '{"reserve_pct": 0.4}'}}]}}]})
        return R({"choices": [{"message": {"tool_calls": [{"id": "2", "function": {
            "name": "submit_decision", "arguments": '{"option_id": "O3"}'}}]}}]})
    monkeypatch.setattr("agent.llm.requests.post", post)
    client = LLMClient("groq", "m", "k", "https://example")
    seen = []
    out = client.run_tools("sys", "user", [{"name": "propose_plan", "input_schema": {}},
                                           {"name": "submit_decision", "input_schema": {}}],
                           lambda n, a: seen.append((n, a)) or {"ok": 1}, "submit_decision")
    assert out == {"option_id": "O3"} and seen == [("propose_plan", {"reserve_pct": 0.4})]
    assert sent[1]["messages"][-1]["role"] == "tool" and json.loads(sent[1]["messages"][-1]["content"]) == {"ok": 1}
