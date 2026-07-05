"""Offline tests for the Studio service layer (no Streamlit, no network).

The light run and demo mode were removed; the deep run is exercised in
``test_studio_deep.py`` with a fake framework. Here we cover the model-agnostic
service pieces: token/cost estimation, store persistence/history, run export,
and the no-Streamlit import guarantee — all built from records directly.
"""

import os
import uuid
from datetime import datetime, timezone

from coscientist.hypothesis_assessment import HypothesisAssessment
from coscientist.studio import (
    RunRecord,
    RunSettings,
    StudioEngine,
    StudioStore,
    estimate_cost,
    estimate_tokens,
)


def _record(project="P", goal="Снизить потери никеля", created_at=None):
    return RunRecord(
        id=uuid.uuid4().hex[:12],
        project=project,
        goal=goal,
        created_at=created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        settings=RunSettings(project=project),
        assessments=[
            HypothesisAssessment(
                hypothesis="Доизмельчение хвостов повысит извлечение никеля",
                overall_score=6.5, mechanism_of_influence="раскрытие пентландита",
                target_kpi_impact="+1–2% к извлечению",
                citations=["[1] report.pdf — p.1 (PDF)"],
            ).model_dump(),
        ],
        status="completed",
        mode="deep",
    )


# --- cost / token estimation --------------------------------------------------


def test_token_and_cost_estimation():
    assert estimate_tokens("") == 1
    assert estimate_tokens("x" * 40) == 10
    c = estimate_cost("google/gemini-2.5-pro", 1000, 1000)
    assert c == round(0.00125 + 0.010, 6)
    # unknown model -> default price, still a positive number
    assert estimate_cost("unknown/model", 1000, 0) > 0


# --- store persistence / history ---------------------------------------------


def test_store_roundtrip_and_history(tmp_path):
    store = StudioStore(base_dir=str(tmp_path))
    r1 = _record(project="A", goal="Goal 1", created_at="2026-07-04T00:00:01")
    r2 = _record(project="B", goal="Goal 2", created_at="2026-07-04T00:00:02")
    store.save(r1)
    store.save(r2)

    runs = store.list_runs()
    assert len(runs) == 2
    assert {m["project"] for m in runs} == {"A", "B"}
    assert runs[0]["created_at"] >= runs[1]["created_at"]  # newest first

    loaded = store.load_run(r1.id)
    assert loaded.id == r1.id and loaded.goal == "Goal 1"
    assert loaded.assessment_objects()[0].hypothesis


def test_store_loads_old_record_without_offline_demo(tmp_path):
    """Records saved before offline_demo was removed still deserialize."""
    store = StudioStore(base_dir=str(tmp_path))
    rec = _record()
    d = rec.model_dump()
    d["settings"]["offline_demo"] = True  # legacy field, now unknown
    import json

    run_dir = store.run_dir(rec.id)
    os.makedirs(os.path.join(run_dir, "exports"), exist_ok=True)
    with open(os.path.join(run_dir, "record.json"), "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False)
    loaded = store.load_run(rec.id)  # must not raise on the unknown field
    assert loaded.id == rec.id


# --- run export ---------------------------------------------------------------


def test_export_run(tmp_path):
    engine = StudioEngine(StudioStore(base_dir=str(tmp_path)))
    rec = _record()
    out = str(tmp_path / "out")
    paths = engine.export_run(rec, out)
    for key in ("md", "html", "docx", "pdf", "csv", "json", "jira", "graph"):
        assert os.path.exists(paths[key]) and os.path.getsize(paths[key]) > 0


# --- import hygiene -----------------------------------------------------------


def test_studio_import_has_no_streamlit_dep():
    import importlib

    mod = importlib.import_module("coscientist.studio")
    assert hasattr(mod, "StudioEngine")
    assert not hasattr(mod.StudioEngine, "run")  # light run removed
    src = open(mod.__file__, encoding="utf-8").read()
    assert "import streamlit" not in src
