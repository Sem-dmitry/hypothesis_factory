# -*- coding: utf-8 -*-

"""
Offline tests for the web UI backend (FastAPI). The real deep run is stubbed by
monkeypatching ``engine.run_deep`` (no keys/network). UTF-8 form fields are
exercised via TestClient (the browser sends proper UTF-8 multipart; a shell +
curl would mangle Cyrillic).
"""

import json
import os

import pytest

os.environ.setdefault("ROUTER_AI_API_KEY", "dummy")

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("COSCIENTIST_DIR", str(tmp_path))
    # Re-import the server so its module-level stores use the tmp COSCIENTIST_DIR.
    import importlib

    import webapp.server as srv
    srv = importlib.reload(srv)
    return TestClient(srv.app)


def test_health_and_settings(client):
    assert client.get("/api/health").status_code == 200
    s = client.get("/api/settings").json()
    assert set(s["agents"].keys()) >= {"literature", "generation", "assessment"}
    phase_keys = [p["key"] for p in s["phases"]]
    assert phase_keys[0] == "Constraints"  # auto-elicitation runs first
    assert "Literature" in phase_keys
    assert s["max_hypotheses"] == 12
    assert s["auto_elicit_constraints"] is True
    assert "google/gemini-2.5-flash" in s["model_options"]


def test_index_and_static(client):
    assert client.get("/").status_code == 200
    html = client.get("/").text
    app_js = client.get("/static/app.js")
    styles = client.get("/static/styles.css")
    assert "Фабрика гипотез" in html
    assert app_js.status_code == 200
    assert styles.status_code == 200
    assert 'id="viewTabs"' in html
    assert 'data-view="analysis"' in html
    assert 'id="viewAnalysis"' in html
    assert 'id="liveHypotheses"' in html
    assert 'id="fileList"' in html
    assert 'id="maxHyp"' in html
    assert 'id="sourceOverlay"' in html
    assert html.index('data-view="analysis"') < html.index('data-view="feed"')
    assert "selectedFiles" in app_js.text
    assert "max_hypotheses" in app_js.text
    assert "activeView: 'analysis'" in app_js.text
    assert "analysis: '#viewAnalysis'" in app_js.text
    assert "source_evidence" in app_js.text
    assert "/sources/" in app_js.text
    assert "appendConsoleLog" in app_js.text
    assert "flex: 0 0 auto" in styles.text
    assert ".source-item blockquote" in styles.text


def test_source_chunk_endpoint(client):
    import webapp.server as srv
    from coscientist.corpus.loaders import CorpusChunk
    from coscientist.corpus.store import CorpusIndex

    def embed(texts):
        return [[float(len(t)), 1.0] for t in texts]

    run_id = "sourcecase"
    chunk = CorpusChunk(
        text="Прямая цитата: никель теряется в тонких хвостах из-за недораскрытия.",
        source_path="report.pdf",
        source_name="report.pdf",
        modality="pdf",
        locator="p.7",
    )
    index = CorpusIndex(embed_fn=embed)
    index.add([chunk])
    os.makedirs(srv.store.run_dir(run_id), exist_ok=True)
    index.save(os.path.join(srv.store.run_dir(run_id), "corpus"))

    r = client.get(f"/api/runs/{run_id}/sources/{chunk.chunk_id}")
    assert r.status_code == 200
    payload = r.json()
    assert payload["source_name"] == "report.pdf"
    assert payload["locator"] == "p.7"
    assert "Прямая цитата" in payload["text"]


def test_projects_crud(client):
    projs = client.get("/api/projects").json()
    assert len(projs) >= 1  # a default project is ensured
    created = client.post("/api/projects", json={"name": "Медный цикл"}).json()
    assert created["name"] == "Медный цикл"
    names = [p["name"] for p in client.get("/api/projects").json()]
    assert "Медный цикл" in names


def _stub_run_deep(monkeypatch, captured=None):
    """Replace the real deep tournament with an offline stub that emits a couple
    of agent events and persists a record — so the e2e (SSE, record, logs) runs
    without keys/network."""
    import uuid
    from datetime import datetime, timezone

    import webapp.server as srv
    from coscientist.hypothesis_assessment import HypothesisAssessment
    from coscientist.studio import RunRecord, TranscriptMessage

    def fake_run_deep(*, goal, constraints, settings, corpus_files, use_web,
                      on_event, pull_messages=None):
        if captured is not None:
            captured["corpus_files"] = list(corpus_files or [])
            captured["use_web"] = use_web
        tm = TranscriptMessage(agent="Generation", role="assistant", model="fake",
                               prompt=f"P: {goal}", content="Гипотеза", tokens_in=10,
                               tokens_out=20, cost_usd=0.0, seconds=0.1)
        on_event(tm)
        on_event(TranscriptMessage(agent="Assessor", role="assistant", model="fake",
                                   prompt="assess", content="{}", tokens_in=5,
                                   tokens_out=5, cost_usd=0.0, seconds=0.1))
        rec = RunRecord(
            id=uuid.uuid4().hex[:12], project=settings.project, goal=goal,
            constraints=constraints,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            settings=settings,
            assessments=[HypothesisAssessment(hypothesis="Гипотеза про никель",
                                              overall_score=7.0).model_dump()],
            transcript=[tm], metrics={"messages": 2}, status="completed", mode="deep",
        )
        srv.store.save(rec)
        return rec

    monkeypatch.setattr(srv.engine, "run_deep", fake_run_deep)


def _consume_run_events(client, run_id):
    types = []
    with client.stream("GET", f"/api/runs/{run_id}/events") as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            payload = line[5:] if line.startswith("data:") else line
            try:
                ev = json.loads(payload)
            except Exception:
                continue
            types.append(ev.get("type"))
            if ev.get("type") in ("done", "error"):
                break
    return types


def test_run_end_to_end_utf8(client, monkeypatch):
    _stub_run_deep(monkeypatch)
    proj = client.get("/api/projects").json()[0]
    pname = proj["name"]  # proper UTF-8 (Cyrillic)
    r = client.post("/api/runs", data={
        "goal": "Снизить потери никеля с хвостами",
        "constraints": "Действующая схема",
        "project": pname,
        "use_web": "false",
        "settings": json.dumps({"num_hypotheses": 3, "max_hypotheses": 9}),
    })
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    # consume the SSE stream to completion
    types = _consume_run_events(client, run_id)
    assert "agent" in types and "log" in types and "done" in types

    # the record is now discoverable under the project with correct UTF-8
    runs = client.get(f"/api/projects/{proj['id']}/runs").json()
    assert runs, "run should appear under its project"
    assert runs[0]["project"] == pname  # UTF-8 preserved (no mojibake)
    rec = client.get(f"/api/runs/{runs[0]['id']}").json()
    assert len(rec["assessments"]) >= 1
    assert rec["settings"]["max_hypotheses"] == 9
    # full request/response log is available and non-trivial
    log = client.get(f"/api/runs/{runs[0]['id']}/log").text
    assert "PROMPT" in log and "RESPONSE" in log
    console_log = client.get(f"/api/runs/{runs[0]['id']}/console-log").text
    assert "real mode" in console_log and "Run completed" in console_log


def test_run_accepts_multipart_corpus_files(client, monkeypatch):
    captured = {}
    _stub_run_deep(monkeypatch, captured)
    r = client.post(
        "/api/runs",
        data={
            "goal": "Снизить потери никеля с хвостами",
            "constraints": "Действующая схема",
            "project": "Хвосты",
            "use_web": "false",
            "settings": json.dumps({"num_hypotheses": 2}),
        },
        files=[
            ("files", ("report.pdf", b"nickel flotation evidence", "application/pdf")),
            ("files", ("tails.xlsx", b"tail grade data", "application/vnd.ms-excel")),
        ],
    )
    assert r.status_code == 200
    _consume_run_events(client, r.json()["run_id"])
    paths = captured["corpus_files"]
    assert len(paths) == 2
    assert {os.path.basename(p) for p in paths} == {"report.pdf", "tails.xlsx"}
    assert all(os.path.exists(p) for p in paths)
    assert captured["use_web"] is False
