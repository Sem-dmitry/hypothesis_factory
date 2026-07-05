"""Offline tests for expert-in-the-loop refinement (customer feature 3).

Covers the pure refinement functions (chat / branch) with a fake LLM and no
corpus, plus the webapp endpoints (feedback / chat / branch) via TestClient so
the feature is proven wired into the running uvicorn app.
"""

import json
import os

import pytest
from fastapi.testclient import TestClient

from coscientist.refinement import (
    branch_hypotheses,
    chat_about_hypothesis,
)


class FakeLLM:
    """Records prompts; returns a canned reply (str or per-call list)."""

    def __init__(self, reply):
        self._reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        reply = self._reply
        if isinstance(reply, list):
            reply = reply[min(len(self.prompts) - 1, len(reply) - 1)]

        class M:
            pass

        m = M()
        m.content = reply
        return m


# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #


def test_chat_grounds_and_returns_answer():
    llm = FakeLLM("Коротко: увеличьте расход собирателя на 10%. [1]")
    reply = chat_about_hypothesis(
        hypothesis="Повысить дозу собирателя",
        assessment={"overall_score": 7, "mechanism_of_influence": "x"},
        goal="Повысить извлечение никеля",
        message="Как проверить на практике?",
        history=[{"role": "user", "content": "привет"}],
        retriever=None,
        llm=llm,
    )
    assert "собирателя" in reply
    # The hypothesis, the question, and the history all reach the prompt.
    prompt = llm.prompts[0]
    assert "Повысить дозу собирателя" in prompt
    assert "Как проверить на практике?" in prompt


def test_chat_never_raises_on_bad_llm():
    class Boom:
        def invoke(self, _):
            raise RuntimeError("model down")

    reply = chat_about_hypothesis(
        hypothesis="h", assessment={}, goal="g", message="m",
        history=None, retriever=None, llm=Boom(),
    )
    assert isinstance(reply, str) and reply  # error string, not an exception


def test_branch_generates_and_assesses_new_hypotheses():
    # First call → JSON array of new hypotheses; subsequent calls → assessments.
    hyps = json.dumps(["Гипотеза A про пенообразователь", "Гипотеза B про pH"], ensure_ascii=False)
    assessment = json.dumps({
        "hypothesis": "ignored", "overall_score": 6.0, "novelty_score": 5,
        "feasibility_score": 7, "impact_score": 6, "risk_level": "medium",
        "mechanism_of_influence": "механизм", "citations": [],
    }, ensure_ascii=False)
    llm = FakeLLM([hyps, assessment, assessment])

    out = branch_hypotheses(
        hypothesis="Базовая гипотеза",
        direction="другой пенообразователь",
        goal="Повысить извлечение",
        constraints="Без нового оборудования",
        retriever=None,
        llm=llm,
        n=2,
    )
    assert len(out) == 2
    assert out[0].hypothesis.startswith("Гипотеза A")
    assert out[1].hypothesis.startswith("Гипотеза B")
    # Direction + constraints reach the branch prompt.
    assert "другой пенообразователь" in llm.prompts[0]
    assert "Без нового оборудования" in llm.prompts[0]


def test_branch_returns_empty_on_bad_reply():
    llm = FakeLLM("no json here")
    out = branch_hypotheses(
        hypothesis="h", direction="d", goal="g", constraints="",
        retriever=None, llm=llm, n=2,
    )
    assert out == []


# --------------------------------------------------------------------------- #
# Webapp endpoints (wired into the running uvicorn app)
# --------------------------------------------------------------------------- #


@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_AI_API_KEY", "dummy")
    monkeypatch.setenv("COSCIENTIST_DIR", str(tmp_path / "data"))
    # Fresh import so the store points at the temp data dir.
    import importlib

    import webapp.server as server

    importlib.reload(server)

    # A saved run with one assessed hypothesis to refine.
    from coscientist.studio import RunRecord, RunSettings

    rec = RunRecord(
        id="run-refine",
        project="p",
        goal="Повысить извлечение никеля",
        constraints="",
        created_at="2026-07-04T00:00:00",
        settings=RunSettings(),
        effective_constraints="Без нового оборудования",
        status="done",
        assessments=[{
            "hypothesis": "Повысить дозу собирателя",
            "overall_score": 7.0, "novelty_score": 6, "feasibility_score": 7,
            "impact_score": 7, "risk_level": "medium", "citations": [],
        }],
    )
    server.store.save(rec)

    # Stub the refinement LLM so the endpoints run offline.
    monkeypatch.setattr(server, "_refine_llm", lambda: FakeLLM(
        json.dumps(["Новая гипотеза X", "Новая гипотеза Y"], ensure_ascii=False)
    ))
    return TestClient(server.app), server


def test_feedback_roundtrip(app_client):
    client, _ = app_client
    r = client.post("/api/runs/run-refine/feedback", json={
        "hypothesis": "Повысить дозу собирателя",
        "outcome": "confirmed", "note": "сработало на пилоте",
    })
    assert r.status_code == 200
    assert r.json()["counts"]["confirmed"] == 1

    got = client.get("/api/runs/run-refine/feedback").json()
    assert got["counts"]["confirmed"] == 1
    assert got["records"][0]["note"] == "сработало на пилоте"


def test_feedback_rejects_bad_outcome(app_client):
    client, _ = app_client
    r = client.post("/api/runs/run-refine/feedback", json={
        "hypothesis": "x", "outcome": "maybe",
    })
    assert r.status_code == 400


def test_chat_endpoint(app_client, monkeypatch):
    client, server = app_client
    monkeypatch.setattr(server, "_refine_llm", lambda: FakeLLM("Ответ эксперта."))
    r = client.post("/api/runs/run-refine/chat", json={
        "hypothesis": "Повысить дозу собирателя",
        "message": "Как проверить?", "history": [],
    })
    assert r.status_code == 200
    assert r.json()["reply"] == "Ответ эксперта."


def test_branch_endpoint(app_client):
    client, _ = app_client
    r = client.post("/api/runs/run-refine/branch", json={
        "hypothesis": "Повысить дозу собирателя",
        "direction": "другой собиратель", "n": 2,
    })
    assert r.status_code == 200
    hyps = r.json()["hypotheses"]
    # Fake LLM returns the same list for every call → 2 new hypotheses assessed.
    assert len(hyps) == 2
    assert hyps[0]["hypothesis"] == "Новая гипотеза X"
