"""
Offline tests for live agent steering (chat-to-running-agents).

Covers the four layers: state guidance, prompt rendering, the framework's
message-collection helper, and the webapp queue + endpoint (via TestClient) —
all without network or real LLMs.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from coscientist.common import load_prompt
from coscientist.framework import _collect_guidance
from coscientist.global_state import CoscientistState, CoscientistStateManager


# --------------------------------------------------------------------------- #
# State layer
# --------------------------------------------------------------------------- #


def _manager():
    # Unique goal per manager: CoscientistState creates a goal-hashed output dir.
    goal = f"steer-test-{uuid.uuid4().hex}"
    try:
        CoscientistState.clear_goal_directory(goal)
    except Exception:
        pass
    return CoscientistStateManager(CoscientistState(goal=goal))


def test_add_guidance_and_text():
    m = _manager()
    assert m.guidance_text == "" and m.num_guidance == 0
    m.add_guidance("  сфокусируйся на реагентном режиме  ")
    m.add_guidance("")  # ignored
    m.add_guidance("без нового оборудования")
    assert m.num_guidance == 2
    assert "реагентном режиме" in m.guidance_text
    assert "без нового оборудования" in m.guidance_text


def test_generation_state_carries_guidance():
    m = _manager()
    m.add_guidance("проверь пенообразователь")
    # next_generation_state needs a literature review present.
    m._state.literature_review = {"subtopic_reports": ["обзор"]}
    st = m.next_generation_state(mode="independent")
    assert "пенообразователь" in st["user_guidance"]


def test_supervisor_state_carries_guidance():
    m = _manager()
    m.add_guidance("пора завершать, хватит эволюций")
    st = m.next_supervisor_state(max_actions=40)
    assert "завершать" in st["user_guidance"]


# --------------------------------------------------------------------------- #
# Prompt rendering
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    ["independent_generation", "collaborative_generation",
     "evolve_from_feedback", "out_of_the_box", "supervisor_decision"],
)
def test_prompts_render_guidance_when_present_and_omit_when_absent(name):
    with_g = load_prompt(name, goal="g", user_guidance="- STEERZ")
    assert "STEERZ" in with_g
    assert "Live guidance" in with_g
    without = load_prompt(name, goal="g", user_guidance="")
    assert "STEERZ" not in without
    assert "Live guidance" not in without


# --------------------------------------------------------------------------- #
# Framework message-collection helper
# --------------------------------------------------------------------------- #


def test_collect_guidance_cleans_and_filters():
    assert _collect_guidance(lambda: ["  a ", "", "  ", "b"]) == ["a", "b"]


def test_collect_guidance_none_and_broken_provider():
    assert _collect_guidance(None) == []

    def boom():
        raise RuntimeError("provider down")

    assert _collect_guidance(boom) == []


# --------------------------------------------------------------------------- #
# Webapp queue + endpoint
# --------------------------------------------------------------------------- #


@pytest.fixture()
def app_and_run(monkeypatch, tmp_path):
    monkeypatch.setenv("ROUTER_AI_API_KEY", "dummy")
    monkeypatch.setenv("COSCIENTIST_DIR", str(tmp_path / "data"))
    import importlib

    import webapp.server as server

    importlib.reload(server)
    run = server._Run("run-steer", "proj")
    run.status = "running"
    server.runs.runs["run-steer"] = run
    return TestClient(server.app), server, run


def test_message_endpoint_queues_and_echoes(app_and_run):
    client, _server, run = app_and_run
    r = client.post("/api/runs/run-steer/message", json={"text": "сфокусируйся на pH"})
    assert r.status_code == 200 and r.json()["ok"] is True
    # Folded into the inbox the framework will drain.
    assert run.drain_inbox() == ["сфокусируйся на pH"]
    # Echoed into the SSE stream as a user_message event.
    ev = run.q.get_nowait()
    assert ev["type"] == "user_message" and ev["text"] == "сфокусируйся на pH"


def test_message_endpoint_rejects_empty_and_inactive(app_and_run):
    client, _server, run = app_and_run
    assert client.post("/api/runs/run-steer/message", json={"text": "  "}).status_code == 400
    assert client.post("/api/runs/missing/message", json={"text": "x"}).status_code == 404
    run.status = "completed"
    assert client.post("/api/runs/run-steer/message", json={"text": "x"}).status_code == 409


def test_drain_inbox_is_idempotent(app_and_run):
    _client, _server, run = app_and_run
    run.add_message("one")
    run.add_message("two")
    assert run.drain_inbox() == ["one", "two"]
    assert run.drain_inbox() == []  # cleared
