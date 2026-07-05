"""
Resilience of hypothesis generation: a transient empty/unparseable model reply
must be retried, and a hypothesis that keeps failing must be skipped rather than
aborting the whole tournament (inspired by the reflector/retry pattern studied in
for_inspiration).
"""

import asyncio

import pytest

import coscientist.framework as fwmod
import coscientist.global_state as gs
from coscientist.custom_types import ParsedHypothesis


def _good():
    return ParsedHypothesis(hypothesis="Доизмельчение повысит извлечение никеля",
                            predictions=["p"], assumptions=["a"])


def _blank():
    return ParsedHypothesis(hypothesis="   ", predictions=["p"], assumptions=["a"])


class _ScriptedAgent:
    """A fake generation agent whose .invoke follows a shared script list."""

    def __init__(self, script):
        self.script = script

    def invoke(self, state):
        action = self.script.pop(0)
        if action == "raise":
            raise ValueError("no content and tool calls in response")
        if action == "blank":
            return {"hypothesis": _blank()}
        return {"hypothesis": _good()}


def _framework(tmp_path, monkeypatch, script):
    monkeypatch.setattr(gs, "_OUTPUT_DIR", str(tmp_path / "cosci"))
    # force the independent path for determinism
    monkeypatch.setattr(fwmod.CoscientistFramework, "list_generation_modes",
                        lambda self: ["independent"])
    monkeypatch.setattr(fwmod, "build_generation_agent",
                        lambda mode, config: _ScriptedAgent(script))

    cfg = fwmod.CoscientistConfig(
        literature_review_agent_llm=object(),
        generation_agent_llms={"m": object()},
        reflection_agent_llms={"m": object()},
        evolution_agent_llms={"m": object()},
        meta_review_agent_llm=object(),
        supervisor_agent_llm=object(),
        final_report_agent_llm=object(),
        proximity_agent_embedding_model=object(),
        specialist_fields=["flotation", "mineral processing"],
    )
    state = gs.CoscientistState(goal="Снизить потери никеля")
    mgr = gs.CoscientistStateManager(state)
    mgr.update_literature_review({
        "goal": state.goal, "max_subtopics": 1,
        "subtopics": ["s"], "subtopic_reports": ["Некоторый обзор."], "meta_review": "",
    })
    return fwmod.CoscientistFramework(cfg, mgr), mgr


def test_generation_retries_then_succeeds(tmp_path, monkeypatch):
    fw, mgr = _framework(tmp_path, monkeypatch, script=["raise", "good"])
    assert fw._generate_new_hypothesis() is True
    assert len(mgr._state.generated_hypotheses) == 1


def test_generation_treats_blank_text_as_failure_then_retries(tmp_path, monkeypatch):
    fw, mgr = _framework(tmp_path, monkeypatch, script=["blank", "good"])
    assert fw._generate_new_hypothesis() is True  # blank retried, then success
    assert len(mgr._state.generated_hypotheses) == 1


def test_generation_all_attempts_fail_returns_false(tmp_path, monkeypatch):
    fw, mgr = _framework(tmp_path, monkeypatch, script=["raise", "raise", "raise"])
    assert fw._generate_new_hypothesis() is False
    assert len(mgr._state.generated_hypotheses) == 0  # nothing stored


def test_generate_new_hypotheses_skips_failures_without_crashing():
    # generate_new_hypotheses must count successes and skip failures.
    fw = fwmod.CoscientistFramework.__new__(fwmod.CoscientistFramework)
    outcomes = iter([True, False, True])
    advanced = {"n": 0}

    class _SM:
        def advance_hypothesis(self, kind): advanced["n"] += 1
        def update_proximity_graph_edges(self): pass

    fw.config = None
    fw.state_manager = _SM()
    fw._generate_new_hypothesis = lambda: next(outcomes)
    fw.process_reflection_queue = lambda: None

    produced = asyncio.run(fw.generate_new_hypotheses(n_hypotheses=3))
    assert produced == 2 and advanced["n"] == 2  # one failure skipped
