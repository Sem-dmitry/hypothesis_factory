# -*- coding: utf-8 -*-

"""
Offline tests for Studio deep mode (full-framework run wired into the Studio).

No network, no real LLMs, no web search, no gpt_researcher: the framework is
replaced by an injected fake and the agent LLMs by a fake factory.
"""

import json
import os

import pytest

from coscientist.hypothesis_assessment import HypothesisAssessment
from coscientist.studio import (
    RunRecord,
    RunSettings,
    StudioEngine,
    StudioStore,
    TranscriptMessage,
    _RecordingLLM,
)


class _FakeLLM:
    def __init__(self, reply="ok"):
        self.reply = reply

    def invoke(self, prompt):
        class M:
            pass
        m = M()
        m.content = self.reply
        return m


# ---------------------------------------------------------------------------
# AC1: RunRecord deep fields, backward compatible
# ---------------------------------------------------------------------------


def test_runrecord_deep_fields_default():
    rec = RunRecord(id="x", project="p", goal="g", created_at="t", settings=RunSettings())
    assert rec.mode == "fast"
    assert rec.final_report == "" and rec.meta_review == ""
    assert rec.tournament_summary == {}
    # old JSON without the new fields still loads
    old = json.dumps({"id": "y", "project": "p", "goal": "g", "created_at": "t",
                      "settings": RunSettings().model_dump()})
    loaded = RunRecord.model_validate_json(old)
    assert loaded.mode == "fast"


# ---------------------------------------------------------------------------
# AC2: recording proxy delegates unknown attributes
# ---------------------------------------------------------------------------


def test_recording_llm_delegates_and_records():
    class Rich:
        flavor = "vanilla"
        def invoke(self, p):
            class M: content = "resp"
            return M()
        def with_structured_output(self, schema):
            return "bound"

    sink = []
    proxy = _RecordingLLM(Rich(), agent="Generator", model_name="m", sink=sink)
    # delegated attribute + method
    assert proxy.flavor == "vanilla"
    assert proxy.with_structured_output(dict) == "bound"
    # invoke is recorded
    out = proxy.invoke("hello")
    assert out.content == "resp"
    assert len(sink) == 1 and sink[0].agent == "Generator" and sink[0].tokens_out > 0


# ---------------------------------------------------------------------------
# deep run with an injected fake framework
# ---------------------------------------------------------------------------


class _FakeFramework:
    """Stands in for CoscientistFramework: touches agent LLMs + sets state."""

    def __init__(self, config, manager):
        self.config = config
        self.manager = manager

    async def run(self, n_hypotheses: int = 4):
        self.n_hypotheses = n_hypotheses  # capture what the engine passed
        # exercise a couple of agent LLMs so the transcript records them
        self.config.generation_agent_llms["studio"].invoke("generate hypotheses")
        self.config.assessment_llm.invoke("assess")
        self.manager.set_assessments([
            HypothesisAssessment(hypothesis="Доизмельчение хвостов", overall_score=6.5,
                                 citations=["[1] report.pdf — p.1 (PDF)"]),
            HypothesisAssessment(hypothesis="Смена собирателя", overall_score=5.0),
        ])
        return ("ФИНАЛЬНЫЙ ОТЧЁТ", "МЕТА-РЕВЬЮ")


def _tmp_state_dir(monkeypatch, tmp_path):
    import coscientist.global_state as gs
    monkeypatch.setattr(gs, "_OUTPUT_DIR", str(tmp_path / "cosci"))


def test_run_deep_with_fake_framework(monkeypatch, tmp_path):
    _tmp_state_dir(monkeypatch, tmp_path)
    engine = StudioEngine(StudioStore(str(tmp_path / "studio")))

    captured = []

    def llm_factory(role, settings):
        captured.append((role, settings.generator_model, settings.generator_temperature,
                         settings.assessor_model, settings.assessor_temperature))
        return _FakeLLM()

    settings = RunSettings(lite=False, project="Хвосты", generator_model="google/gemini-2.5-pro",
                           generator_temperature=1.0, assessor_model="google/gemini-2.5-flash",
                           assessor_temperature=0.2)
    record = engine.run_deep(
        goal="Снизить потери никеля", constraints="демо", settings=settings,
        llm_factory=llm_factory, framework_factory=_FakeFramework, use_web=False,
    )

    assert record.mode == "deep"
    assert record.final_report == "ФИНАЛЬНЫЙ ОТЧЁТ"
    assert record.meta_review == "МЕТА-РЕВЬЮ"
    assert len(record.assessments) == 2
    # transcript recorded the agent invokes
    assert record.metrics["messages"] >= 2
    assert record.metrics["seconds_wall"] >= 0
    # persisted + reloadable
    reloaded = engine.store.load_run(record.id)
    assert reloaded.mode == "deep" and len(reloaded.assessments) == 2
    # AC4: settings reached the llm factory
    assert any(c[1] == "google/gemini-2.5-pro" for c in captured)


def test_ui_settings_propagate_to_deep_run(monkeypatch, tmp_path):
    """num_hypotheses reaches framework.run, and per-agent models reach the LLM factory."""
    from coscientist.studio import AgentLLMSettings

    _tmp_state_dir(monkeypatch, tmp_path)
    engine = StudioEngine(StudioStore(str(tmp_path / "studio")))

    made = {}

    def framework_factory(config, manager):
        fw = _FakeFramework(config, manager)
        made["fw"] = fw
        return fw

    seen_roles = {}

    def llm_factory(role, settings):
        seen_roles[role] = settings.agent_model_temp(role)
        return _FakeLLM()

    settings = RunSettings(
        lite=False,
        num_hypotheses=6,
        agents={
            "reflection": AgentLLMSettings(model="anthropic/claude-sonnet-4", temperature=0.15),
            "generation": AgentLLMSettings(model="google/gemini-2.5-flash", temperature=0.9),
        },
    )
    engine.run_deep(goal="g", settings=settings, llm_factory=llm_factory,
                    framework_factory=framework_factory, use_web=False)

    # num_hypotheses propagated into framework.run
    assert made["fw"].n_hypotheses == 6
    # per-agent non-default models propagated
    assert seen_roles["reflection"] == ("anthropic/claude-sonnet-4", 0.15)
    assert seen_roles["generation"] == ("google/gemini-2.5-flash", 0.9)


def test_run_deep_emits_progress_events(monkeypatch, tmp_path):
    _tmp_state_dir(monkeypatch, tmp_path)
    engine = StudioEngine(StudioStore(str(tmp_path / "studio")))

    events = []
    engine.run_deep(
        goal="Снизить потери никеля", settings=RunSettings(),
        llm_factory=lambda role, s: _FakeLLM(), framework_factory=_FakeFramework,
        use_web=False, on_event=lambda m: events.append(m.agent),
    )
    # the fake framework touches the generation + assessment agent LLMs
    assert len(events) >= 2
    assert "Literature" in events
    assert all(isinstance(a, str) for a in events)


def test_run_deep_use_web_false_seeds_no_network(monkeypatch, tmp_path):
    # use_web=False must not require gpt_researcher/web search; the fake framework
    # never does web work, and run_deep seeds a corpus-only literature review.
    _tmp_state_dir(monkeypatch, tmp_path)
    engine = StudioEngine(StudioStore(str(tmp_path / "studio")))
    captured = {}

    def framework_factory(config, manager):
        captured["web_research_enabled"] = config.web_research_enabled
        return _FakeFramework(config, manager)

    rec = engine.run_deep(
        goal="Извлечение меди", settings=RunSettings(),
        llm_factory=lambda role, s: _FakeLLM(), framework_factory=framework_factory,
        use_web=False,
    )
    assert rec.status == "completed" and rec.mode == "deep"
    assert captured["web_research_enabled"] is False
    assert any(t.agent == "Literature" and t.model == "corpus-only" for t in rec.transcript)
