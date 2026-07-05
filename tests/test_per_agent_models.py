"""Offline tests for per-agent model/temperature configuration (Phase 8)."""

import asyncio
import json
import os

import pytest

from coscientist.studio import (
    DEFAULT_AGENT_MODELS,
    AgentLLMSettings,
    RunSettings,
    StudioEngine,
    StudioStore,
)


class _FakeLLM:
    def invoke(self, prompt):
        class M:
            content = "ok"
        return M()


# ---------------------------------------------------------------------------
# AC2: RunSettings agent defaults + backward compatibility
# ---------------------------------------------------------------------------


def test_runsettings_agent_defaults():
    s = RunSettings()
    assert set(s.agents.keys()) == set(DEFAULT_AGENT_MODELS.keys())
    assert s.agents["generation"].temperature == 1.0
    assert s.agents["ranking"].model == "google/gemini-2.5-flash"


def test_runsettings_old_record_loads():
    # a persisted record from before per-agent settings existed
    old = json.dumps({"project": "p", "generator_model": "openai/o3"})
    s = RunSettings.model_validate_json(old)
    assert s.generator_model == "openai/o3"
    assert set(s.agents.keys()) == set(DEFAULT_AGENT_MODELS.keys())  # defaults applied


# ---------------------------------------------------------------------------
# AC3: per-agent resolution
# ---------------------------------------------------------------------------


def test_agent_model_temp_resolution():
    s = RunSettings(
        generator_model="openai/o3", generator_temperature=0.9,
        assessor_model="google/gemini-2.5-flash", assessor_temperature=0.1,
        agents={"reflection": AgentLLMSettings(model="anthropic/claude-sonnet-4", temperature=0.15)},
    )
    assert s.agent_model_temp("reflection") == ("anthropic/claude-sonnet-4", 0.15)
    assert s.agent_model_temp("generator") == ("openai/o3", 0.9)
    assert s.agent_model_temp("assessor") == ("google/gemini-2.5-flash", 0.1)


# ---------------------------------------------------------------------------
# AC1: ranking_agent_llm in the framework
# ---------------------------------------------------------------------------


def test_config_ranking_llm_defaults_to_meta_review():
    from coscientist.framework import CoscientistConfig

    meta = _FakeLLM()
    cfg = CoscientistConfig(
        generation_agent_llms={"m": _FakeLLM()}, reflection_agent_llms={"m": _FakeLLM()},
        evolution_agent_llms={"m": _FakeLLM()}, meta_review_agent_llm=meta,
        supervisor_agent_llm=_FakeLLM(), final_report_agent_llm=_FakeLLM(),
        literature_review_agent_llm=_FakeLLM(), proximity_agent_embedding_model=object(),
    )
    assert cfg.ranking_agent_llm is meta  # falls back to meta-review

    ranking = _FakeLLM()
    cfg2 = CoscientistConfig(
        generation_agent_llms={"m": _FakeLLM()}, reflection_agent_llms={"m": _FakeLLM()},
        evolution_agent_llms={"m": _FakeLLM()}, meta_review_agent_llm=meta,
        supervisor_agent_llm=_FakeLLM(), final_report_agent_llm=_FakeLLM(),
        literature_review_agent_llm=_FakeLLM(), proximity_agent_embedding_model=object(),
        ranking_agent_llm=ranking,
    )
    assert cfg2.ranking_agent_llm is ranking


def test_run_tournament_uses_ranking_llm():
    from coscientist.framework import CoscientistConfig, CoscientistFramework

    ranking = _FakeLLM()
    cfg = CoscientistConfig(
        generation_agent_llms={"m": _FakeLLM()}, reflection_agent_llms={"m": _FakeLLM()},
        evolution_agent_llms={"m": _FakeLLM()}, meta_review_agent_llm=_FakeLLM(),
        supervisor_agent_llm=_FakeLLM(), final_report_agent_llm=_FakeLLM(),
        literature_review_agent_llm=_FakeLLM(), proximity_agent_embedding_model=object(),
        ranking_agent_llm=ranking,
    )

    class _FakeManager:
        num_tournament_hypotheses = 4
        def __init__(self): self.captured = None
        def run_tournament(self, llm, k_bracket): self.captured = llm

    mgr = _FakeManager()
    fw = CoscientistFramework(cfg, mgr)
    asyncio.run(fw.run_tournament(k_bracket=2))
    assert mgr.captured is ranking


# ---------------------------------------------------------------------------
# AC4: run_deep threads distinct per-agent models
# ---------------------------------------------------------------------------


class _FakeFramework:
    def __init__(self, config, manager):
        self.config = config
        self.manager = manager

    async def run(self, n_hypotheses: int = 4):
        from coscientist.hypothesis_assessment import HypothesisAssessment
        self.manager.set_assessments([HypothesisAssessment(hypothesis="h", overall_score=6.0)])
        return ("report", "meta")


def test_run_deep_threads_per_agent_models(tmp_path, monkeypatch):
    import coscientist.global_state as gs
    monkeypatch.setattr(gs, "_OUTPUT_DIR", str(tmp_path / "cosci"))
    engine = StudioEngine(StudioStore(str(tmp_path / "studio")))

    seen = []

    def llm_factory(key, settings):
        seen.append((key, *settings.agent_model_temp(key)))
        return _FakeLLM()

    settings = RunSettings(
        lite=False,
        agents={**RunSettings().agents,
                "reflection": AgentLLMSettings(model="openai/o3", temperature=0.15)},
    )
    engine.run_deep(
        goal="цель", settings=settings, llm_factory=llm_factory,
        framework_factory=_FakeFramework, use_web=False,
    )
    seen_keys = {s[0] for s in seen}
    # all deep agents resolved, each by its own key
    for key in ("literature", "generation", "reflection", "ranking", "evolution",
                "meta_review", "supervisor", "final_report", "assessment"):
        assert key in seen_keys, key
    # the customized reflection model was used
    assert ("reflection", "openai/o3", 0.15) in seen


def test_agent_thinking_resolution():
    from coscientist.studio import AgentLLMSettings, RunSettings
    s = RunSettings(agents={
        "reflection": AgentLLMSettings(model="google/gemini-2.5-pro", temperature=0.2, thinking="high"),
    })
    assert s.agent_thinking("reflection") == "high"
    assert s.agent_thinking("generation") == "default"  # unset -> default
