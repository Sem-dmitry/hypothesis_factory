# -*- coding: utf-8 -*-
"""Offline tests for LITE / FULL mode.

LITE is the default: flash everywhere, reasoning off, fixed 4/8, reduced web
constants, top-3 assumptions. FULL keeps the configured settings.
No network / no real LLMs (framework + LLMs injected).
"""

import os

import pytest

from coscientist.studio import (
    LITE_MODEL,
    RunSettings,
    StudioEngine,
    StudioStore,
    _researcher_config_file,
)


class _FakeLLM:
    def invoke(self, prompt):
        class M:
            content = "ok"
        return M()


class _CaptureFramework:
    """Captures the config and what run() received."""

    def __init__(self, config, manager):
        self.config = config
        self.manager = manager

    async def run(self, n_hypotheses: int = 4):
        self.n_hypotheses = n_hypotheses
        from coscientist.hypothesis_assessment import HypothesisAssessment
        self.manager.set_assessments([HypothesisAssessment(hypothesis="h", overall_score=5.0)])
        return ("report", "meta")


def _tmp_state_dir(monkeypatch, tmp_path):
    import coscientist.global_state as gs
    monkeypatch.setattr(gs, "_OUTPUT_DIR", str(tmp_path / "cosci"))


# --------------------------------------------------------------------------- #
# AC1 — RunSettings.lite default + overrides
# --------------------------------------------------------------------------- #


def test_lite_is_default():
    assert RunSettings().lite is True


def test_lite_overrides():
    s = RunSettings(
        lite=True, num_hypotheses=7, max_hypotheses=25,
        generator_model="google/gemini-2.5-pro", assessor_model="google/gemini-2.5-pro",
    )
    ov = s.lite_overrides()
    assert ov.num_hypotheses == 4 and ov.max_hypotheses == 8
    assert ov.generator_model == LITE_MODEL and ov.assessor_model == LITE_MODEL
    assert ov.web_search_model == LITE_MODEL
    assert ov.agents  # every agent forced to flash + reasoning off
    for a in ov.agents.values():
        assert a.model == LITE_MODEL and a.thinking == "off"
    # original untouched
    assert s.num_hypotheses == 7 and s.generator_model == "google/gemini-2.5-pro"


# --------------------------------------------------------------------------- #
# AC3 — researcher config selection
# --------------------------------------------------------------------------- #


def test_researcher_config_file_paths():
    lite = _researcher_config_file(True)
    full = _researcher_config_file(False)
    assert lite.endswith("researcher_config_lite.json") and os.path.exists(lite)
    assert full.endswith("researcher_config.json") and os.path.exists(full)


def test_researcher_config_path_honors_env(monkeypatch, tmp_path):
    from coscientist.web_search import researcher_config_path
    monkeypatch.delenv("COSCIENTIST_RESEARCHER_CONFIG", raising=False)
    assert researcher_config_path().endswith("researcher_config.json")
    p = tmp_path / "cfg.json"; p.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("COSCIENTIST_RESEARCHER_CONFIG", str(p))
    assert researcher_config_path() == str(p)


def test_lite_config_has_reduced_constants():
    import json
    lite = json.load(open(_researcher_config_file(True), encoding="utf-8"))
    assert lite["MAX_ITERATIONS"] == 1
    assert lite["MAX_SEARCH_RESULTS_PER_QUERY"] == 3
    assert lite["MAX_SUBTOPICS"] == 2


# --------------------------------------------------------------------------- #
# AC2/AC4 — run_deep applies mode end to end
# --------------------------------------------------------------------------- #


def test_run_deep_lite_applies_everything(monkeypatch, tmp_path):
    _tmp_state_dir(monkeypatch, tmp_path)
    monkeypatch.setenv("COSCIENTIST_RESEARCHER_CONFIG", "sentinel")
    engine = StudioEngine(StudioStore(str(tmp_path / "studio")))

    made = {}
    def framework_factory(config, manager):
        fw = _CaptureFramework(config, manager); made["fw"] = fw; return fw
    resolved = {}
    def llm_factory(role, settings):
        resolved[role] = settings.agent_model_temp(role)[0]
        return _FakeLLM()

    # Ask for pro + 6 hypotheses, but LITE (default) must override all of it.
    settings = RunSettings(num_hypotheses=6, generator_model="google/gemini-2.5-pro")
    rec = engine.run_deep(goal="g", settings=settings, llm_factory=llm_factory,
                          framework_factory=framework_factory, use_web=False)

    fw = made["fw"]
    assert fw.n_hypotheses == 4                              # fixed
    assert fw.config.literature_subtopics == 2              # reduced breadth
    assert fw.config.max_assumptions_researched == 3        # top-K
    assert resolved.get("generation") == LITE_MODEL         # flash, not pro
    assert os.environ["COSCIENTIST_RESEARCHER_CONFIG"].endswith("researcher_config_lite.json")
    assert rec.settings.num_hypotheses == 4                 # effective settings stored
    assert rec.settings.lite is True


def test_run_deep_full_mode_keeps_settings(monkeypatch, tmp_path):
    _tmp_state_dir(monkeypatch, tmp_path)
    engine = StudioEngine(StudioStore(str(tmp_path / "studio")))

    made = {}
    def framework_factory(config, manager):
        fw = _CaptureFramework(config, manager); made["fw"] = fw; return fw
    resolved = {}
    def llm_factory(role, settings):
        resolved[role] = settings.agent_model_temp(role)[0]
        return _FakeLLM()

    from coscientist.studio import AgentLLMSettings
    settings = RunSettings(
        lite=False, num_hypotheses=6, max_hypotheses=20,
        agents={"generation": AgentLLMSettings(model="google/gemini-2.5-pro", temperature=1.0)},
    )
    rec = engine.run_deep(goal="g", settings=settings, llm_factory=llm_factory,
                          framework_factory=framework_factory, use_web=False)

    fw = made["fw"]
    assert fw.n_hypotheses == 6
    assert fw.config.literature_subtopics == 5
    assert fw.config.max_assumptions_researched == 0        # all assumptions
    assert resolved.get("generation") == "google/gemini-2.5-pro"
    assert os.environ["COSCIENTIST_RESEARCHER_CONFIG"].endswith("researcher_config.json")
    assert rec.settings.num_hypotheses == 6


# --------------------------------------------------------------------------- #
# AC4 — top-K assumption cap in the decomposer
# --------------------------------------------------------------------------- #


def test_assumption_decomposer_caps_top_k(monkeypatch):
    import coscientist.reflection_agent as ra
    from coscientist.custom_types import ParsedHypothesis

    monkeypatch.setattr(ra, "parse_assumption_decomposition",
                        lambda text: {f"a{i}": [f"s{i}"] for i in range(6)})
    state = {"hypothesis_to_review": ParsedHypothesis(
        hypothesis="h", predictions=["p"], assumptions=["a"])}

    out_all = ra.assumption_decomposer_node(state, _FakeLLM(), max_assumptions=0)
    assert len(out_all["_parsed_assumptions"]) == 6           # all

    out_k = ra.assumption_decomposer_node(state, _FakeLLM(), max_assumptions=3)
    assert len(out_k["_parsed_assumptions"]) == 3             # top-K only
    assert list(out_k["_parsed_assumptions"]) == ["a0", "a1", "a2"]
