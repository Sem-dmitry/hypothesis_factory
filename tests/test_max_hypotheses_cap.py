# -*- coding: utf-8 -*-

"""
Tests for the hard hypothesis-pool cap (max_hypotheses).

The pure guard `_hypothesis_cap_override` redirects pool-growing actions once the
pool is full; the engine threads `RunSettings.max_hypotheses` into
`CoscientistConfig.max_total_hypotheses` (clamped to the seed count).
"""

from coscientist.framework import _hypothesis_cap_override
from coscientist.studio import RunSettings, StudioEngine, StudioStore


# --------------------------------------------------------------------------- #
# Pure guard
# --------------------------------------------------------------------------- #


def test_unlimited_never_overrides():
    assert _hypothesis_cap_override("generate_new_hypotheses", total=99, max_total=0, num_unranked=0) is None


def test_below_cap_no_override():
    assert _hypothesis_cap_override("generate_new_hypotheses", total=5, max_total=10, num_unranked=0) is None


def test_at_cap_generate_redirects_to_tournament_when_unranked():
    assert _hypothesis_cap_override(
        "generate_new_hypotheses", total=10, max_total=10, num_unranked=3
    ) == "run_tournament"


def test_at_cap_evolve_redirects_to_finish_when_all_ranked():
    assert _hypothesis_cap_override(
        "evolve_hypotheses", total=12, max_total=10, num_unranked=0
    ) == "finish"


def test_at_cap_non_growing_action_untouched():
    # run_tournament / run_meta_review / finish are not pool-growing → no override.
    for action in ("run_tournament", "run_meta_review", "finish", "expand_literature_review"):
        assert _hypothesis_cap_override(action, total=10, max_total=10, num_unranked=2) is None


# --------------------------------------------------------------------------- #
# Soft signal: supervisor state + prompt
# --------------------------------------------------------------------------- #


def test_supervisor_state_surfaces_hypotheses_budget(monkeypatch, tmp_path):
    import coscientist.global_state as gs
    from coscientist.common import load_prompt

    monkeypatch.setattr(gs, "_OUTPUT_DIR", str(tmp_path / "cosci"))
    manager = gs.CoscientistStateManager(gs.CoscientistState(goal="g"))

    capped = manager.next_supervisor_state(max_hypotheses=10)
    assert capped["hypotheses_budget"] == "0 of max 10"
    prompt = load_prompt("supervisor_decision", **capped)
    assert "Hypothesis pool budget" in prompt
    assert "0 of max 10" in prompt

    unlimited = manager.next_supervisor_state(max_hypotheses=0)
    assert unlimited["hypotheses_budget"] == "0 (no cap)"


# --------------------------------------------------------------------------- #
# Threading: RunSettings.max_hypotheses -> CoscientistConfig.max_total_hypotheses
# --------------------------------------------------------------------------- #


class _CapturingFramework:
    """Captures the config it was built with; sets a trivial result."""

    def __init__(self, config, manager):
        _CapturingFramework.seen_config = config
        self.manager = manager

    async def run(self, n_hypotheses: int = 4):
        from coscientist.hypothesis_assessment import HypothesisAssessment
        self.manager.set_assessments([HypothesisAssessment(hypothesis="H", overall_score=6.0)])
        return ("report", "meta")


class _FakeLLM:
    def invoke(self, prompt):
        class M:
            content = "ok"
        return M()


def _engine(monkeypatch, tmp_path):
    import coscientist.global_state as gs
    monkeypatch.setattr(gs, "_OUTPUT_DIR", str(tmp_path / "cosci"))
    return StudioEngine(StudioStore(str(tmp_path / "studio")))


def test_max_hypotheses_threads_and_clamps(monkeypatch, tmp_path):
    engine = _engine(monkeypatch, tmp_path)
    engine.run_deep(
        goal="Снизить потери никеля",
        settings=RunSettings(lite=False, num_hypotheses=4, max_hypotheses=10),
        llm_factory=lambda role, s: _FakeLLM(),
        framework_factory=_CapturingFramework, use_web=False,
    )
    assert _CapturingFramework.seen_config.max_total_hypotheses == 10


def test_max_hypotheses_clamped_to_seed(monkeypatch, tmp_path):
    engine = _engine(monkeypatch, tmp_path)
    engine.run_deep(
        goal="g", settings=RunSettings(lite=False, num_hypotheses=6, max_hypotheses=3),
        llm_factory=lambda role, s: _FakeLLM(),
        framework_factory=_CapturingFramework, use_web=False,
    )
    # cap must not sit below the seed count, else start() would exceed it
    assert _CapturingFramework.seen_config.max_total_hypotheses == 6


def test_max_hypotheses_zero_is_unlimited(monkeypatch, tmp_path):
    engine = _engine(monkeypatch, tmp_path)
    engine.run_deep(
        goal="g", settings=RunSettings(lite=False, num_hypotheses=4, max_hypotheses=0),
        llm_factory=lambda role, s: _FakeLLM(),
        framework_factory=_CapturingFramework, use_web=False,
    )
    assert _CapturingFramework.seen_config.max_total_hypotheses == 0
