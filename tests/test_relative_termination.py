"""
Tests for relative/adaptive tournament termination (no absolute Elo thresholds).

Signals adapt to the run's own rating distribution + trajectory, so a converged
run on a light model (max Elo well below the old 1400 gate) is recognized as
done, while a still-climbing run is not — and a soft action budget guarantees
termination.
"""

import asyncio
import os

import pytest

from coscientist.ranking_agent import (
    K_FACTOR,
    LEADER_GAP_K,
    EloTournament,
)


def _tourney(rounds):
    t = EloTournament("goal")
    t._past_tournament_ratings = [list(r) for r in rounds]
    return t


# ---------------------------------------------------------------------------
# AC1: relative signals
# ---------------------------------------------------------------------------


def test_light_model_converged_is_recognized():
    # Max ≈ 1290 (BELOW the old 1400 gate) but clearly separated + flat.
    rounds = [
        [1280, 1200, 1190, 1180],
        [1288, 1200, 1188, 1176],
        [1290, 1200, 1186, 1174],
    ]
    s = _tourney(rounds).relative_trajectory_signals()
    assert s["leader_gap"] >= LEADER_GAP_K * K_FACTOR       # separated from the pack
    assert s["has_clear_leader"] is True
    assert s["is_plateau"] is True                          # stopped climbing
    assert s["max_elo_delta_recent"] < K_FACTOR


def test_still_climbing_is_not_plateau():
    rounds = [[1240, 1200, 1180], [1300, 1200, 1170], [1372, 1200, 1160]]
    s = _tourney(rounds).relative_trajectory_signals()
    assert s["is_plateau"] is False                         # top still rising fast
    assert s["max_elo_delta_recent"] >= K_FACTOR
    assert s["has_clear_leader"] is True                    # gap 172 is large


def test_undifferentiated_pack_has_no_clear_leader():
    rounds = [[1210, 1205, 1200, 1195], [1212, 1206, 1200, 1194]]
    s = _tourney(rounds).relative_trajectory_signals()
    assert s["has_clear_leader"] is False                   # gap ≈ 12 < threshold


def test_empty_and_single_round_safe_defaults():
    empty = EloTournament("g").relative_trajectory_signals()
    assert empty["rounds"] == 0 and empty["is_plateau"] is False and empty["has_clear_leader"] is False
    single = _tourney([[1300, 1200, 1100]]).relative_trajectory_signals()
    assert single["is_plateau"] is False                    # need ≥2 rounds to call a plateau
    assert single["leader_gap"] == 100.0


def test_no_absolute_1400_in_prompt():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "coscientist", "prompts", "supervisor_decision.md")
    text = open(p, encoding="utf-8").read()
    assert "1400" not in text and "1300" not in text


# ---------------------------------------------------------------------------
# AC2: signals surfaced to the supervisor state
# ---------------------------------------------------------------------------


def _manager(tmp_path, goal="Снизить потери никеля"):
    import coscientist.global_state as gs
    gs._OUTPUT_DIR = str(tmp_path / "cosci")
    return gs.CoscientistStateManager(gs.CoscientistState(goal=goal))


def test_next_supervisor_state_exposes_relative_signals(tmp_path, monkeypatch):
    import coscientist.global_state as gs
    monkeypatch.setattr(gs, "_OUTPUT_DIR", str(tmp_path / "c"))
    mgr = gs.CoscientistStateManager(gs.CoscientistState(goal="g"))
    mgr._state.tournament._past_tournament_ratings = [
        [1280, 1200, 1190], [1290, 1200, 1188], [1291, 1200, 1186],
    ]
    st = mgr.next_supervisor_state(max_actions=12)
    assert st["has_clear_leader"] == "yes"
    assert st["is_plateau"] == "yes"
    assert "budget" in st["actions_budget"] and "12" in st["actions_budget"]
    assert float(st["leader_gap"]) >= 64  # top separated from the pack


# ---------------------------------------------------------------------------
# AC4: soft action budget guarantees termination
# ---------------------------------------------------------------------------


class _FakeLLM:
    def invoke(self, prompt):
        class M:
            content = "ok"
        return M()


def test_budget_forces_finish(tmp_path, monkeypatch):
    import coscientist.global_state as gs
    monkeypatch.setattr(gs, "_OUTPUT_DIR", str(tmp_path / "b"))
    from coscientist.framework import CoscientistConfig, CoscientistFramework

    mgr = gs.CoscientistStateManager(gs.CoscientistState(goal="g"))
    # Make it look "already started" so run() skips start(), and pre-spend budget.
    mgr._state.meta_reviews = [{"result": "m"}]
    mgr._state.actions = ["run_tournament", "run_meta_review", "run_tournament"]

    cfg = CoscientistConfig(
        generation_agent_llms={"m": _FakeLLM()}, reflection_agent_llms={"m": _FakeLLM()},
        evolution_agent_llms={"m": _FakeLLM()}, meta_review_agent_llm=_FakeLLM(),
        supervisor_agent_llm=_FakeLLM(), final_report_agent_llm=_FakeLLM(),
        literature_review_agent_llm=_FakeLLM(), proximity_agent_embedding_model=object(),
        max_supervisor_actions=3,
    )
    fw = CoscientistFramework(cfg, mgr)

    async def _fake_finish():
        mgr._state.final_report = {"result": "forced final report"}

    fw.finish = _fake_finish  # avoid real assessment/report work

    report, meta = asyncio.run(fw.run())
    assert report == "forced final report"
    assert mgr._state.actions[-1] == "finish"   # budget forced a finish
