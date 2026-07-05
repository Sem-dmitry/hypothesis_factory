"""
Tests for the deterministic run-termination guard (P1).

`_termination_reason` decides whether `CoscientistFramework.run()` must finish
now, without consulting the LLM supervisor. This closes the observed failure
mode where the supervisor loops on `evolve_hypotheses` (each blocks on the
reflection queue) and never reaches finish() / Assessor / Final report.
"""

from coscientist.framework import _termination_reason

BUDGET = 40
K = 3


def _reason(**kw):
    base = dict(
        has_clear_leader=False,
        is_plateau=False,
        recent_actions=[],
        num_actions=5,
        budget=BUDGET,
        max_consecutive_evolve=K,
    )
    base.update(kw)
    return _termination_reason(**base)


def test_keep_going_when_nothing_triggers():
    assert _reason(recent_actions=["run_tournament", "evolve_hypotheses"]) is None


def test_budget_spent_forces_finish():
    assert "budget" in _reason(num_actions=BUDGET).lower()
    assert "budget" in _reason(num_actions=BUDGET + 3).lower()


def test_settled_tournament_forces_finish():
    r = _reason(has_clear_leader=True, is_plateau=True)
    assert r and "plateau" in r.lower()


def test_leader_without_plateau_keeps_going():
    assert _reason(has_clear_leader=True, is_plateau=False) is None


def test_plateau_without_leader_keeps_going():
    assert _reason(has_clear_leader=False, is_plateau=True) is None


def test_runaway_evolve_forces_finish():
    r = _reason(recent_actions=["evolve_hypotheses"] * K)
    assert r and "evolve" in r.lower()


def test_two_consecutive_evolves_not_enough():
    assert _reason(recent_actions=["evolve_hypotheses", "evolve_hypotheses"]) is None


def test_mixed_recent_actions_not_runaway():
    # Last K contains a non-evolve action → not a runaway loop.
    assert _reason(
        recent_actions=["evolve_hypotheses", "generate_new_hypotheses", "evolve_hypotheses"]
    ) is None


def test_runaway_guard_disabled_when_zero():
    assert _reason(recent_actions=["evolve_hypotheses"] * 5, max_consecutive_evolve=0) is None


def test_budget_takes_priority_reason():
    # Even if a runaway loop is also present, a spent budget is reported.
    r = _reason(num_actions=BUDGET, recent_actions=["evolve_hypotheses"] * K)
    assert "budget" in r.lower()
