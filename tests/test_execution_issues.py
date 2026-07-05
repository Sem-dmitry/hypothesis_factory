"""
Failures/skips must be surfaced to the supervisor (planner), mirroring the
for_inspiration project that renders failed-subtask status into its planner
prompt so it can compensate.
"""

import coscientist.global_state as gs
from coscientist.common import load_prompt


def _manager(tmp_path, monkeypatch, goal="Снизить потери никеля"):
    monkeypatch.setattr(gs, "_OUTPUT_DIR", str(tmp_path / "cosci"))
    return gs.CoscientistStateManager(gs.CoscientistState(goal=goal))


def test_record_and_summarize_execution_issues(tmp_path, monkeypatch):
    mgr = _manager(tmp_path, monkeypatch)
    assert mgr.execution_issues_summary() == "None"

    mgr.record_execution_issue("Generation skipped")
    mgr.record_execution_issue("Generation skipped")
    mgr.record_execution_issue("Meta-review failed (placeholder)")

    summary = mgr.execution_issues_summary()
    assert "Generation skipped: 2" in summary
    assert "Meta-review failed (placeholder): 1" in summary


def test_tournament_fallbacks_folded_into_summary(tmp_path, monkeypatch):
    mgr = _manager(tmp_path, monkeypatch)
    mgr._state.tournament.fallback_count = 3
    assert "Tournament match fallback (random winner): 3" in mgr.execution_issues_summary()


def test_backward_compat_missing_field(tmp_path, monkeypatch):
    mgr = _manager(tmp_path, monkeypatch)
    del mgr._state.execution_issues  # simulate an old pickle
    assert mgr.execution_issues == {}
    mgr.record_execution_issue("x")
    assert mgr.execution_issues == {"x": 1}


def test_supervisor_state_and_prompt_surface_issues(tmp_path, monkeypatch):
    mgr = _manager(tmp_path, monkeypatch)
    mgr.record_execution_issue("Generation skipped")

    st = mgr.next_supervisor_state()
    assert "Generation skipped" in st["execution_issues"]

    prompt = load_prompt("supervisor_decision", **st)
    assert "Execution Issues" in prompt
    assert "Generation skipped" in prompt
