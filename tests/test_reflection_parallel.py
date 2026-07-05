# -*- coding: utf-8 -*-
"""Offline tests: reflection queue is processed concurrently (bounded pool).

No network — `build_deep_verification_agent` is stubbed with a fake agent that
sleeps and records how many verifications run at once, so we can prove real
parallelism, correct result application, and the sequential fallback.
"""

import os
import threading
import time

import pytest

from coscientist.custom_types import ParsedHypothesis


def _manager(tmp_path, goal="recover nickel"):
    import coscientist.global_state as gs

    monkey_dir = str(tmp_path / "cosci")
    gs._OUTPUT_DIR = monkey_dir
    os.makedirs(monkey_dir, exist_ok=True)
    return gs.CoscientistStateManager(gs.CoscientistState(goal=goal))


def _config(**kw):
    from coscientist.framework import CoscientistConfig

    return CoscientistConfig(
        literature_review_agent_llm=object(),
        generation_agent_llms={"m1": object()},
        reflection_agent_llms={"m1": object()},
        evolution_agent_llms={"m1": object()},
        meta_review_agent_llm=object(),
        supervisor_agent_llm=object(),
        final_report_agent_llm=object(),
        proximity_agent_embedding_model=object(),
        specialist_fields=["flotation"],
        **kw,
    )


def _prime_queue(mgr, n):
    for i in range(n):
        mgr._state.reflection_queue.append(
            ParsedHypothesis(hypothesis=f"h{i}", predictions=["p"], assumptions=["a"])
        )


class _ConcurrencyProbe:
    """Fake deep-verification agent factory that tracks concurrent invokes."""

    def __init__(self, sleep=0.05, pass_filter=True):
        self.sleep = sleep
        self.pass_filter = pass_filter
        self.active = 0
        self.max_active = 0
        self.count = 0
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        probe = self

        class _Agent:
            def invoke(self, state):
                with probe._lock:
                    probe.active += 1
                    probe.count += 1
                    probe.max_active = max(probe.max_active, probe.active)
                try:
                    time.sleep(probe.sleep)
                finally:
                    with probe._lock:
                        probe.active -= 1
                hyp = state["hypothesis_to_review"]
                return {
                    "passed_initial_filter": probe.pass_filter,
                    "reviewed_hypothesis": hyp,
                }

        return _Agent()


def test_reflection_runs_in_parallel(monkeypatch, tmp_path):
    import coscientist.framework as fwmod

    probe = _ConcurrencyProbe(sleep=0.06)
    monkeypatch.setattr(fwmod, "build_deep_verification_agent", probe)
    # add_reviewed/advance must not choke on the fake ReviewedHypothesis
    applied = []
    monkeypatch.setattr(fwmod.CoscientistStateManager, "add_reviewed_hypothesis",
                        lambda self, rh: applied.append(rh), raising=True)
    monkeypatch.setattr(fwmod.CoscientistStateManager, "advance_reviewed_hypothesis",
                        lambda self: None, raising=True)

    mgr = _manager(tmp_path)
    _prime_queue(mgr, 6)
    fw = fwmod.CoscientistFramework(_config(reflection_concurrency=4), state_manager=mgr)

    t0 = time.time()
    fw.process_reflection_queue()
    elapsed = time.time() - t0

    assert probe.count == 6                     # every hypothesis verified
    assert len(applied) == 6                    # every result applied
    assert probe.max_active >= 2                # genuinely concurrent
    assert mgr.reflection_queue_is_empty        # queue drained
    # 6 tasks x 0.06s with 4 workers -> ~0.12s, far below the 0.36s sequential floor.
    assert elapsed < 0.30


def test_reflection_sequential_when_concurrency_one(monkeypatch, tmp_path):
    import coscientist.framework as fwmod

    probe = _ConcurrencyProbe(sleep=0.01)
    monkeypatch.setattr(fwmod, "build_deep_verification_agent", probe)
    monkeypatch.setattr(fwmod.CoscientistStateManager, "add_reviewed_hypothesis",
                        lambda self, rh: None, raising=True)
    monkeypatch.setattr(fwmod.CoscientistStateManager, "advance_reviewed_hypothesis",
                        lambda self: None, raising=True)

    mgr = _manager(tmp_path)
    _prime_queue(mgr, 4)
    fw = fwmod.CoscientistFramework(_config(reflection_concurrency=1), state_manager=mgr)
    fw.process_reflection_queue()

    assert probe.count == 4
    assert probe.max_active == 1                # strictly one at a time


def test_failed_verification_skips_not_aborts(monkeypatch, tmp_path):
    import coscientist.framework as fwmod

    class _Boom:
        def __call__(self, *a, **k):
            class _Agent:
                def invoke(self, state):
                    raise RuntimeError("web down")
            return _Agent()

    monkeypatch.setattr(fwmod, "build_deep_verification_agent", _Boom())
    issues = []
    monkeypatch.setattr(fwmod.CoscientistStateManager, "record_execution_issue",
                        lambda self, msg: issues.append(msg), raising=True)

    mgr = _manager(tmp_path)
    _prime_queue(mgr, 3)
    fw = fwmod.CoscientistFramework(_config(reflection_concurrency=3), state_manager=mgr)
    fw.process_reflection_queue()  # must not raise

    assert len(issues) == 3
    assert mgr.reflection_queue_is_empty


def test_config_default_concurrency():
    cfg = _config()
    assert cfg.reflection_concurrency == 4
    assert _config(reflection_concurrency=0).reflection_concurrency == 1  # clamped
