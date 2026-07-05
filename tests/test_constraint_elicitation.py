"""Offline tests for constraint auto-elicitation (killer feature)."""

import json

import pytest

from coscientist.common import load_prompt
from coscientist.constraint_elicitation import (
    ElicitedConstraint,
    elicit_constraints,
    merge_constraints,
    parse_elicited,
)

_ELICITED_JSON = json.dumps([
    {"text": "Целевой металл — никель", "rationale": "определяет KPI", "kind": "assumption"},
    {"text": "Без нового капитального оборудования", "rationale": "промышленный контекст", "kind": "assumption"},
    {"text": "Уточнить бюджет на реагенты", "rationale": "влияет на выбор", "kind": "clarification"},
], ensure_ascii=False)


class FakeLLM:
    def __init__(self, reply=_ELICITED_JSON):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        class M:
            pass
        m = M()
        m.content = self.reply
        return m


def test_parse_elicited_objects_and_strings():
    items = parse_elicited(_ELICITED_JSON)
    assert len(items) == 3
    assert items[0].kind == "assumption" and items[2].kind == "clarification"
    # bare strings and code fences tolerated
    fenced = "```json\n[\"no new mills\", {\"text\": \"MgO limit\"}]\n```"
    got = parse_elicited(fenced)
    assert [g.text for g in got] == ["no new mills", "MgO limit"]
    assert parse_elicited("garbage") == []


def test_elicit_constraints_uses_prompt_and_parses():
    llm = FakeLLM()
    out = elicit_constraints("Снизить потери никеля", "Действующая схема", retriever=None, llm=llm)
    assert len(out) == 3
    assert "Действующая схема" in llm.prompts[0]  # user constraints in the prompt
    assert "Снизить потери никеля" in llm.prompts[0]


def test_elicit_never_raises_on_bad_llm():
    class Boom:
        def invoke(self, p):
            raise RuntimeError("down")
    assert elicit_constraints("g", "", retriever=None, llm=Boom()) == []


def test_merge_constraints_folds_assumptions():
    elicited = parse_elicited(_ELICITED_JSON)
    merged = merge_constraints("Действующая схема флотации", elicited)
    assert "Действующая схема флотации" in merged
    assert "assumed constraints" in merged.lower()
    assert "Целевой металл — никель" in merged
    assert "clarifications" in merged.lower()  # clarification section present


def test_prompt_renders():
    p = load_prompt("constraint_elicitation", goal="g", constraints="c", corpus_context="ctx")
    assert "JSON array" in p and "assumption" in p


# ---------------------------------------------------------------------------
# run_deep integration (killer feature wired end-to-end, offline)
# ---------------------------------------------------------------------------


class _FakeFramework:
    def __init__(self, config, manager):
        self.config = config
        self.manager = manager

    async def run(self, n_hypotheses=4):
        from coscientist.hypothesis_assessment import HypothesisAssessment
        self.manager.set_assessments([HypothesisAssessment(hypothesis="H", overall_score=6.0)])
        return ("report", "meta")


def test_run_deep_elicits_and_records(tmp_path, monkeypatch):
    import coscientist.global_state as gs
    monkeypatch.setattr(gs, "_OUTPUT_DIR", str(tmp_path / "c"))
    from coscientist.studio import RunSettings, StudioEngine, StudioStore

    engine = StudioEngine(StudioStore(str(tmp_path / "s")))
    events = []

    rec = engine.run_deep(
        goal="Снизить потери никеля", constraints="Действующая схема", use_web=False,
        settings=RunSettings(auto_elicit_constraints=True),
        llm_factory=lambda role, s: FakeLLM(), framework_factory=_FakeFramework,
        on_event=lambda m: events.append(m.agent),
    )
    assert rec.elicited_constraints and len(rec.elicited_constraints) == 3
    assert "assumed constraints" in rec.effective_constraints.lower()
    assert "Constraints" in events  # streamed to the UI


def test_run_deep_skips_elicitation_when_off(tmp_path, monkeypatch):
    import coscientist.global_state as gs
    monkeypatch.setattr(gs, "_OUTPUT_DIR", str(tmp_path / "c2"))
    from coscientist.studio import RunSettings, StudioEngine, StudioStore

    engine = StudioEngine(StudioStore(str(tmp_path / "s2")))
    rec = engine.run_deep(
        goal="g", constraints="orig", use_web=False,
        settings=RunSettings(auto_elicit_constraints=False),
        llm_factory=lambda role, s: FakeLLM(), framework_factory=_FakeFramework,
    )
    assert rec.elicited_constraints == []
    assert rec.effective_constraints == "orig"
