"""
Tests for the richer, constraint-aware hypothesis assessment (customer features
1/4/5/6): causal chain, world practice, novelty-vs-input, constraint adherence,
economic + kinetics estimate, and constraints threaded through the pipeline.
Offline (fake LLM + fake retriever).
"""

import json

import pytest

from coscientist.common import load_prompt
from coscientist.hypothesis_assessment import (
    HypothesisAssessment,
    assess_hypothesis,
    assessment_from_data,
)

_RICH_JSON = {
    "justification": "grounded", "mechanism_of_influence": "mech",
    "novelty": "n", "novelty_score": 6, "feasibility_score": 8, "impact_score": 7,
    "risk_level": 3, "technical_risks": ["r"], "economic_risks": ["e"],
    "expected_value": "v", "target_kpi_impact": "+1.5% Ni", "verification_plan": ["step"],
    "confidence": 0.6,
    "causal_chain": "Nickel is lost because pentlandite surfaces are passivated.",
    "world_practice": "Industry uses attritioning + selective collectors [1].",
    "novelty_vs_input": "Not in the corpus; draws on an external patent.",
    "constraint_adherence": "Fits the existing flowsheet; no new mills.",
    "constraint_violations": ["may exceed reagent budget"],
    "economic_estimate": "≈ +1% Ni recovery vs modest reagent cost (rough).",
    "kinetics_note": "Longer conditioning time needed.",
}


class RecordingLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        class M:
            pass
        m = M()
        m.content = self.reply
        return m


def test_schema_has_new_fields_default_empty():
    a = HypothesisAssessment(hypothesis="h")
    assert a.causal_chain == "" and a.world_practice == "" and a.novelty_vs_input == ""
    assert a.constraint_adherence == "" and a.constraint_violations == []
    assert a.economic_estimate == "" and a.kinetics_note == ""


def test_old_dict_without_new_fields_loads():
    old = {"hypothesis": "h", "overall_score": 5.0, "mechanism_of_influence": "m"}
    a = HypothesisAssessment(**old)
    assert a.causal_chain == "" and a.constraint_violations == []


def test_assessment_from_data_parses_new_fields():
    a = assessment_from_data("H", _RICH_JSON)
    assert "passivated" in a.causal_chain
    assert "patent" in a.novelty_vs_input.lower()
    assert a.constraint_violations == ["may exceed reagent budget"]
    assert a.kinetics_note.startswith("Longer conditioning")


def test_assess_hypothesis_threads_constraints_and_parses():
    llm = RecordingLLM(json.dumps(_RICH_JSON, ensure_ascii=False))
    a = assess_hypothesis(
        "Доизмельчение хвостов", goal="Снизить потери никеля", retriever=None,
        llm=llm, constraints="Без замены основного оборудования; бюджет ограничен",
    )
    # constraints reached the prompt
    assert "Без замены основного оборудования" in llm.prompts[0]
    # and the rich fields parsed
    assert a.world_practice and a.causal_chain
    assert a.constraint_violations == ["may exceed reagent budget"]


def test_prompt_renders_with_constraints_and_new_keys():
    p = load_prompt("hypothesis_assessment", goal="g", hypothesis="h",
                    corpus_context="ctx", constraints="no new equipment")
    assert "no new equipment" in p
    for key in ("causal_chain", "world_practice", "novelty_vs_input",
                "constraint_adherence", "constraint_violations", "economic_estimate",
                "evidence_refs"):
        assert key in p


def test_export_report_includes_new_fields():
    from coscientist.export.report import render_markdown

    a = assessment_from_data("Доизмельчение хвостов", _RICH_JSON)
    a.overall_score = 6.5
    md = render_markdown([a], goal="g")
    assert "Причинно-следственная связь" in md
    assert "Мировая/промышленная практика" in md
    assert "Новизна относительно входных данных" in md
    assert "Соблюдение ограничений" in md
    assert "Экономическая оценка" in md


def test_state_and_generation_thread_constraints(tmp_path, monkeypatch):
    import coscientist.global_state as gs
    monkeypatch.setattr(gs, "_OUTPUT_DIR", str(tmp_path / "c"))
    state = gs.CoscientistState(goal="g", constraints="no new mills")
    mgr = gs.CoscientistStateManager(state)
    assert mgr.constraints == "no new mills"
    # generation state carries constraints for the prompt
    mgr._state.literature_review = {"subtopics": ["s"], "subtopic_reports": ["report"]}
    gstate = mgr.next_generation_state("independent")
    assert gstate["constraints"] == "no new mills"
