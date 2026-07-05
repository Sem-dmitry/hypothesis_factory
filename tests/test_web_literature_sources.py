# -*- coding: utf-8 -*-

import json
import os
import uuid

from coscientist.global_state import CoscientistState, CoscientistStateManager
from coscientist.hypothesis_assessment import (
    AssessmentWeights,
    HypothesisAssessment,
    assess_hypothesis,
)
from coscientist.web_literature import (
    WebLiteratureSource,
    append_web_references_to_review,
    extract_web_literature_sources,
)


class _RecordingLLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)

        class M:
            content = json.dumps(self.payload)

        return M()


def test_extract_web_literature_sources_from_markdown_and_plain_urls():
    text = (
        "Surface oxidation is discussed in "
        "[Pentlandite flotation review](https://example.org/pentlandite-review).\n"
        "A second source is available at https://doi.org/10.1234/nicu.2024."
    )
    sources = extract_web_literature_sources(text)
    assert [s.url for s in sources] == [
        "https://example.org/pentlandite-review",
        "https://doi.org/10.1234/nicu.2024",
    ]
    assert sources[0].title == "Pentlandite flotation review"
    assert "Surface oxidation" in sources[0].snippet


def test_generation_literature_review_gets_normalized_web_refs():
    state = CoscientistState(goal=f"Reduce nickel losses {uuid.uuid4()}")
    state.literature_review = {
        "subtopics": ["surface chemistry"],
        "subtopic_reports": [
            "Relevant paper: [Citric acid and iron oxides](https://example.com/citric)."
        ],
    }
    manager = CoscientistStateManager(state)
    gen_state = manager.next_generation_state("independent")
    assert "[W1] Citric acid and iron oxides" in gen_state["literature_review"]
    assert "https://example.com/citric" in gen_state["literature_review"]


def test_append_web_references_leaves_text_without_urls_unchanged():
    review = "No web references here."
    assert append_web_references_to_review(review) == review


def test_assessment_can_select_web_literature_evidence_without_rag():
    payload = {
        "justification": "Supported by web literature.",
        "mechanism_of_influence": "Citric acid chelates iron films.",
        "novelty": "Specific use in Ni flotation.",
        "novelty_score": 7,
        "feasibility_score": 6,
        "impact_score": 6,
        "risk_level": 4,
        "technical_risks": [],
        "economic_risks": [],
        "expected_value": "More nickel recovery.",
        "target_kpi_impact": "Lower Ni losses.",
        "verification_plan": ["bench flotation"],
        "confidence": 0.6,
        "evidence_refs": [],
        "web_evidence_refs": [1],
    }
    llm = _RecordingLLM(payload)
    source = WebLiteratureSource(
        title="Citric acid surface cleaning",
        url="https://example.com/citric",
        snippet="Citric acid dissolves iron oxide films on sulfide minerals.",
    )
    assessment = assess_hypothesis(
        "Use citric acid pre-conditioning before nickel flotation",
        goal="Reduce nickel losses to tailings",
        retriever=None,
        llm=llm,
        web_sources=[source],
    )
    assert "[W1] Citric acid surface cleaning" in llm.prompts[0]
    assert assessment.web_evidence_refs == [1]
    assert len(assessment.source_evidence) == 1
    ev = assessment.source_evidence[0]
    assert ev.source_type == "web"
    assert ev.url == "https://example.com/citric"
    assert ev.chunk_id == ""


def test_webapp_source_html_supports_external_web_links():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_js = open(os.path.join(root, "webapp", "static", "app.js"), encoding="utf-8").read()
    assert 'class="source-link"' in app_js
    assert 'target="_blank"' in app_js
    assert 'data-chunk="${escAttr(s.chunk_id || \'\')}"' in app_js


def test_framework_assessment_passes_literature_web_sources(monkeypatch):
    import coscientist.framework as framework_mod

    captured = {}

    def fake_assess_hypothesis(hypothesis, **kwargs):
        captured["web_sources"] = kwargs.get("web_sources")
        return HypothesisAssessment(hypothesis=hypothesis, overall_score=5.0)

    monkeypatch.setattr(framework_mod, "assess_hypothesis", fake_assess_hypothesis)

    class Config:
        retriever = None
        assessment_weights = AssessmentWeights()
        assessment_llm = object()
        supervisor_agent_llm = object()

    class Manager:
        goal = "Reduce nickel losses"
        constraints = "existing equipment"
        literature_review_reports = [
            "Literature source: [Ni-Cu flotation](https://example.com/nicu)."
        ]

        def top_tournament_hypotheses(self, top_k):
            return ["Tune nickel flotation chemistry"]

        def set_assessments(self, assessments):
            self.assessments = assessments

    fw = object.__new__(framework_mod.CoscientistFramework)
    fw.config = Config()
    fw.state_manager = Manager()
    result = fw.assess_hypotheses(top_k=1)
    assert result[0].hypothesis == "Tune nickel flotation chemistry"
    assert captured["web_sources"][0].url == "https://example.com/nicu"
