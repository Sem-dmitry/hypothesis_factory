# -*- coding: utf-8 -*-

"""
Offline tests for Phase 3: RAG-grounded generation/reflection + structured
cited hypothesis assessment. No network, only a dummy ROUTER_AI_API_KEY.
"""

import json

import pytest

from coscientist.corpus.loaders import CorpusChunk
from coscientist.corpus.retrieval import CorpusRetriever
from coscientist.corpus.store import CorpusIndex
from coscientist.hypothesis_assessment import (
    AssessmentWeights,
    HypothesisAssessment,
    assess_hypothesis,
    assessment_from_data,
    parse_assessment_json,
    rank_assessments,
)

_VOCAB = ["flotation", "grinding", "reagent", "nickel", "tailings", "hydrocyclone"]


def fake_embed(texts):
    out = []
    for t in texts:
        low = t.lower()
        out.append([float(low.count(w)) for w in _VOCAB] + [1.0])
    return out


class RecordingLLM:
    """Minimal chat-model stand-in that records prompts and returns canned text."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)

        class _Msg:
            content = self.reply

        return _Msg()


def _sample_chunks():
    return [
        CorpusChunk(text="flotation reagent dosing improves nickel recovery", source_path="a",
                    source_name="report.pdf", modality="pdf", locator="p.5"),
        CorpusChunk(text="grinding to finer size reduces tailings losses", source_path="b",
                    source_name="regs.xlsx", modality="xlsx", locator="sheet 'S' rows 1-9"),
        CorpusChunk(text="hydrocyclone classifier tuning", source_path="c",
                    source_name="flowsheet.png", modality="image", locator="image"),
    ]


def _retriever():
    idx = CorpusIndex(embed_fn=fake_embed)
    idx.add(_sample_chunks())
    return CorpusRetriever(idx, default_k=3)


# ---------------------------------------------------------------------------
# AC1: retrieval
# ---------------------------------------------------------------------------


def test_retriever_ground():
    r = _retriever()
    gc = r.ground("nickel recovery with flotation reagent", k=2)
    assert not gc.is_empty
    assert len(gc.hits) == 2
    assert "Sources:" in gc.context_block
    lines = gc.citation_lines()
    assert lines and "report.pdf" in " ".join(lines)


def test_retriever_from_path(tmp_path):
    idx = CorpusIndex(embed_fn=fake_embed)
    idx.add(_sample_chunks())
    idx.save(str(tmp_path / "idx"))
    r = CorpusRetriever.from_path(str(tmp_path / "idx"), embed_fn=fake_embed)
    gc = r.ground("tailings grinding", k=1)
    assert gc.hits[0].chunk.source_name == "regs.xlsx"


# ---------------------------------------------------------------------------
# AC2/AC4: weights, scoring, ranking
# ---------------------------------------------------------------------------


def test_weights_normalized():
    w = AssessmentWeights(novelty=2, feasibility=2, impact=4, risk=2).normalized()
    assert abs((w.novelty + w.feasibility + w.impact + w.risk) - 1.0) < 1e-9


def test_overall_score_prefers_low_risk():
    hi = HypothesisAssessment(hypothesis="h", novelty_score=8, feasibility_score=8,
                              impact_score=8, risk_level=1)
    lo = HypothesisAssessment(hypothesis="h", novelty_score=8, feasibility_score=8,
                              impact_score=8, risk_level=9)
    w = AssessmentWeights()
    assert hi.compute_overall_score(w) > lo.compute_overall_score(w)


def test_rank_assessments():
    a = HypothesisAssessment(hypothesis="a", overall_score=3.0)
    b = HypothesisAssessment(hypothesis="b", overall_score=7.0)
    c = HypothesisAssessment(hypothesis="c", overall_score=5.0)
    ranked = rank_assessments([a, b, c])
    assert [x.hypothesis for x in ranked] == ["b", "c", "a"]


# ---------------------------------------------------------------------------
# AC3: JSON parsing + assess_hypothesis
# ---------------------------------------------------------------------------


def test_parse_assessment_json_variants():
    payload = {"novelty_score": 7}
    raw = json.dumps(payload)
    assert parse_assessment_json(raw)["novelty_score"] == 7
    assert parse_assessment_json(f"```json\n{raw}\n```")["novelty_score"] == 7
    assert parse_assessment_json(f"Here you go:\n{raw}\nThanks")["novelty_score"] == 7
    assert parse_assessment_json("not json at all") == {}


_ASSESSMENT_JSON = json.dumps({
    "justification": "Finer grind liberates pentlandite",
    "mechanism_of_influence": "Higher liberation raises flotation recovery",
    "novelty": "Combines grind + reagent tuning",
    "novelty_score": 6,
    "feasibility_score": 8,
    "impact_score": 7,
    "risk_level": 3,
    "technical_risks": ["overgrinding slimes"],
    "economic_risks": ["higher energy cost"],
    "expected_value": "Recover more nickel from tailings",
    "target_kpi_impact": "+1.5% Ni recovery",
    "verification_plan": ["bench flotation at 2 grind sizes"],
    "confidence": 0.7,
    "evidence_refs": [1],
})


def test_assess_hypothesis_grounded():
    llm = RecordingLLM(_ASSESSMENT_JSON)
    r = _retriever()
    a = assess_hypothesis(
        "Finer grinding increases nickel recovery",
        goal="Reduce nickel losses to tailings",
        retriever=r,
        llm=llm,
        weights=AssessmentWeights(),
    )
    assert a.novelty_score == 6 and a.feasibility_score == 8
    assert a.target_kpi_impact.startswith("+1.5%")
    assert a.technical_risks and a.economic_risks
    assert a.grounded is True
    assert a.citations  # structural citations attached from retrieval
    assert a.evidence_refs == [1]
    assert len(a.source_evidence) == 1
    assert a.overall_score > 0
    # the prompt actually included the grounded corpus context
    assert "Sources:" in llm.prompts[0]


def test_assess_hypothesis_without_corpus():
    llm = RecordingLLM(_ASSESSMENT_JSON)
    a = assess_hypothesis("h", goal="g", retriever=None, llm=llm)
    assert a.grounded is False
    assert a.citations == []
    assert "No private corpus" in llm.prompts[0]


def test_source_evidence_uses_only_llm_selected_refs():
    ctx = _retriever().ground("nickel flotation reagent grinding", k=3)
    data = json.loads(_ASSESSMENT_JSON)
    data["evidence_refs"] = [2]
    a = assessment_from_data("h", data, ctx)
    assert a.citations and len(a.citations) == 3
    assert a.evidence_refs == [2]
    assert len(a.source_evidence) == 1
    assert a.source_evidence[0].citation == a.citations[1]


def test_source_evidence_rejects_invalid_or_missing_refs():
    ctx = _retriever().ground("nickel flotation reagent grinding", k=3)
    data = json.loads(_ASSESSMENT_JSON)
    data["evidence_refs"] = [0, 99, "not-a-visible-ref"]
    a = assessment_from_data("h", data, ctx)
    assert a.grounded is True
    assert a.citations  # retrieval trace remains available
    assert a.evidence_refs == []
    assert a.source_evidence == []


# ---------------------------------------------------------------------------
# AC5: prompt renders
# ---------------------------------------------------------------------------


def test_assessment_prompt_renders():
    from coscientist.common import load_prompt

    text = load_prompt("hypothesis_assessment", goal="g", hypothesis="h",
                       corpus_context="[1] evidence\n\nSources:\n[1] x.pdf — p.1 (PDF)")
    assert "JSON" in text and "mechanism_of_influence" in text


# ---------------------------------------------------------------------------
# AC6: generation grounding wired end-to-end through the compiled graph
# ---------------------------------------------------------------------------

_GEN_MARKDOWN = (
    "# Hypothesis\nFiner grinding raises nickel recovery.\n"
    "# Falsifiable Predictions\n1. Recovery rises at finer P80.\n"
    "# Assumptions\n1. Pentlandite is locked in coarse fractions.\n"
)


def test_generation_grounding_reaches_prompt():
    from coscientist.generation_agent import IndependentConfig, build_generation_agent
    from coscientist.reasoning_types import ReasoningType

    llm = RecordingLLM(_GEN_MARKDOWN)
    config = IndependentConfig(
        field="mineral processing",
        reasoning_type=ReasoningType.CAUSAL,
        llm=llm,
        retriever=_retriever(),
    )
    agent = build_generation_agent("independent", config)
    result = agent.invoke({
        "goal": "Reduce nickel losses to tailings via flotation reagent tuning",
        "literature_review": "some lit review",
        "meta_review": "Not Available",
    })
    assert result["hypothesis"].hypothesis  # parsed successfully
    # grounding block + a real citation reached the generator prompt
    assert "private knowledge base" in llm.prompts[0]
    assert "report.pdf" in llm.prompts[0]


def test_generation_without_retriever_is_unchanged():
    from coscientist.generation_agent import IndependentConfig, build_generation_agent
    from coscientist.reasoning_types import ReasoningType

    llm = RecordingLLM(_GEN_MARKDOWN)
    config = IndependentConfig(field="x", reasoning_type=ReasoningType.CAUSAL, llm=llm)
    agent = build_generation_agent("independent", config)
    agent.invoke({"goal": "g", "literature_review": "l", "meta_review": "Not Available"})
    assert "private knowledge base" not in llm.prompts[0]


# ---------------------------------------------------------------------------
# AC7: reflection imports offline + deep-verification grounding
# ---------------------------------------------------------------------------


def test_reflection_imports_without_gpt_researcher():
    # reflection_agent must import lazily: loading the module must NOT pull in
    # gpt_researcher (so it works whether or not that heavy dep is installed).
    import sys

    sys.modules.pop("coscientist.reflection_agent", None)
    already_loaded = "gpt_researcher" in sys.modules
    import coscientist.reflection_agent as ra  # must not raise

    assert hasattr(ra, "build_deep_verification_agent")
    if not already_loaded:
        assert "gpt_researcher" not in sys.modules, (
            "reflection_agent imported gpt_researcher at module load; it must be lazy"
        )


def test_deep_verification_grounding_reaches_prompt():
    from coscientist.custom_types import ParsedHypothesis
    from coscientist.reflection_agent import deep_verification_node

    llm = RecordingLLM("A rigorous verification narrative.")
    hyp = ParsedHypothesis(hypothesis="Finer grinding raises nickel recovery",
                           predictions=["p1"], assumptions=["a1"])
    state = {
        "hypothesis_to_review": hyp,
        "_causal_reasoning": "cause->effect",
        "_assumption_research_results": {"a1": "supported"},
    }
    out = deep_verification_node(state, llm, _retriever())
    assert out["reviewed_hypothesis"].verification_result
    assert "private knowledge base" in llm.prompts[0]
    assert "report.pdf" in llm.prompts[0]


def test_deep_verification_web_disabled_skips_gpt_researcher(monkeypatch):
    import sys

    from coscientist.custom_types import ParsedHypothesis
    import coscientist.reflection_agent as ra

    async def boom(*args, **kwargs):
        raise AssertionError("GPTResearcher-backed web research was called")

    class RoutedLLM:
        def invoke(self, prompt):
            class _Msg:
                pass

            msg = _Msg()
            if "FINAL EVALUATION:" in prompt:
                msg.content = "Looks plausible.\nFINAL EVALUATION: PASS"
            elif "scientific assumption analyzer" in prompt:
                msg.content = (
                    "## Assumptions\n"
                    "1. **Particle size controls losses**\n"
                    "   - Sub-assumption 1.1: Fine tailings retain nickel.\n"
                )
            elif "expert in causality" in prompt:
                msg.content = "## Causal Chain\n### Step 1: grinding -> liberation"
            else:
                msg.content = "Offline verification complete."
            return msg

    monkeypatch.setattr(ra, "_write_assumption_research_report", boom)
    sys.modules.pop("gpt_researcher", None)

    agent = ra.build_deep_verification_agent(
        llm=RoutedLLM(),
        review_llm=RoutedLLM(),
        retriever=None,
        web_research_enabled=False,
    )
    hyp = ParsedHypothesis(
        hypothesis="Finer grinding reduces nickel tailings losses",
        predictions=["p1"],
        assumptions=["a1"],
    )
    out = agent.invoke({"hypothesis_to_review": hyp})

    reviewed = out["reviewed_hypothesis"]
    assert reviewed.verification_result == "Offline verification complete."
    assert "Particle size controls losses" in reviewed.assumption_research_results
    assert "Web research disabled" in " ".join(
        reviewed.assumption_research_results.values()
    )
    assert "gpt_researcher" not in sys.modules
