# -*- coding: utf-8 -*-

"""
Offline tests for the Phase-5 stitching: corpus retriever threaded through the
framework into generation/reflection, plus the assessment + export seam on
finish(). No network; dummy ROUTER_AI_API_KEY only.
"""

import asyncio
import json
import os

import pytest

from coscientist.corpus.loaders import CorpusChunk
from coscientist.corpus.retrieval import CorpusRetriever
from coscientist.corpus.store import CorpusIndex
from coscientist.hypothesis_assessment import AssessmentWeights, HypothesisAssessment

_ASSESS_JSON = json.dumps(
    {
        "justification": "grounded", "mechanism_of_influence": "mech",
        "novelty": "n", "novelty_score": 6, "feasibility_score": 8,
        "impact_score": 7, "risk_level": 3, "technical_risks": ["r"],
        "economic_risks": ["e"], "expected_value": "v",
        "target_kpi_impact": "+1% Ni", "verification_plan": ["step"], "confidence": 0.6,
        "evidence_refs": [1],
    }
)


class FakeLLM:
    def __init__(self, reply=_ASSESS_JSON):
        self.reply = reply
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        class M:
            pass
        m = M()
        m.content = self.reply
        return m


def _fake_embed(texts):
    vocab = ["flotation", "nickel", "grinding", "tailings"]
    return [[float(t.lower().count(w)) for w in vocab] + [1.0] for t in texts]


def _retriever():
    idx = CorpusIndex(embed_fn=_fake_embed)
    idx.add([
        CorpusChunk(text="flotation of nickel from tailings", source_path="a",
                    source_name="report.pdf", modality="pdf", locator="p.1"),
    ])
    return CorpusRetriever(idx, default_k=1)


def _config(**kw):
    """A CoscientistConfig with cheap fakes for the generation pool."""
    from coscientist.framework import CoscientistConfig

    fake = FakeLLM()
    return CoscientistConfig(
        literature_review_agent_llm=fake,
        generation_agent_llms={"m1": fake, "m2": fake},
        reflection_agent_llms={"m1": fake},
        evolution_agent_llms={"m1": fake},
        meta_review_agent_llm=fake,
        supervisor_agent_llm=fake,
        final_report_agent_llm=fake,
        proximity_agent_embedding_model=object(),  # not exercised offline
        specialist_fields=["flotation", "mineral processing"],
        **kw,
    )


# ---------------------------------------------------------------------------
# AC1: config carries retriever + assessment settings, backward compatible
# ---------------------------------------------------------------------------


def test_config_backward_compatible_and_new_fields():
    cfg_default = _config()
    assert cfg_default.retriever is None
    assert isinstance(cfg_default.assessment_weights, AssessmentWeights)
    assert cfg_default.web_research_enabled is True

    r = _retriever()
    w = AssessmentWeights(impact=0.5)
    cfg = _config(retriever=r, assessment_weights=w, web_research_enabled=False)
    assert cfg.retriever is r and cfg.assessment_weights is w
    assert cfg.web_research_enabled is False


# ---------------------------------------------------------------------------
# AC2: framework threads the retriever into both generation configs
# ---------------------------------------------------------------------------


def test_generation_configs_get_retriever():
    from coscientist.framework import CoscientistFramework

    r = _retriever()
    fw = CoscientistFramework(_config(retriever=r), state_manager=None)
    ind = fw._build_independent_config()
    col = fw._build_collaborative_config()
    assert ind.retriever is r
    assert col.retriever is r
    # backward compatible: no retriever -> None
    fw2 = CoscientistFramework(_config(), state_manager=None)
    assert fw2._build_independent_config().retriever is None


# ---------------------------------------------------------------------------
# AC3: reflection gets the retriever
# ---------------------------------------------------------------------------


def test_reflection_receives_retriever(monkeypatch, tmp_path):
    import coscientist.framework as fwmod

    captured = {}

    class _StubAgent:
        def invoke(self, state):
            return {"passed_initial_filter": False}

    def _spy(*args, **kwargs):
        captured["retriever"] = kwargs.get("retriever")
        captured["web_research_enabled"] = kwargs.get("web_research_enabled")
        return _StubAgent()

    monkeypatch.setattr(fwmod, "build_deep_verification_agent", _spy)

    mgr = _manager(tmp_path, "recover nickel")
    # prime one hypothesis into the reflection queue
    from coscientist.custom_types import ParsedHypothesis
    mgr._state.reflection_queue.append(
        ParsedHypothesis(hypothesis="h", predictions=["p"], assumptions=["a"])
    )
    r = _retriever()
    fw = fwmod.CoscientistFramework(
        _config(retriever=r, web_research_enabled=False), state_manager=mgr
    )
    fw.process_reflection_queue()
    assert captured["retriever"] is r
    assert captured["web_research_enabled"] is False


def test_expand_literature_review_uses_corpus_when_web_disabled(monkeypatch, tmp_path):
    import coscientist.framework as fwmod

    def _boom(*args, **kwargs):
        raise AssertionError("web literature review agent was built")

    monkeypatch.setattr(fwmod, "build_literature_review_agent", _boom)

    mgr = _manager(tmp_path, "recover nickel")
    fw = fwmod.CoscientistFramework(
        _config(retriever=_retriever(), web_research_enabled=False), state_manager=mgr
    )
    asyncio.run(fw.expand_literature_review())
    assert mgr.has_literature_review
    assert "Corpus-Only" in mgr._state.literature_review["subtopic_reports"][0]
    assert "report.pdf" in mgr._state.literature_review["subtopic_reports"][0]


# ---------------------------------------------------------------------------
# AC4 + AC6: assessment produced, ranked, stored; helpers + backward compat
# ---------------------------------------------------------------------------


def _manager(tmp_path, goal):
    import coscientist.global_state as gs

    monkey_dir = str(tmp_path / "cosci")
    gs._OUTPUT_DIR = monkey_dir  # module-level output dir
    os.makedirs(monkey_dir, exist_ok=True)
    state = gs.CoscientistState(goal=goal)
    return gs.CoscientistStateManager(state)


def test_assess_hypotheses_stores_ranked(tmp_path):
    from coscientist.framework import CoscientistFramework

    mgr = _manager(tmp_path, "Снизить потери никеля")
    fw = CoscientistFramework(
        _config(retriever=_retriever(), assessment_llm=FakeLLM()), state_manager=mgr
    )
    result = fw.assess_hypotheses(hypotheses=["Доизмельчение хвостов", "Смена собирателя"])
    assert len(result) == 2
    assert all(isinstance(a, HypothesisAssessment) for a in result)
    assert result[0].grounded and result[0].citations  # grounded via corpus
    assert result[0].source_evidence
    first_source = result[0].source_evidence[0]
    assert first_source.source_name == "report.pdf"
    assert first_source.locator == "p.1"
    assert "flotation of nickel" in first_source.quote
    assert first_source.chunk_id
    # stored on state and retrievable
    assert len(mgr.assessments) == 2


def test_assessments_backward_compat(tmp_path):
    mgr = _manager(tmp_path, "goal x")
    del mgr._state.assessments  # simulate an old pickle without the field
    assert mgr.assessments == []


def test_top_tournament_hypotheses(tmp_path):
    mgr = _manager(tmp_path, "goal y")

    class _FakeTournament:
        hypotheses = {
            "u1": HypothesisAssessment(hypothesis="top one"),
            "u2": HypothesisAssessment(hypothesis="second"),
        }
        def get_sorted_hypotheses(self):
            return [("u1", 1500), ("u2", 1400)]

    mgr._state.tournament = _FakeTournament()
    assert mgr.top_tournament_hypotheses(1) == ["top one"]
    assert mgr.top_tournament_hypotheses(5) == ["top one", "second"]


# ---------------------------------------------------------------------------
# AC5: export_results writes deliverables
# ---------------------------------------------------------------------------


def test_export_results(tmp_path):
    from coscientist.framework import CoscientistFramework

    mgr = _manager(tmp_path, "goal z")
    mgr.set_assessments([
        HypothesisAssessment(hypothesis="h1", overall_score=6.0,
                             citations=["[1] report.pdf — p.1 (PDF)"]),
    ])
    fw = CoscientistFramework(_config(), state_manager=mgr)
    out = str(tmp_path / "out")
    paths = fw.export_results(out)
    for key in ("md", "html", "docx", "pdf", "csv", "json", "jira", "graph"):
        assert os.path.exists(paths[key]) and os.path.getsize(paths[key]) > 0


# ---------------------------------------------------------------------------
# AC6: finish() invokes assessment
# ---------------------------------------------------------------------------


def test_finish_runs_assessment(tmp_path, monkeypatch):
    import coscientist.framework as fwmod

    mgr = _manager(tmp_path, "goal finish")
    fw = fwmod.CoscientistFramework(_config(), state_manager=mgr)

    called = {}
    fw.assess_hypotheses = lambda **kw: called.setdefault("yes", kw) or []

    class _FRAgent:
        def invoke(self, state):
            return {**state, "result": "final"}

    monkeypatch.setattr(fwmod, "build_final_report_agent", lambda llm: _FRAgent())
    asyncio.run(fw.finish())
    assert "yes" in called


# ---------------------------------------------------------------------------
# AC7: build_retriever helper
# ---------------------------------------------------------------------------


def test_build_retriever_from_data_dir(tmp_path):
    from coscientist.retrieval_helpers import build_retriever

    # tiny data dir with one docx-like file is heavy; use a saved index instead
    idx = CorpusIndex(embed_fn=_fake_embed)
    idx.add([CorpusChunk(text="nickel flotation", source_path="a", source_name="a.pdf",
                         modality="pdf", locator="p.1")])
    index_path = str(tmp_path / "idx")
    idx.save(index_path)

    r = build_retriever(index_path=index_path, embed_fn=_fake_embed)
    assert isinstance(r, CorpusRetriever)
    hits = r.ground("nickel").hits
    assert hits and hits[0].chunk.source_name == "a.pdf"

    with pytest.raises(ValueError):
        build_retriever()  # neither index nor data_dir
