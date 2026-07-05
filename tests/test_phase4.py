"""
Offline tests for Phase 4 product features: export (report + tasks), feedback
learning, i18n, visualization, collaborative grounding, and the demo UI page.
No network; dummy ROUTER_AI_API_KEY only.
"""

import csv
import io
import json
import os
import subprocess
import sys

import pytest

from coscientist.hypothesis_assessment import AssessmentWeights, HypothesisAssessment


def _sample():
    return [
        HypothesisAssessment(
            hypothesis="Доизмельчение хвостов повышает извлечение никеля",
            justification="Раскрытие пентландита", mechanism_of_influence="Рост степени раскрытия",
            novelty="Комбинация помола и реагента", novelty_score=6, feasibility_score=8,
            impact_score=7, risk_level=3, technical_risks=["ошламование"],
            economic_risks=["энергозатраты"], expected_value="Меньше потерь Ni",
            target_kpi_impact="+1.5% Ni recovery", verification_plan=["флотация при 2 P80"],
            citations=["[1] report.pdf — p.5 (PDF)"], overall_score=6.5,
        ),
        HypothesisAssessment(
            hypothesis="Смена собирателя улучшает селективность",
            mechanism_of_influence="Гидрофобизация Cp", novelty_score=4, feasibility_score=9,
            impact_score=5, risk_level=2, citations=["[1] regs.xlsx — sheet 'S' rows 1-9 (XLSX)"],
            overall_score=5.0,
        ),
    ]


# ---------------------------------------------------------------------------
# AC1: report export (md/html/docx/pdf)
# ---------------------------------------------------------------------------


def test_report_all_formats(tmp_path):
    from coscientist.export import report

    written = report.write_report(_sample(), str(tmp_path), goal="Снизить потери никеля")
    for fmt in ("md", "html", "docx", "pdf"):
        assert os.path.exists(written[fmt]) and os.path.getsize(written[fmt]) > 0

    md = report.render_markdown(_sample(), goal="g")
    assert "извлечение никеля" in md and "report.pdf" in md
    html = report.render_html(_sample(), goal="g")
    assert "<html" in html.lower() and "http://" not in html and "https://" not in html


def test_report_ranked_order():
    from coscientist.export import report

    md = report.render_markdown(_sample())
    # higher overall_score (6.5) must appear before the lower (5.0)
    assert md.index("Доизмельчение") < md.index("Смена собирателя")


# ---------------------------------------------------------------------------
# AC2: task export (csv/json/jira)
# ---------------------------------------------------------------------------


def test_tasks_csv_json_jira():
    from coscientist.export import tasks

    js = json.loads(tasks.assessments_to_json(_sample()))
    assert isinstance(js, list) and js[0]["rank"] == 1 and js[0]["overall_score"] == 6.5

    reader = list(csv.DictReader(io.StringIO(tasks.assessments_to_csv(_sample()))))
    assert reader[0]["hypothesis"] and "target_kpi_impact" in reader[0]
    assert "|" in reader[0]["technical_risks"] or reader[0]["technical_risks"] != ""

    jira = tasks.assessments_to_jira(_sample(), project_key="ORE")
    assert jira[0]["fields"]["project"]["key"] == "ORE"
    assert jira[0]["fields"]["summary"].startswith("[Гипотеза]")
    assert "hypothesis" in jira[0]["fields"]["labels"]


# ---------------------------------------------------------------------------
# AC3: feedback learning
# ---------------------------------------------------------------------------


def test_feedback_store_roundtrip_and_snippet(tmp_path):
    from coscientist.feedback import FeedbackStore

    store = FeedbackStore()
    store.add("Finer grind helps", "confirmed", note="verified in bench test")
    store.add("Add reagent X", "refuted", note="no effect")
    assert store.counts() == {"confirmed": 1, "refuted": 1, "inconclusive": 0}

    path = str(tmp_path / "fb.json")
    store.save(path)
    loaded = FeedbackStore.load(path)
    assert len(loaded.records) == 2

    snippet = loaded.feedback_prompt_snippet()
    assert "CONFIRMED" in snippet and "REFUTED" in snippet and "Finer grind" in snippet
    assert FeedbackStore().feedback_prompt_snippet() == ""


def test_feedback_adjusts_weights():
    from coscientist.feedback import FeedbackStore

    store = FeedbackStore()
    for _ in range(3):
        store.add("h", "refuted")
    w = store.adjust_weights(AssessmentWeights())
    base = AssessmentWeights().normalized()
    # more refuted -> feasibility weight should not drop below baseline
    assert w.feasibility >= base.feasibility
    assert abs((w.novelty + w.feasibility + w.impact + w.risk) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# AC4: i18n
# ---------------------------------------------------------------------------


def test_detect_language():
    from coscientist.i18n import detect_language

    assert detect_language("извлечение никеля из хвостов") == "ru"
    assert detect_language("nickel recovery from tailings") == "en"
    assert detect_language("从尾矿中回收镍") == "zh"
    assert detect_language("12345 %%%") == "other"


def test_translate_with_fake_llm():
    from coscientist.i18n import translate

    class FakeLLM:
        def __init__(self): self.prompts = []
        def invoke(self, p):
            self.prompts.append(p)
            class M: content = "nickel recovery"
            return M()

    llm = FakeLLM()
    out = translate("извлечение никеля", "en", llm)
    assert out == "nickel recovery" and "English" in llm.prompts[0]
    # no-op when already target language
    assert translate("already english", "en", llm) == "already english"


# ---------------------------------------------------------------------------
# AC5: visualization
# ---------------------------------------------------------------------------


def test_viz_graph_json_and_html():
    from coscientist import viz

    g = viz.build_graph(_sample())
    hyp_nodes = [n for n, d in g.nodes(data=True) if d["kind"] == "hypothesis"]
    src_nodes = [n for n, d in g.nodes(data=True) if d["kind"] == "source"]
    assert len(hyp_nodes) == 2 and len(src_nodes) >= 1
    assert g.number_of_edges() >= 2  # each hypothesis cites a source

    data = json.loads(viz.to_json(_sample()))
    assert data["nodes"] and data["edges"]

    html = viz.to_html(_sample())
    assert "<svg" in html and "Доизмельчение" in html
    # self-contained: no external resource fetches
    assert "cdn" not in html.lower()
    assert "src=\"http" not in html and "href=\"http" not in html


# ---------------------------------------------------------------------------
# AC6: collaborative generation grounding
# ---------------------------------------------------------------------------


def _retriever():
    from coscientist.corpus.loaders import CorpusChunk
    from coscientist.corpus.retrieval import CorpusRetriever
    from coscientist.corpus.store import CorpusIndex

    def fake_embed(texts):
        vocab = ["flotation", "nickel", "grinding", "tailings"]
        return [[float(t.lower().count(w)) for w in vocab] + [1.0] for t in texts]

    idx = CorpusIndex(embed_fn=fake_embed)
    idx.add([
        CorpusChunk(text="flotation of nickel from tailings", source_path="a",
                    source_name="report.pdf", modality="pdf", locator="p.1"),
    ])
    return CorpusRetriever(idx, default_k=1)


def test_collaborative_agent_node_grounding():
    from coscientist.multiturn import create_agent_node_fn

    class RecLLM:
        def __init__(self): self.prompts = []
        def invoke(self, p):
            self.prompts.append(p)
            class M: content = "let's discuss"
            return M()

    llm = RecLLM()
    fn = create_agent_node_fn(
        agent_name="expert_a", llm=llm, prompt_name="collaborative_generation",
        prompt_keys_from_state=["goal", "literature_review", "meta_review"],
        retriever=_retriever(), field="mineral processing", reasoning_type="Causal.",
    )
    state = {"goal": "recover nickel from tailings via flotation", "literature_review": "lit",
             "meta_review": "Not Available", "transcript": [], "turn": 0,
             "next_agent": "expert_a", "finished": False}
    fn(state)
    assert "private knowledge base" in llm.prompts[0]
    assert "report.pdf" in llm.prompts[0]


def test_collaborative_config_has_retriever():
    from coscientist.generation_agent import CollaborativeConfig
    from coscientist.reasoning_types import ReasoningType

    cfg = CollaborativeConfig(
        agent_names=["a"], agent_fields={"a": "x"},
        agent_reasoning_types={"a": ReasoningType.CAUSAL}, llms={"a": object()},
        retriever=_retriever(),
    )
    assert cfg.retriever is not None


# ---------------------------------------------------------------------------
# AC7: demo UI page compiles
# ---------------------------------------------------------------------------


def test_ui_page_compiles():
    page = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "app", "hypotheses_page.py")
    result = subprocess.run([sys.executable, "-m", "py_compile", page],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
