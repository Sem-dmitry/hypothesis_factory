# -*- coding: utf-8 -*-

import json
import os
import re

from coscientist.corpus.loaders import CorpusChunk, load_xlsx
from coscientist.corpus.retrieval import CorpusRetriever
from coscientist.corpus.store import CorpusIndex
from coscientist.hypothesis_assessment import assess_hypothesis
from coscientist.reasoning_types import ReasoningType

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")


def _fake_embed(texts):
    vocab = [
        "tailings",
        "nickel",
        "element",
        "28",
        "roasting",
        "flotation",
        "pnt",
        "millerite",
    ]
    out = []
    for text in texts:
        low = text.lower()
        out.append([float(low.count(w)) for w in vocab] + [1.0])
    return out


def _tailings_xlsx_path():
    for root, _dirs, files in os.walk(DATA_DIR):
        for name in files:
            if name.lower().endswith(".xlsx") and "хвост" in name.lower():
                return os.path.join(root, name)
    raise AssertionError("expected a tailings .xlsx under data/")


def test_tailings_xlsx_loader_enriches_chunks_with_domain_context():
    chunks = load_xlsx(_tailings_xlsx_path())
    assert chunks
    first = chunks[0]
    assert first.modality == "xlsx"
    assert first.metadata["tailings_report"] is True
    assert first.metadata["headers"]
    assert "Domain interpretation" in first.text
    assert "element 28 (nickel)" in first.text
    assert "particle-size" in first.text
    assert "Spreadsheet rows:" in first.text


def test_tailings_retrieval_forces_xlsx_and_limits_single_pdf_flood():
    idx = CorpusIndex(embed_fn=_fake_embed)
    idx.add(
        [
            CorpusChunk(
                text="tailings nickel flotation roasting " * 8,
                source_path="p",
                source_name="large.pdf",
                modality="pdf",
                locator=f"p.{i}",
            )
            for i in range(1, 6)
        ]
        + [
            CorpusChunk(
                text="Element 28 losses by size fraction: Pnt and millerite in tailings",
                source_path="x",
                source_name="tailings.xlsx",
                modality="xlsx",
                locator="sheet 'Итог' rows 54-104",
                metadata={"tailings_report": True},
            )
        ]
    )
    hits = CorpusRetriever(idx, default_k=3).ground(
        "Reduce element 28 nickel losses to tailings by flotation", k=3
    ).hits
    assert any(h.chunk.modality == "xlsx" for h in hits)
    assert sum(1 for h in hits if h.chunk.source_name == "large.pdf") <= 2


class _SelectingLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        refs = [
            int(m.group(1))
            for m in re.finditer(r"^\[(\d+)\].*\(XLSX\)", prompt, re.MULTILINE)
        ]
        payload = {
            "justification": "uses plant tailings evidence",
            "mechanism_of_influence": "targets recoverable Ni minerals in tailings",
            "novelty": "plant-specific",
            "novelty_score": 7,
            "feasibility_score": 8,
            "impact_score": 7,
            "risk_level": 3,
            "technical_risks": [],
            "economic_risks": [],
            "expected_value": "lower element 28 losses",
            "target_kpi_impact": "reduced Ni in tailings",
            "verification_plan": ["check size-fraction recovery"],
            "confidence": 0.8,
            "evidence_refs": refs[:1],
        }

        class M:
            content = json.dumps(payload)

        return M()


def test_assessment_uses_goal_plus_hypothesis_and_can_cite_tailings_xlsx():
    idx = CorpusIndex(embed_fn=_fake_embed)
    idx.add(
        [
            CorpusChunk(
                text="roasting gold silver concentrate " * 10,
                source_path="p",
                source_name="weak.pdf",
                modality="pdf",
                locator="p.1",
            ),
            CorpusChunk(
                text="Element 28 nickel losses to tailings by size fraction; Pnt millerite",
                source_path="x",
                source_name="tailings.xlsx",
                modality="xlsx",
                locator="sheet 'Итог' rows 54-104",
                metadata={"tailings_report": True},
            ),
        ]
    )
    llm = _SelectingLLM()
    assessment = assess_hypothesis(
        "Tune reagent regime for recoverable sulfide minerals",
        goal="Reduce element 28 nickel losses to flotation tailings",
        constraints="existing flotation circuit",
        retriever=CorpusRetriever(idx, default_k=2),
        llm=llm,
    )
    assert assessment.grounded is True
    assert "Reduce element 28 nickel losses" in llm.prompts[0]
    assert "(XLSX)" in llm.prompts[0]
    assert assessment.source_evidence
    assert assessment.source_evidence[0].source_name == "tailings.xlsx"


class _PromptRecorder:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)

        class M:
            content = self.reply

        return M()


def test_builtin_tailings_guide_reaches_generation_without_retriever():
    from coscientist.generation_agent import IndependentConfig, build_generation_agent

    llm = _PromptRecorder(
        "# Hypothesis\nTune flotation for element 28 tailings losses.\n"
        "# Falsifiable Predictions\n1. Nickel in tailings decreases.\n"
        "# Assumptions\n1. Tailings contain recoverable nickel minerals.\n"
    )
    agent = build_generation_agent(
        "independent",
        IndependentConfig(
            field="mineral processing",
            reasoning_type=ReasoningType.CAUSAL,
            llm=llm,
            retriever=None,
        ),
    )
    agent.invoke(
        {
            "goal": "Reduce element 28 nickel losses to flotation tailings",
            "literature_review": "corpus-only context",
            "meta_review": "Not Available",
        }
    )
    assert "Built-in guide" in llm.prompts[0]
    assert "Element 28 is nickel" in llm.prompts[0]


def test_builtin_tailings_guide_reaches_assessment_without_retriever():
    llm = _PromptRecorder(
        json.dumps(
            {
                "justification": "domain-guided",
                "mechanism_of_influence": "mechanism",
                "novelty": "n",
                "novelty_score": 1,
                "feasibility_score": 1,
                "impact_score": 1,
                "risk_level": 1,
                "evidence_refs": [],
            }
        )
    )
    assess_hypothesis(
        "Tune flotation regime",
        goal="Reduce element 28 nickel losses to flotation tailings",
        retriever=None,
        llm=llm,
    )
    assert "Built-in guide" in llm.prompts[0]
    assert "No private corpus was provided" in llm.prompts[0]
