# -*- coding: utf-8 -*-

"""
Structured, cited per-hypothesis assessment.

Produces exactly the fields the "Фабрика гипотез" case requires for each
hypothesis — justification, sources, mechanism of influence, novelty, technical
and economic risks, expected value, target-KPI impact and a verification plan —
grounded in the private corpus (Phase 2) and scored with configurable ranking
weights (the "expert tuning" wish).

Everything is dependency-injectable (LLM + retriever), so the whole scoring path
is testable offline with a fake model and a deterministic index.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from coscientist.common import load_prompt
from coscientist.corpus.retrieval import CorpusRetriever, GroundedContext
from coscientist.tailings_domain import is_tailings_related, tailings_guide_block
from coscientist.web_literature import (
    WebLiteratureSource,
    format_web_context_block,
    format_web_reference,
    format_web_references,
)


class AssessmentWeights(BaseModel):
    """Configurable ranking-criterion weights (expert tuning)."""

    novelty: float = 0.25
    feasibility: float = 0.25
    impact: float = 0.30
    risk: float = 0.20  # applied to (10 - risk_level): lower risk scores higher

    def normalized(self) -> "AssessmentWeights":
        total = self.novelty + self.feasibility + self.impact + self.risk
        if total <= 0:
            return AssessmentWeights()
        return AssessmentWeights(
            novelty=self.novelty / total,
            feasibility=self.feasibility / total,
            impact=self.impact / total,
            risk=self.risk / total,
        )


class SourceEvidence(BaseModel):
    """A source item that grounded an assessed hypothesis."""

    chunk_id: str = Field(
        default="", description="Stable id of the retrieved corpus chunk"
    )
    source_name: str = Field(description="User-facing source file name")
    locator: str = Field(description="Location inside the source, e.g. page or rows")
    modality: str = Field(description="Corpus modality: pdf/docx/xlsx/image")
    citation: str = Field(description="Formatted structural citation")
    quote: str = Field(description="Direct excerpt from the retrieved chunk")
    rank: int = Field(default=0, description="Retrieval rank for this assessment")
    score: float = Field(default=0.0, description="Retriever similarity score")
    source_type: str = Field(
        default="corpus", description="'corpus' for uploads, 'web' for Literature links"
    )
    url: str = Field(default="", description="External URL for web literature evidence")


class HypothesisAssessment(BaseModel):
    """The case-mandated structured evaluation of a single hypothesis."""

    hypothesis: str = Field(description="The hypothesis statement being assessed")

    # Justification & mechanism
    justification: str = Field(default="", description="Why this hypothesis is plausible")
    mechanism_of_influence: str = Field(
        default="", description="Expected physical/chemical mechanism on the target KPI"
    )

    # Scored criteria (0-10)
    novelty: str = Field(default="", description="Novelty rationale vs known solutions")
    novelty_score: float = Field(default=0.0, description="Novelty score 0-10")
    feasibility_score: float = Field(
        default=0.0, description="Lab/industrial feasibility score 0-10"
    )
    impact_score: float = Field(default=0.0, description="Potential effect score 0-10")
    risk_level: float = Field(
        default=5.0, description="Aggregate risk 0-10 (higher = riskier)"
    )

    # Risks / value / KPI
    technical_risks: list[str] = Field(default_factory=list)
    economic_risks: list[str] = Field(default_factory=list)
    expected_value: str = Field(default="", description="Expected business/technical value")
    target_kpi_impact: str = Field(
        default="", description="Effect on the target KPI (e.g. Ni/Cu recovery, tailings loss)"
    )

    # Verification roadmap (optional case output)
    verification_plan: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, description="Model confidence 0-1")

    # Richer, "wow-level" justification for the industrial case (all optional,
    # backward compatible). Cause -> world practice -> what to do -> why it matters.
    causal_chain: str = Field(
        default="", description="Why the problem/loss occurs (cause->effect chain)"
    )
    world_practice: str = Field(
        default="", description="How this is addressed in world/industrial practice (cited)"
    )
    novelty_vs_input: str = Field(
        default="", description="How the idea goes beyond the provided corpus (external/patent)"
    )
    constraint_adherence: str = Field(
        default="", description="How the hypothesis respects the stated constraints"
    )
    constraint_violations: list[str] = Field(
        default_factory=list, description="Stated constraints the hypothesis may violate"
    )
    economic_estimate: str = Field(
        default="", description="Qualitative (LLM) estimate of economic effect — not a model"
    )
    kinetics_note: str = Field(
        default="", description="Flotation/reaction kinetics consideration (optional)"
    )

    # Grounding
    citations: list[str] = Field(
        default_factory=list, description="Structural citations: source_name + locator"
    )
    source_evidence: list[SourceEvidence] = Field(
        default_factory=list,
        description="Retrieved source chunks with direct excerpts",
    )
    evidence_refs: list[int] = Field(
        default_factory=list,
        description="1-based corpus citation numbers explicitly selected by the LLM",
    )
    web_evidence_refs: list[int] = Field(
        default_factory=list,
        description="1-based web Literature citation numbers explicitly selected by the LLM",
    )
    grounded: bool = Field(
        default=False, description="Whether corpus evidence was retrieved"
    )

    # Ranking
    overall_score: float = Field(default=0.0, description="Weighted ranking score 0-10")

    def compute_overall_score(self, weights: AssessmentWeights) -> float:
        """Weighted blend of criteria; risk contributes inversely (10 - risk)."""
        w = weights.normalized()
        score = (
            w.novelty * self.novelty_score
            + w.feasibility * self.feasibility_score
            + w.impact * self.impact_score
            + w.risk * (10.0 - self.risk_level)
        )
        self.overall_score = round(float(score), 3)
        return self.overall_score


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _extract_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def parse_assessment_json(text: str) -> dict:
    """Extract a JSON object from a model response (tolerates ```json fences)."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start : end + 1] if start != -1 and end > start else "{}"
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        data = {}
    return data if isinstance(data, dict) else {}


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compact_quote(text: str, *, limit: int = 500) -> str:
    """Return a concise direct excerpt suitable for hypothesis cards."""
    quote = re.sub(r"\s+", " ", (text or "")).strip()
    if len(quote) <= limit:
        return quote
    return quote[:limit].rstrip() + " ..."


def _coerce_evidence_refs(value: Any, *, max_ref: int) -> list[int]:
    """Parse and validate 1-based citation refs selected by the assessment LLM."""
    if value is None or max_ref <= 0:
        return []
    raw_items = value if isinstance(value, list) else [value]
    refs: list[int] = []
    seen: set[int] = set()
    for item in raw_items:
        if isinstance(item, bool):
            candidates: list[int] = []
        elif isinstance(item, int):
            candidates = [item]
        else:
            candidates = [int(n) for n in re.findall(r"\d+", str(item))]
        for ref in candidates:
            if 1 <= ref <= max_ref and ref not in seen:
                refs.append(ref)
                seen.add(ref)
    return refs


def _source_evidence(
    grounded_context: GroundedContext,
    evidence_refs: list[int],
) -> list[SourceEvidence]:
    evidence: list[SourceEvidence] = []
    citation_lines = grounded_context.citation_lines()
    for ref in evidence_refs:
        hit = grounded_context.hits[ref - 1]
        chunk = hit.chunk
        evidence.append(
            SourceEvidence(
                chunk_id=chunk.chunk_id,
                source_name=chunk.source_name,
                locator=chunk.locator,
                modality=chunk.modality,
                citation=citation_lines[ref - 1],
                quote=_compact_quote(chunk.text),
                rank=ref,
                score=round(float(hit.score), 6),
                source_type="corpus",
            )
        )
    return evidence


def _web_source_evidence(
    web_sources: list[WebLiteratureSource],
    web_evidence_refs: list[int],
) -> list[SourceEvidence]:
    evidence: list[SourceEvidence] = []
    for ref in web_evidence_refs:
        source = web_sources[ref - 1]
        evidence.append(
            SourceEvidence(
                source_name=source.title,
                locator=f"W{ref}",
                modality="web",
                citation=format_web_reference(source, ref),
                quote=_compact_quote(source.snippet or source.title),
                rank=ref,
                score=0.0,
                source_type="web",
                url=source.url,
            )
        )
    return evidence


def assessment_from_data(
    hypothesis: str,
    data: dict,
    grounded_context: Optional[GroundedContext] = None,
    web_sources: Optional[list[WebLiteratureSource]] = None,
) -> HypothesisAssessment:
    """Build a HypothesisAssessment from parsed JSON + structural citations."""
    assessment = HypothesisAssessment(
        hypothesis=hypothesis,
        justification=str(data.get("justification", "")).strip(),
        mechanism_of_influence=str(data.get("mechanism_of_influence", "")).strip(),
        novelty=str(data.get("novelty", "")).strip(),
        novelty_score=_coerce_float(data.get("novelty_score"), 0.0),
        feasibility_score=_coerce_float(data.get("feasibility_score"), 0.0),
        impact_score=_coerce_float(data.get("impact_score"), 0.0),
        risk_level=_coerce_float(data.get("risk_level"), 5.0),
        technical_risks=_coerce_list(data.get("technical_risks")),
        economic_risks=_coerce_list(data.get("economic_risks")),
        expected_value=str(data.get("expected_value", "")).strip(),
        target_kpi_impact=str(data.get("target_kpi_impact", "")).strip(),
        verification_plan=_coerce_list(data.get("verification_plan")),
        confidence=_coerce_float(data.get("confidence"), 0.5),
        causal_chain=str(data.get("causal_chain", "")).strip(),
        world_practice=str(data.get("world_practice", "")).strip(),
        novelty_vs_input=str(data.get("novelty_vs_input", "")).strip(),
        constraint_adherence=str(data.get("constraint_adherence", "")).strip(),
        constraint_violations=_coerce_list(data.get("constraint_violations")),
        economic_estimate=str(data.get("economic_estimate", "")).strip(),
        kinetics_note=str(data.get("kinetics_note", "")).strip(),
    )
    # Citations are structural: taken from the retrieved hits, never invented.
    if grounded_context is not None and not grounded_context.is_empty:
        assessment.citations = grounded_context.citation_lines()
        assessment.evidence_refs = _coerce_evidence_refs(
            data.get("evidence_refs"), max_ref=len(grounded_context.hits)
        )
        assessment.source_evidence = _source_evidence(
            grounded_context, assessment.evidence_refs
        )
        assessment.grounded = True
    if web_sources:
        assessment.citations.extend(format_web_references(web_sources))
        assessment.web_evidence_refs = _coerce_evidence_refs(
            data.get("web_evidence_refs"), max_ref=len(web_sources)
        )
        assessment.source_evidence.extend(
            _web_source_evidence(web_sources, assessment.web_evidence_refs)
        )
        assessment.grounded = assessment.grounded or bool(assessment.web_evidence_refs)
    return assessment


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def assess_hypothesis(
    hypothesis: str,
    *,
    goal: str,
    retriever: Optional[CorpusRetriever],
    llm: Any,
    weights: Optional[AssessmentWeights] = None,
    k: Optional[int] = None,
    constraints: str = "",
    web_sources: Optional[list[WebLiteratureSource]] = None,
) -> HypothesisAssessment:
    """
    Produce a structured, cited assessment for one hypothesis.

    Parameters
    ----------
    retriever : CorpusRetriever | None
        Private-corpus retriever for grounding. If None, the assessment is
        produced without corpus citations.
    llm : object with ``.invoke(prompt) -> obj.content``
        Chat model (built via model_factory in production; a fake in tests).
    weights : AssessmentWeights | None
        Ranking-criterion weights; defaults to a balanced profile.
    """
    weights = weights or AssessmentWeights()

    grounded_context: Optional[GroundedContext] = None
    if retriever is not None:
        retrieval_query = (
            f"Research goal:\n{goal}\n\n"
            f"Hypothesis:\n{hypothesis}\n\n"
            f"Constraints:\n{constraints or ''}"
        )
        grounded_context = retriever.ground(retrieval_query, k=k)
        corpus_context = grounded_context.context_block
    elif is_tailings_related(goal, hypothesis, constraints):
        corpus_context = (
            f"{tailings_guide_block()}\n\n"
            "No private corpus was provided, so do not cite uploaded evidence."
        )
    else:
        corpus_context = "No private corpus was provided."

    web_sources = web_sources or []
    web_literature_context = format_web_context_block(web_sources)

    prompt = load_prompt(
        "hypothesis_assessment",
        goal=goal,
        hypothesis=hypothesis,
        corpus_context=corpus_context,
        web_literature_context=web_literature_context,
        constraints=constraints or "No explicit constraints were provided.",
    )
    raw = _extract_content(llm.invoke(prompt))
    data = parse_assessment_json(raw)

    assessment = assessment_from_data(hypothesis, data, grounded_context, web_sources)
    assessment.compute_overall_score(weights)
    return assessment


def rank_assessments(
    assessments: list[HypothesisAssessment],
) -> list[HypothesisAssessment]:
    """Return assessments sorted by ``overall_score`` (descending)."""
    return sorted(assessments, key=lambda a: a.overall_score, reverse=True)
