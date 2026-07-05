"""
Constraint auto-elicitation (killer feature).

Before generating hypotheses, surface the *missing but relevant* constraints a
domain expert would assume — target metal, existing flowsheet/equipment, no new
CAPEX, reagent/regulatory limits, particle-size realities — so the hypotheses
stay realistic for an industrial plant even when the user under-specifies them.

Proposed items are explicit ASSUMPTIONS or CLARIFICATIONS (never invented
facts). The "assumption" items are folded into the effective constraints used
by generation and assessment. LLM + retriever are injectable for offline tests.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from coscientist.common import load_prompt
from coscientist.corpus.retrieval import CorpusRetriever
from coscientist.tailings_domain import is_tailings_related, tailings_guide_block


class ElicitedConstraint(BaseModel):
    text: str = Field(description="The proposed constraint/assumption, one sentence")
    rationale: str = Field(default="", description="Why it matters for the hypotheses")
    kind: str = Field(default="assumption", description="'assumption' or 'clarification'")


def _extract_json_array(text: str) -> list:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start, end = text.find("["), text.rfind("]")
        candidate = text[start : end + 1] if start != -1 and end > start else "[]"
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _content(response: Any) -> str:
    if isinstance(response, str):
        return response
    c = getattr(response, "content", response)
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def parse_elicited(raw: str) -> list[ElicitedConstraint]:
    """Parse the model's JSON array into ElicitedConstraint objects (tolerant)."""
    out: list[ElicitedConstraint] = []
    for item in _extract_json_array(raw):
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(ElicitedConstraint(text=text))
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("constraint") or "").strip()
            if not text:
                continue
            kind = str(item.get("kind", "assumption")).strip().lower()
            if kind not in ("assumption", "clarification"):
                kind = "assumption"
            out.append(ElicitedConstraint(
                text=text, rationale=str(item.get("rationale", "")).strip(), kind=kind
            ))
    return out


def elicit_constraints(
    goal: str,
    constraints: str,
    retriever: Optional[CorpusRetriever],
    llm: Any,
    *,
    k: Optional[int] = None,
) -> list[ElicitedConstraint]:
    """
    Propose missing-but-relevant constraints/assumptions for the goal.

    ``llm`` needs ``.invoke(prompt) -> obj.content``; ``retriever`` optionally
    grounds the proposal in the private corpus. Never raises for a bad response
    (returns an empty list), so it can't abort a run.
    """
    corpus_context = "No private corpus was provided."
    if retriever is not None:
        try:
            corpus_context = retriever.ground(goal, k=k).context_block
        except Exception:
            pass
    elif is_tailings_related(goal, constraints):
        corpus_context = tailings_guide_block()
    prompt = load_prompt(
        "constraint_elicitation",
        goal=goal,
        constraints=constraints or "No explicit constraints were provided.",
        corpus_context=corpus_context,
    )
    try:
        raw = _content(llm.invoke(prompt))
    except Exception:
        return []
    return parse_elicited(raw)


def merge_constraints(original: str, elicited: list[ElicitedConstraint]) -> str:
    """
    Combine the user's constraints with the assumed ones into a single text used
    downstream. Clarification-only items are noted but not treated as hard
    requirements.
    """
    original = (original or "").strip()
    assumptions = [c for c in elicited if c.kind == "assumption"]
    clarifications = [c for c in elicited if c.kind == "clarification"]
    parts = []
    if original:
        parts.append(original)
    if assumptions:
        parts.append(
            "Additional assumed constraints (inferred, treat as requirements):\n"
            + "\n".join(f"- {c.text}" for c in assumptions)
        )
    if clarifications:
        parts.append(
            "Open clarifications (state explicitly if wrong):\n"
            + "\n".join(f"- {c.text}" for c in clarifications)
        )
    return "\n\n".join(parts).strip()
