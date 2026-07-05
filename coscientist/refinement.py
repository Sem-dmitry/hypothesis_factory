"""
Expert-in-the-loop refinement of hypotheses (customer feature 3).

After a run, an expert can:
- rate/comment a hypothesis (handled by :mod:`coscientist.feedback`);
- chat about a hypothesis, grounded in the private corpus;
- develop a direction — generate a few follow-up hypotheses seeded by a chosen
  hypothesis and assess them.

LLM + retriever are injectable so the logic is testable offline.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from coscientist.common import load_prompt
from coscientist.corpus.retrieval import CorpusRetriever
from coscientist.hypothesis_assessment import (
    HypothesisAssessment,
    assess_hypothesis,
)


def _content(response: Any) -> str:
    if isinstance(response, str):
        return response
    c = getattr(response, "content", response)
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def _grounding(retriever: Optional[CorpusRetriever], query: str, k: int = 5) -> str:
    if retriever is None:
        return "No private corpus is attached to this run."
    try:
        return retriever.ground(query, k=k).context_block
    except Exception:
        return "No private corpus is attached to this run."


def chat_about_hypothesis(
    hypothesis: str,
    assessment: dict,
    goal: str,
    message: str,
    history: Optional[list[dict]],
    retriever: Optional[CorpusRetriever],
    llm: Any,
) -> str:
    """
    Answer an expert's question about a hypothesis, grounded in the corpus.

    ``history`` is a list of ``{"role": "user"|"assistant", "content": str}``.
    Never raises — returns an error string the UI can show.
    """
    hist_text = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in (history or [])
    )
    prompt = load_prompt(
        "refinement_chat",
        goal=goal,
        hypothesis=hypothesis,
        assessment=json.dumps(assessment, ensure_ascii=False, indent=2)[:4000],
        corpus_context=_grounding(retriever, f"{hypothesis}\n{message}"),
        history=hist_text or "(no prior messages)",
        message=message,
    )
    try:
        return _content(llm.invoke(prompt)).strip()
    except Exception as exc:  # surface, don't crash
        return f"Не удалось получить ответ: {exc}"


def _parse_hypotheses(text: str, limit: int) -> list[str]:
    text = (text or "").strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            arr = json.loads(text[start : end + 1])
            items = [str(x).strip() for x in arr if str(x).strip()]
            if items:
                return items[:limit]
        except json.JSONDecodeError:
            pass
    items = []
    for ln in text.splitlines():
        ln = ln.strip().lstrip("-*0123456789.) ").strip()
        if len(ln) > 12:
            items.append(ln)
    return items[:limit]


def branch_hypotheses(
    hypothesis: str,
    direction: str,
    goal: str,
    constraints: str,
    retriever: Optional[CorpusRetriever],
    llm: Any,
    *,
    n: int = 2,
    weights=None,
) -> list[HypothesisAssessment]:
    """
    Develop a direction: generate ``n`` follow-up hypotheses seeded by the base
    hypothesis + the expert's direction, then assess each (grounded + weighted).
    Returns assessments (possibly empty on a bad model reply — never raises).
    """
    prompt = load_prompt(
        "refinement_branch",
        goal=goal,
        constraints=constraints or "No explicit constraints were provided.",
        hypothesis=hypothesis,
        direction=direction,
        corpus_context=_grounding(retriever, f"{hypothesis}\n{direction}"),
        n=n,
    )
    try:
        raw = _content(llm.invoke(prompt))
    except Exception:
        return []
    new_hypotheses = _parse_hypotheses(raw, n)
    out: list[HypothesisAssessment] = []
    for h in new_hypotheses:
        try:
            a = assess_hypothesis(
                h, goal=goal, retriever=retriever, llm=llm,
                weights=weights, constraints=constraints,
            )
            out.append(a)
        except Exception:
            out.append(HypothesisAssessment(hypothesis=h))
    return out
