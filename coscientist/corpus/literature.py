"""Corpus-only literature/context review helpers."""

from __future__ import annotations

from typing import Optional

from coscientist.corpus.retrieval import CorpusRetriever
from coscientist.tailings_domain import is_tailings_related, tailings_guide_block


def build_corpus_literature_review(
    *,
    goal: str,
    retriever: Optional[CorpusRetriever],
    k: int = 6,
) -> dict:
    """
    Build a literature-review-shaped state from the private corpus only.

    This is used when web research is disabled: the framework still receives a
    real contextual review, but no GPTResearcher/network call is required.
    """
    if retriever is not None:
        grounding = retriever.ground(goal, k=k).context_block
    elif is_tailings_related(goal):
        grounding = (
            f"{tailings_guide_block()}\n\n"
            "No uploaded private corpus was provided for this run."
        )
    else:
        grounding = "No uploaded private corpus was provided for this run."

    report = (
        "# Corpus-Only Literature And Plant Evidence Review\n\n"
        "Web literature search is disabled for this run. The current review is "
        "built from the uploaded/private corpus and built-in domain guidance.\n\n"
        "Use this context as the literature/evidence foundation for generation, "
        "reflection, ranking, and assessment. Uploaded plant evidence, especially "
        "tailings spreadsheets for tailings-loss goals, should be treated as "
        "primary evidence.\n\n"
        f"{grounding}"
    )
    return {
        "goal": goal,
        "max_subtopics": 0,
        "subtopics": ["Private corpus and plant evidence"],
        "subtopic_reports": [report],
        "meta_review": "",
    }
