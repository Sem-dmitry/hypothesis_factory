"""
Feedback learning (no back-propagation).

Experts mark generated hypotheses as confirmed / refuted / inconclusive. The
store persists these outcomes and turns them into (a) a natural-language prompt
snippet that biases future generation toward what worked and away from what
failed, and (b) small nudges to the ranking weights. This mirrors the paper's
meta-review feedback loop: learning by appending context, not fine-tuning.
"""

from __future__ import annotations

import json
import os
from typing import Literal, Optional

from pydantic import BaseModel, Field

from coscientist.hypothesis_assessment import AssessmentWeights

Outcome = Literal["confirmed", "refuted", "inconclusive"]


class FeedbackRecord(BaseModel):
    hypothesis: str
    outcome: Outcome
    note: str = ""
    tags: list[str] = Field(default_factory=list)


class FeedbackStore(BaseModel):
    """A persistable collection of expert feedback records."""

    records: list[FeedbackRecord] = Field(default_factory=list)

    # -- recording ---------------------------------------------------------

    def add(
        self,
        hypothesis: str,
        outcome: Outcome,
        note: str = "",
        tags: Optional[list[str]] = None,
    ) -> FeedbackRecord:
        rec = FeedbackRecord(
            hypothesis=hypothesis, outcome=outcome, note=note, tags=tags or []
        )
        self.records.append(rec)
        return rec

    def counts(self) -> dict[str, int]:
        c = {"confirmed": 0, "refuted": 0, "inconclusive": 0}
        for r in self.records:
            c[r.outcome] += 1
        return c

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> str:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str) -> "FeedbackStore":
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate(json.load(fh))

    # -- learning ----------------------------------------------------------

    def feedback_prompt_snippet(self, max_items: int = 8) -> str:
        """
        A natural-language summary of prior outcomes, appended to generation
        prompts so the system avoids refuted directions and reinforces confirmed
        ones. Empty when there is no feedback.
        """
        if not self.records:
            return ""
        confirmed = [r for r in self.records if r.outcome == "confirmed"]
        refuted = [r for r in self.records if r.outcome == "refuted"]
        lines = ["# Feedback from prior experiments (expert-validated)"]
        if confirmed:
            lines.append("Directions that were CONFIRMED — build on these:")
            for r in confirmed[:max_items]:
                suffix = f" ({r.note})" if r.note else ""
                lines.append(f"- {r.hypothesis}{suffix}")
        if refuted:
            lines.append("Directions that were REFUTED — avoid repeating these:")
            for r in refuted[:max_items]:
                suffix = f" ({r.note})" if r.note else ""
                lines.append(f"- {r.hypothesis}{suffix}")
        return "\n".join(lines)

    def adjust_weights(
        self, weights: Optional[AssessmentWeights] = None, *, step: float = 0.05
    ) -> AssessmentWeights:
        """
        Nudge ranking weights from the outcome mix. If many hypotheses were
        refuted, lean more on feasibility and risk aversion; if many were
        confirmed, lean more on impact and novelty. Returns a normalized copy.
        """
        w = (weights or AssessmentWeights()).model_copy(deep=True)
        counts = self.counts()
        total = sum(counts.values())
        if total == 0:
            return w.normalized()
        refuted_frac = counts["refuted"] / total
        confirmed_frac = counts["confirmed"] / total
        if refuted_frac > confirmed_frac:
            w.feasibility += step
            w.risk += step
            w.impact = max(0.0, w.impact - step)
            w.novelty = max(0.0, w.novelty - step)
        elif confirmed_frac > refuted_frac:
            w.impact += step
            w.novelty += step
            w.feasibility = max(0.0, w.feasibility - step)
        return w.normalized()
