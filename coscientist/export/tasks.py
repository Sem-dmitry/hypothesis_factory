"""
Export hypothesis assessments as machine-consumable tasks.

- CSV / JSON for spreadsheets and external APIs.
- Jira / YouTrack-style issue payloads (summary, description, labels, custom
  fields) ready to POST to an issue tracker. We build payloads only; no network.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Sequence

from coscientist.hypothesis_assessment import HypothesisAssessment, rank_assessments

CSV_COLUMNS = [
    "rank",
    "hypothesis",
    "overall_score",
    "novelty_score",
    "feasibility_score",
    "impact_score",
    "risk_level",
    "mechanism_of_influence",
    "expected_value",
    "target_kpi_impact",
    "technical_risks",
    "economic_risks",
    "verification_plan",
    "citations",
]


def _rows(assessments: Sequence[HypothesisAssessment]) -> list[dict[str, Any]]:
    ranked = rank_assessments(list(assessments))
    rows = []
    for rank, a in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": rank,
                "hypothesis": a.hypothesis,
                "overall_score": a.overall_score,
                "novelty_score": a.novelty_score,
                "feasibility_score": a.feasibility_score,
                "impact_score": a.impact_score,
                "risk_level": a.risk_level,
                "mechanism_of_influence": a.mechanism_of_influence,
                "expected_value": a.expected_value,
                "target_kpi_impact": a.target_kpi_impact,
                "technical_risks": a.technical_risks,
                "economic_risks": a.economic_risks,
                "verification_plan": a.verification_plan,
                "citations": a.citations,
            }
        )
    return rows


def assessments_to_json(assessments: Sequence[HypothesisAssessment]) -> str:
    """Serialize ranked assessments to a JSON array string."""
    return json.dumps(_rows(assessments), ensure_ascii=False, indent=2)


def assessments_to_csv(assessments: Sequence[HypothesisAssessment]) -> str:
    """Serialize ranked assessments to CSV (list fields joined by ' | ')."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in _rows(assessments):
        flat = dict(row)
        for key in ("technical_risks", "economic_risks", "verification_plan", "citations"):
            flat[key] = " | ".join(str(x) for x in row[key])
        writer.writerow(flat)
    return buf.getvalue()


def assessments_to_jira(
    assessments: Sequence[HypothesisAssessment],
    *,
    project_key: str = "HYP",
    issue_type: str = "Task",
) -> list[dict[str, Any]]:
    """
    Build Jira/YouTrack-style issue payloads from ranked assessments.

    Each payload has the shape most trackers accept (fields.summary/description/
    labels/...), ready to be POSTed by an integration layer.
    """
    payloads: list[dict[str, Any]] = []
    for row in _rows(assessments):
        a_desc = _description(row)
        payloads.append(
            {
                "fields": {
                    "project": {"key": project_key},
                    "issuetype": {"name": issue_type},
                    "summary": f"[Гипотеза] {row['hypothesis'][:120]}",
                    "description": a_desc,
                    "labels": ["hypothesis", "co-scientist", f"score-{int(round(row['overall_score']))}"],
                    "customfield_overall_score": row["overall_score"],
                    "customfield_kpi_impact": row["target_kpi_impact"],
                }
            }
        )
    return payloads


def _description(row: dict[str, Any]) -> str:
    lines = [
        f"Механизм влияния: {row['mechanism_of_influence']}",
        f"Ожидаемая ценность: {row['expected_value']}",
        f"Влияние на KPI: {row['target_kpi_impact']}",
        f"Оценка: {row['overall_score']} (новизна {row['novelty_score']}, "
        f"реализуемость {row['feasibility_score']}, эффект {row['impact_score']}, "
        f"риск {row['risk_level']})",
    ]
    if row["technical_risks"]:
        lines.append("Технические риски: " + "; ".join(row["technical_risks"]))
    if row["economic_risks"]:
        lines.append("Экономические риски: " + "; ".join(row["economic_risks"]))
    if row["verification_plan"]:
        lines.append("Дорожная карта проверки:")
        lines += [f"  - {s}" for s in row["verification_plan"]]
    if row["citations"]:
        lines.append("Источники:")
        lines += [f"  - {c}" for c in row["citations"]]
    return "\n".join(lines)
