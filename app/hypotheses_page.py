"""
Streamlit page: ranked hypothesis assessments + one-click export.

All report/task building is delegated to the tested `coscientist.export`
modules; this page only wires them to UI widgets.
"""

import streamlit as st

from coscientist.export import (
    assessments_to_csv,
    assessments_to_jira,
    assessments_to_json,
    render_html,
    render_markdown,
)
from coscientist.hypothesis_assessment import HypothesisAssessment, rank_assessments


def _coerce_assessments(raw) -> list:
    """Accept HypothesisAssessment objects or plain dicts from state."""
    out = []
    for item in raw or []:
        if isinstance(item, HypothesisAssessment):
            out.append(item)
        elif isinstance(item, dict):
            try:
                out.append(HypothesisAssessment(**item))
            except Exception:
                continue
    return out


def display_hypotheses_page(state):
    """Render ranked assessments with export buttons.

    Parameters
    ----------
    state : object
        Any object exposing an ``assessments`` attribute (list of
        HypothesisAssessment or dicts). Falls back to a friendly message.
    """
    st.header("🧪 Ранжированные гипотезы")

    goal = getattr(state, "goal", "") or ""
    assessments = _coerce_assessments(getattr(state, "assessments", None))

    if not assessments:
        st.warning(
            "Оценённых гипотез пока нет. Запустите оценку (assess_hypothesis) "
            "или загрузите состояние с готовыми оценками."
        )
        return

    ranked = rank_assessments(assessments)

    # Export bar
    st.download_button("⬇️ Markdown", render_markdown(ranked, goal=goal),
                       file_name="hypotheses_report.md")
    st.download_button("⬇️ HTML", render_html(ranked, goal=goal),
                       file_name="hypotheses_report.html")
    st.download_button("⬇️ CSV", assessments_to_csv(ranked),
                       file_name="hypotheses.csv")
    st.download_button("⬇️ JSON", assessments_to_json(ranked),
                       file_name="hypotheses.json")
    import json as _json
    st.download_button("⬇️ Jira payloads", _json.dumps(assessments_to_jira(ranked),
                       ensure_ascii=False, indent=2), file_name="jira_tasks.json")

    for i, a in enumerate(ranked, start=1):
        with st.expander(f"{i}. {a.hypothesis}  —  score {a.overall_score}"):
            st.markdown(f"**Механизм влияния:** {a.mechanism_of_influence}")
            st.markdown(f"**Новизна:** {a.novelty} (score {a.novelty_score})")
            st.markdown(f"**Ожидаемая ценность:** {a.expected_value}")
            st.markdown(f"**Влияние на KPI:** {a.target_kpi_impact}")
            if a.technical_risks:
                st.markdown("**Технические риски:** " + "; ".join(a.technical_risks))
            if a.economic_risks:
                st.markdown("**Экономические риски:** " + "; ".join(a.economic_risks))
            if a.verification_plan:
                st.markdown("**Дорожная карта проверки:**")
                for s in a.verification_plan:
                    st.markdown(f"- {s}")
            if a.citations:
                st.markdown("**Источники:**")
                for c in a.citations:
                    st.markdown(f"- {c}")
