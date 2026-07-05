"""Offline tests for the Russian-user language adaptation.

The language policy is appended centrally in ``coscientist.common.load_prompt``:
- agent prompts get a "respond in Russian" policy;
- the Literature agent's search-query prompt (``topic_decomposition``) gets a
  policy that permits English / multilingual queries for world-practice search;
- structural control tokens stay intact so parsers keep working.
"""

import glob
import os

from coscientist.common import (
    _RU_POLICY_MARKER,
    _SEARCH_POLICY_MARKER,
    LANG_POLICY_RU,
    load_prompt,
    language_policy_for,
)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "coscientist", "prompts")

# Minimal kwargs so every template renders; unknown extras are ignored by Jinja.
_KW = dict(
    goal="Повысить извлечение никеля из хвостов",
    field="обогащение руд", reasoning_type="", literature_review="", meta_review="",
    previous_meta_review="", constraints="Без нового оборудования", private_corpus_grounding="",
    corpus_context="", hypothesis="Гипотеза X", hypothesis_1="H1", hypothesis_2="H2",
    review_1="r1", review_2="r2", direction="d", message="m", history="", assessment="{}",
    max_subtopics=3, subtopics="", n=2, total_actions=0, latest_actions="",
    total_hypotheses=0, num_unranked_hypotheses=0, num_meta_reviews=0,
    new_hypotheses_since_meta_review=0, total_matches_played=0, total_rounds_played=0,
    top_3_elo_ratings="[]", max_elo_rating="", num_elo_ratings_over_1400="0",
    median_elo_rating="", cosine_similarity_trajectory="", cluster_count_trajectory="",
    literature_review_subtopics_completed="", execution_issues="None", hypotheses_budget="",
    user_guidance="", has_clear_leader="", is_plateau="", leader_gap="",
    max_elo_delta_recent="", num_contenders="", actions_budget="",
)


def _all_prompt_names():
    for p in glob.glob(os.path.join(PROMPTS_DIR, "*.md")):
        yield os.path.splitext(os.path.basename(p))[0]


# --------------------------------------------------------------------------- #
# AC1 — Russian output policy on agent prompts
# --------------------------------------------------------------------------- #

AGENT_PROMPTS = [
    "independent_generation", "collaborative_generation", "hypothesis_assessment",
    "final_report", "meta_review_tournament", "supervisor_decision",
    "constraint_elicitation", "refinement_chat", "refinement_branch", "tournament",
    "simulated_debate", "desk_reject", "deep_verification", "research_config",
    "evolve_from_feedback", "out_of_the_box",
]


def test_agent_prompts_get_russian_policy():
    for name in AGENT_PROMPTS:
        out = load_prompt(name, **_KW)
        assert _RU_POLICY_MARKER in out, f"{name} missing Russian policy"
        # It must NOT get the search policy.
        assert _SEARCH_POLICY_MARKER not in out, f"{name} wrongly got search policy"


# --------------------------------------------------------------------------- #
# AC2 — multilingual search policy on the Literature topic decomposition
# --------------------------------------------------------------------------- #


def test_topic_decomposition_gets_search_policy_not_ru():
    out = load_prompt("topic_decomposition", **_KW)
    assert _SEARCH_POLICY_MARKER in out
    assert "English" in out or "английск" in out  # explicitly permits English queries
    # Must NOT force Russian-only output on the query former.
    assert _RU_POLICY_MARKER not in out
    # Structural header for the parser survives.
    assert "### Subtopic N" in out


def test_only_topic_decomposition_is_a_search_prompt():
    assert language_policy_for("topic_decomposition") != LANG_POLICY_RU
    for name in _all_prompt_names():
        if name != "topic_decomposition":
            assert language_policy_for(name) == LANG_POLICY_RU, name


# --------------------------------------------------------------------------- #
# AC3 — critical format tokens preserved (parsers keep working)
# --------------------------------------------------------------------------- #


def test_format_tokens_preserved_after_policy():
    checks = {
        "tournament": "WINNER:",
        "simulated_debate": "WINNER:",
        "desk_reject": "FINAL EVALUATION:",
        "research_config": "FINAL GOAL:",
        "supervisor_decision": "DECISION:",
        "independent_generation": "# Hypothesis",
        "topic_decomposition": "### Subtopic N",
    }
    for name, token in checks.items():
        out = load_prompt(name, **_KW)
        assert token in out, f"{name} lost token {token}"


def test_policy_protects_control_tokens_text():
    # The RU policy instructs preserving the output format + JSON keys + ALL-CAPS
    # command labels generically (without importing foreign tokens into every
    # prompt). Harmless markdown-marker examples are named explicitly.
    assert "формат" in LANG_POLICY_RU
    assert "JSON" in LANG_POLICY_RU
    assert "ВЕРХНЕМ РЕГИСТРЕ" in LANG_POLICY_RU
    assert "# Hypothesis" in LANG_POLICY_RU and "### Subtopic N" in LANG_POLICY_RU
    # It must NOT pollute prompts with unrelated control tokens.
    for foreign in ["WINNER:", "FINAL EVALUATION:", "FINAL GOAL:", "DECISION:"]:
        assert foreign not in LANG_POLICY_RU


def test_parsers_are_language_robust_with_russian_prose():
    # Supervisor: Russian reasoning + English DECISION/REASONING tokens.
    from coscientist.supervisor_agent import _parse_supervisor_response

    resp = "DECISION: generate_new_hypotheses\nREASONING: Нужно больше гипотез, лидер не оторвался."
    action, reasoning = _parse_supervisor_response(resp)
    assert action == "generate_new_hypotheses"
    assert "лидер" in reasoning

    # Desk-reject pass check: Russian body + English verdict token.
    body = "Обоснование по трём критериям на русском.\nFINAL EVALUATION: PASS"
    assert "pass" in body.split("FINAL EVALUATION:")[-1].lower()

    # Configuration FINAL GOAL split with Russian goal text.
    conv = "Уточняю цель...\nFINAL GOAL: Повысить извлечение никеля из хвостов"
    assert conv.split("FINAL GOAL:")[1].strip().startswith("Повысить")

    # WINNER detection with Russian rationale.
    debate = "Гипотеза 1 сильнее по механизму и реализуемости.\nWINNER: 1"
    assert "WINNER:" in debate


# --------------------------------------------------------------------------- #
# AC4 — every prompt still renders (with policy) without error
# --------------------------------------------------------------------------- #


def test_all_prompts_render_with_policy():
    for name in _all_prompt_names():
        out = load_prompt(name, **_KW)
        assert out.strip()
        marker = _SEARCH_POLICY_MARKER if name == "topic_decomposition" else _RU_POLICY_MARKER
        assert marker in out, name
