"""Tests for robust topic-decomposition parsing (deep-mode literature review)."""

from coscientist.literature_review_agent import parse_topic_decomposition


def test_strict_english_format():
    text = (
        "## Research Subtopics\n"
        "### Subtopic 1\nMechanisms of nickel loss in tailings.\n\n"
        "### Subtopic 2\nEffect of grind size on liberation.\n"
    )
    subs = parse_topic_decomposition(text)
    assert len(subs) == 2
    assert "nickel loss" in subs[0]


def test_localized_headers_russian():
    text = (
        "## Подтемы\n"
        "### Подтема 1\nМеханизмы потерь никеля с хвостами флотации.\n\n"
        "### Подтема 2\nВлияние крупности измельчения на раскрытие.\n"
    )
    subs = parse_topic_decomposition(text)
    assert len(subs) == 2
    assert "никел" in subs[0].lower()


def test_code_fence_wrapped():
    text = (
        "```markdown\n"
        "### Subtopic 1\nReagent regime and selectivity.\n"
        "### Subtopic 2\npH control of pyrrhotite.\n"
        "```"
    )
    subs = parse_topic_decomposition(text)
    assert len(subs) == 2


def test_numbered_list_fallback():
    text = (
        "Here are the subtopics:\n"
        "1. How does collector dosage affect chalcopyrite selectivity?\n"
        "2. What is the role of serpentine in nickel losses?\n"
    )
    subs = parse_topic_decomposition(text)
    assert len(subs) == 2
    assert "collector dosage" in subs[0]


def test_empty_returns_empty():
    assert parse_topic_decomposition("") == []
    assert parse_topic_decomposition("   \n  ") == []


def test_node_falls_back_to_goal(monkeypatch):
    from coscientist import literature_review_agent as lra

    class _LLM:
        def invoke(self, prompt):
            class M:
                content = "Sorry, I cannot help with that."  # no parseable topics
            return M()

    state = {"goal": "Снизить потери никеля", "max_subtopics": 5,
             "subtopics": [], "meta_review": ""}
    out = lra._topic_decomposition_node(state, _LLM())
    assert out["subtopics"] == ["Снизить потери никеля"]  # graceful fallback, no crash
