"""Robustness tests for parse_hypothesis_markdown (generation/evolution output)."""

import pytest

from coscientist.common import parse_hypothesis_markdown


def test_standard_english_format():
    md = (
        "# Hypothesis\nFiner grinding raises nickel recovery.\n\n"
        "# Falsifiable Predictions\n1. Recovery rises at P80=45um.\n2. Tailings Ni drops.\n\n"
        "# Assumptions\n1. Pentlandite is locked.\n"
    )
    h = parse_hypothesis_markdown(md)
    assert "nickel recovery" in h.hypothesis
    assert len(h.predictions) == 2 and len(h.assumptions) == 1


def test_russian_headings():
    md = (
        "# Гипотеза\nДоизмельчение хвостов повысит извлечение никеля.\n\n"
        "# Проверяемые предсказания\n1. Извлечение растёт при P80=45мкм.\n\n"
        "# Допущения\n1. Пентландит в сростках.\n"
    )
    h = parse_hypothesis_markdown(md)
    assert "никел" in h.hypothesis.lower()
    assert len(h.predictions) == 1 and len(h.assumptions) == 1


def test_inline_content_and_levels():
    md = (
        "## Hypothesis: Changing the collector improves selectivity.\n"
        "### Falsifiable Predictions\n1. Cu grade rises.\n"
        "### Assumptions\n1. Chalcopyrite is liberated.\n"
    )
    h = parse_hypothesis_markdown(md)
    assert "collector" in h.hypothesis
    assert h.predictions and h.assumptions


def test_code_fence_wrapped():
    md = (
        "```markdown\n# Hypothesis\nX raises Y.\n"
        "# Falsifiable Predictions\n1. p\n# Assumptions\n1. a\n```"
    )
    h = parse_hypothesis_markdown(md)
    assert h.hypothesis.startswith("X raises Y")


def test_final_report_marker():
    md = "blah blah\n#FINAL REPORT#\n# Hypothesis\nThe real one.\n# Assumptions\n1. a\n"
    h = parse_hypothesis_markdown(md)
    assert h.hypothesis == "The real one."


def test_missing_predictions_does_not_crash():
    # Only a hypothesis — must not raise (predictions/assumptions default empty).
    md = "# Hypothesis\nSome standalone idea without the other sections.\n"
    h = parse_hypothesis_markdown(md)
    assert "standalone idea" in h.hypothesis
    assert h.predictions == [] and h.assumptions == []


def test_no_headings_fallback_to_whole_text():
    md = "Adding niobium to alloy X improves heat resistance via carbides."
    h = parse_hypothesis_markdown(md)
    assert "niobium" in h.hypothesis


def test_truly_empty_raises():
    with pytest.raises(ValueError):
        parse_hypothesis_markdown("   \n  ")
