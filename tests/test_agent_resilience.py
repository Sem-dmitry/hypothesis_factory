"""
Tests for the shared retry/skip resilience helper and its use in the ranking
tournament (a malformed debate must not abort the tournament).
"""

from coscientist.common import RETRY_FAILED, retry_call


def test_retry_call_returns_first_success():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "ok"

    assert retry_call(fn, label="x") == "ok"
    assert calls["n"] == 1


def test_retry_call_retries_then_succeeds():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "recovered"

    assert retry_call(fn, label="x", attempts=3) == "recovered"
    assert calls["n"] == 3


def test_retry_call_all_fail_returns_sentinel():
    def fn():
        raise RuntimeError("always")

    assert retry_call(fn, label="x", attempts=2) is RETRY_FAILED


# ---------------------------------------------------------------------------
# ranking tournament: malformed judgment -> random fallback winner, no crash
# ---------------------------------------------------------------------------


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def invoke(self, prompt):
        class M:
            content = self.reply
        return M()


def _reviewed(uid, text):
    from coscientist.custom_types import ReviewedHypothesis
    return ReviewedHypothesis(
        uid=uid, hypothesis=text, predictions=["p"], assumptions=["a"],
        causal_reasoning="c", assumption_research_results={"a": "ok"},
        verification_result="v",
    )


def test_determine_winner_falls_back_on_malformed_output():
    from coscientist.ranking_agent import EloTournament

    t = EloTournament("goal")
    llm = _FakeLLM("I cannot decide.")  # no 'WINNER: 1/2' -> always unparseable
    winner, debate = t._determine_winner(
        _reviewed("u1", "H1"), _reviewed("u2", "H2"), "tournament", llm
    )
    assert winner in (1, 2)          # a winner is still produced
    assert "fallback" in debate.lower()  # via the resilient fallback path


def test_determine_winner_parses_valid_output():
    from coscientist.ranking_agent import EloTournament

    t = EloTournament("goal")
    llm = _FakeLLM("Reasoning... WINNER: 2")
    winner, _ = t._determine_winner(
        _reviewed("u1", "H1"), _reviewed("u2", "H2"), "tournament", llm
    )
    assert winner == 2
