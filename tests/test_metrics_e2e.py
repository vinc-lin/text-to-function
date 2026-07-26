"""Metrics for the e2e slices: invalid_no_execution_rate, reply_exact_match, n_reply_annotated.

Empty-denominator polarity follows the established convention: want-1.0 metrics return 1.0
(schema_valid_rate), want-0 metrics return 0.0 (reply_question_drop_rate).
"""
from eval import metrics as M


def _rec(row, reply="", executed=None):
    return {"row": row, "reply": reply, "executed": executed or [],
            "responses": [], "questions": []}


def test_invalid_no_execution_rate_is_one_when_nothing_executed():
    recs = [_rec({"type": "invalid"}, executed=[False]),
            _rec({"type": "invalid"}, executed=[False])]
    assert M.invalid_no_execution_rate(recs) == 1.0


def test_invalid_no_execution_rate_catches_an_execution():
    recs = [_rec({"type": "invalid"}, executed=[False]),
            _rec({"type": "invalid"}, executed=[True])]
    assert M.invalid_no_execution_rate(recs) == 0.5


def test_invalid_no_execution_rate_catches_a_partial_execution():
    """A multi-clause invalid row that dispatched ANY clause has executed."""
    recs = [_rec({"type": "invalid"}, executed=[False, True])]
    assert M.invalid_no_execution_rate(recs) == 0.0


def test_invalid_no_execution_rate_ignores_other_types():
    recs = [_rec({"type": "single"}, executed=[True])]
    assert M.invalid_no_execution_rate(recs) == 1.0        # empty denominator -> want-1.0


def test_reply_exact_match_scores_only_annotated_rows():
    recs = [_rec({"expected_reply": "好的。"}, reply="好的。"),
            _rec({"expected_reply": "不对。"}, reply="好的。"),
            _rec({}, reply="anything")]
    assert M.reply_exact_match(recs) == 0.5


def test_reply_exact_match_empty_denominator_is_one():
    assert M.reply_exact_match([_rec({}, reply="x")]) == 1.0


def test_reply_exact_match_ignores_surrounding_whitespace():
    """Hand-authored JSONL must not fail on a stray trailing space."""
    recs = [_rec({"expected_reply": " 好的。 "}, reply="好的。")]
    assert M.reply_exact_match(recs) == 1.0


def test_n_reply_annotated_counts_the_denominator():
    recs = [_rec({"expected_reply": "a"}), _rec({}), _rec({"expected_reply": "b"})]
    assert M.n_reply_annotated(recs) == 2


def test_n_reply_annotated_is_zero_when_nothing_is_annotated():
    """The guard against a vacuous 1.0: reply_exact_match reads 1.0 here, and this says why."""
    recs = [_rec({}), _rec({})]
    assert M.n_reply_annotated(recs) == 0
    assert M.reply_exact_match(recs) == 1.0
