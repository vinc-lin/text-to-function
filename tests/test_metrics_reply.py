# tests/test_metrics_reply.py
from eval.metrics import reply_action_coverage, reply_single_question, reply_nonempty_rate


def _rec(reply, responses=(), questions=()):
    return {"row": {"type": "single"}, "reply": reply,
            "responses": list(responses), "questions": list(questions)}


def test_reply_nonempty_rate():
    recs = [_rec("已将天窗开度调整到50%。"), _rec(""), _rec("   ")]
    assert reply_nonempty_rate(recs) == 1 / 3


def test_reply_action_coverage_all_present():
    assert reply_action_coverage([_rec("甲。乙。", ["甲。", "乙。"])]) == 1.0


def test_reply_action_coverage_missing_one():
    assert reply_action_coverage([_rec("甲。", ["甲。", "乙。"])]) == 0.0


def test_reply_action_coverage_skips_rows_with_no_execution():
    recs = [_rec("好的。", [None]), _rec("甲。", ["甲。"])]
    assert reply_action_coverage(recs) == 1.0


def test_reply_action_coverage_tolerates_dedup():
    """The reply dedups identical confirmations; containment must still hold."""
    assert reply_action_coverage([_rec("甲。", ["甲。", "甲。"])]) == 1.0


def test_reply_single_question_ok_when_repeated_identically():
    # _route_plan attaches the SAME question object to every unresolved clause
    assert reply_single_question([_rec("甲。问A", questions=["问A", "问A"])]) == 1.0


def test_reply_single_question_fails_on_two_distinct():
    assert reply_single_question([_rec("问A问B", questions=["问A", "问B"])]) == 0.0


def test_reply_single_question_is_not_punctuation_based():
    """build_plan_clarification contains no '？' — a '？'-counting metric would pass trivially."""
    q1 = "关于「甲」我还需要确认一下，请补充信息。"
    q2 = "关于「乙」我还需要确认一下，请补充信息。"
    assert reply_single_question([_rec(q1 + q2, questions=[q1, q2])]) == 0.0
