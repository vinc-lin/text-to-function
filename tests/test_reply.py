# tests/test_reply.py
import pytest
from t2f.types import (RouteResult, ClauseResult, Decision, Band,
                       ClarificationRequest, ValidationError)
from t2f.reply import compose_reply


def _clause(response=None, question=None, errors=None):
    clar = ClarificationRequest(question=question) if question is not None else None
    return ClauseResult(clause="x", decision=Decision(Band.HIGH, "f", []),
                        response=response, clarification=clar,
                        validation_errors=list(errors or []))


def _result(*clauses):
    return RouteResult(utterance="u", clauses=list(clauses))


def test_single_confirmation_passes_through():
    res = _result(_clause(response="已将天窗开度调整到50%。"))
    assert compose_reply(res) == "已将天窗开度调整到50%。"


def test_multiple_confirmations_are_sentence_joined():
    res = _result(_clause(response="已为您调整车窗儿童锁状态。"),
                  _clause(response="已将主驾车窗开度调整到40%。"),
                  _clause(response="已将天窗开度调整到50%。"))
    assert compose_reply(res) == ("已为您调整车窗儿童锁状态。"
                                  "已将主驾车窗开度调整到40%。"
                                  "已将天窗开度调整到50%。")


def test_duplicate_confirmations_are_deduped():
    res = _result(_clause(response="已为您调整当前区域车窗状态。"),
                  _clause(response="已为您调整当前区域车窗状态。"))
    assert compose_reply(res) == "已为您调整当前区域车窗状态。"


def test_clause_order_is_preserved():
    res = _result(_clause(response="甲。"), _clause(response="乙。"))
    assert compose_reply(res) == "甲。乙。"


def test_confirmation_without_terminator_gets_one():
    res = _result(_clause(response="已执行set_fan_speed"))
    assert compose_reply(res) == "已执行set_fan_speed。"


def test_clause_that_produced_nothing_is_not_reported_as_success():
    """A clause that neither spoke nor asked is an UNRESOLVED request (e.g. MEDIUM band with
    no LLM). Saying 好的。 there would falsely confirm work that never happened."""
    assert compose_reply(_result(_clause())) == "抱歉，这个操作没能完成。"


def test_ack_only_when_there_were_no_clauses():
    assert compose_reply(RouteResult(utterance="u", clauses=[])) == "好的。"


def test_confirmation_then_single_question():
    res = _result(_clause(response="已为您调整车窗儿童锁状态。"),
                  _clause(question="关于「温度调高」我还需要确认一下，请补充信息。"))
    assert compose_reply(res) == ("已为您调整车窗儿童锁状态。"
                                  "关于「温度调高」我还需要确认一下，请补充信息。")


def test_repeated_plan_question_collapses_to_one():
    # _route_plan attaches the SAME ClarificationRequest to every unresolved clause
    q = "关于「温度调高」「天窗开到一半」我还需要确认一下，请补充信息。"
    res = _result(_clause(response="已为您调整车窗儿童锁状态。"),
                  _clause(question=q), _clause(question=q))
    assert compose_reply(res) == "已为您调整车窗儿童锁状态。" + q
    assert compose_reply(res).count(q) == 1


def test_distinct_questions_first_wins():
    res = _result(_clause(question="问甲？"), _clause(question="问乙？"))
    assert compose_reply(res) == "问甲？"


def test_question_only():
    res = _result(_clause(question="抱歉，我不太确定您的意思，可以换个说法吗？"))
    assert compose_reply(res) == "抱歉，我不太确定您的意思，可以换个说法吗？"


def test_question_without_terminator_gets_one():
    res = _result(_clause(question="请补充信息"))
    assert compose_reply(res) == "请补充信息。"


def test_hard_failure_line_when_no_question():
    res = _result(_clause(errors=[ValidationError("out_of_range", "temperature 99 > 32")]))
    assert compose_reply(res) == "抱歉，这个操作没能完成。"


def test_hard_failure_appended_after_confirmations():
    res = _result(_clause(response="已为您调整当前区域车窗状态。"),
                  _clause(errors=[ValidationError("out_of_range", "bad")]))
    assert compose_reply(res) == "已为您调整当前区域车窗状态。抱歉，这个操作没能完成。"


def test_question_suppresses_the_failure_line():
    res = _result(_clause(errors=[ValidationError("out_of_range", "bad")]),
                  _clause(question="请补充信息。"))
    reply = compose_reply(res)
    assert reply == "请补充信息。"
    assert "没能完成" not in reply


def test_single_clause_with_both_error_and_question_asks_only():
    """LLMResolver (pipeline.py:73-75) puts errors AND a clarification on one clause."""
    res = _result(_clause(question="您想设置到多少度？",
                          errors=[ValidationError("missing_required", "missing temperature")]))
    reply = compose_reply(res)
    assert reply == "您想设置到多少度？"
    assert "没能完成" not in reply


def test_whitespace_response_treated_as_absent():
    """Whitespace-only response counts as not having spoken, so the clause is unresolved."""
    assert compose_reply(_result(_clause(response="   "))) == "抱歉，这个操作没能完成。"


def test_blank_question_treated_as_absent():
    """Blank question counts as not having asked, so the clause is unresolved."""
    assert compose_reply(_result(_clause(question=""))) == "抱歉，这个操作没能完成。"


def test_none_question_treated_as_absent():
    res = _result(ClauseResult(clause="x", decision=Decision(Band.LOW, None, []),
                               clarification=ClarificationRequest(question=None)))
    assert compose_reply(res) == "抱歉，这个操作没能完成。"


def test_missing_decision_is_never_read():
    res = _result(ClauseResult(clause="x", decision=None, response="已执行。"))
    assert compose_reply(res) == "已执行。"


@pytest.mark.parametrize("res", [
    RouteResult(utterance="", clauses=[]),
    RouteResult(utterance="u", clauses=None),
    _result(_clause()),
    _result(_clause(response=None, question=None, errors=[])),
])
def test_composer_never_raises(res):
    assert isinstance(compose_reply(res), str)
    assert compose_reply(res)


def test_unresolved_clause_alongside_a_confirmation_is_reported():
    """开车窗，温度调到25度 with the 2nd span MEDIUM: the driver must not be told only about
    the window and left believing the temperature was set."""
    res = _result(_clause(response="已为您调整当前区域车窗状态。"), _clause())
    reply = compose_reply(res)
    assert reply == "已为您调整当前区域车窗状态。抱歉，这个操作没能完成。"


def test_question_still_suppresses_the_failure_line():
    """An actionable question outranks the failure line — unchanged by this fix."""
    res = _result(_clause(), _clause(question="您想设置到多少度？"))
    assert compose_reply(res) == "您想设置到多少度？"
