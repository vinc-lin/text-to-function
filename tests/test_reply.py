# tests/test_reply.py
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


def test_nothing_acted_returns_ack():
    assert compose_reply(_result(_clause())) == "好的。"
