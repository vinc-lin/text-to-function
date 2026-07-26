# tests/test_reply_exec.py
"""Requirement 4b's third failure branch: the car understood, tried, and refused.

Distinct from the other two branches ("I didn't understand" → a clarification question,
"a parameter is unusable" → the validation table, still gap 2). Here the detail string is
authored by the vehicle for the driver, so the reply speaks it verbatim.
"""
from t2f.reply import compose_reply
from t2f.types import RouteResult, ClauseResult, Decision, Band, ValidationError


def _clause(exec_error=None, response=None):
    return ClauseResult(clause="x", decision=Decision(Band.HIGH, "f", []),
                        response=response, exec_error=exec_error)


def test_refusal_states_the_cause():
    r = RouteResult(utterance="u", clauses=[_clause(ValidationError("precondition_failed", "空调尚未开启"))])
    assert "空调尚未开启" in compose_reply(r)


def test_refusal_never_claims_success():
    r = RouteResult(utterance="u", clauses=[_clause(ValidationError("device_unavailable", "执行器无响应"))])
    reply = compose_reply(r)
    assert "已" not in reply and "执行器无响应" in reply


def test_a_refusal_beside_a_success_reports_both():
    r = RouteResult(utterance="u", clauses=[
        _clause(response="已开启车窗。"),
        _clause(ValidationError("precondition_failed", "空调尚未开启"))])
    reply = compose_reply(r)
    assert "已开启车窗。" in reply and "空调尚未开启" in reply


def test_exec_error_without_detail_falls_back_to_the_generic_line():
    r = RouteResult(utterance="u", clauses=[_clause(ValidationError("exec_failed", ""))])
    assert compose_reply(r) == "抱歉，这个操作没能完成。"
