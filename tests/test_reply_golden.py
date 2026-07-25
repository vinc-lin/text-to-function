# tests/test_reply_golden.py
"""Exact replies rendered through REAL catalog cards with fixed tool calls.
Catches drift in either respond.py's templates or reply.py's composition."""
from pathlib import Path
from t2f.cards import load_catalog
from t2f.types import (RouteResult, ClauseResult, Decision, Band, ToolCall,
                       ClarificationRequest, ValidationError)
from t2f.respond import render_response
from t2f.reply import compose_reply

CATALOG = Path(__file__).resolve().parents[1] / "data" / "catalog"
CARDS = {c.name: c for c in load_catalog(CATALOG)}


def _executed(name, params):
    tc = ToolCall(name=name, parameters=params)
    return ClauseResult(clause=name, decision=Decision(Band.HIGH, name, []),
                        tool_call=tc, response=render_response(CARDS[name], tc))


def _asked(question):
    return ClauseResult(clause="q", decision=Decision(Band.MEDIUM, None, []),
                        clarification=ClarificationRequest(question=question))


def _failed():
    return ClauseResult(clause="bad", decision=Decision(Band.HIGH, "set_temperature", []),
                        validation_errors=[ValidationError("out_of_range", "temperature 99 > 32")])


def _reply(*clauses):
    return compose_reply(RouteResult(utterance="u", clauses=list(clauses)))


def test_golden_canonical_three_actions():
    assert _reply(
        _executed("set_window_child_lock", {"enabled": True}),
        _executed("set_window_position", {"percent": 40, "position": "driver"}),
        _executed("set_sunroof_position", {"percent": 50}),
    ) == "已为您调整车窗儿童锁状态。已将主驾车窗开度调整到40%。已将天窗开度调整到50%。"


def test_golden_partial_failure():
    assert _reply(
        _executed("set_window_child_lock", {"enabled": True}),
        _asked("关于「温度调高」我还需要确认一下，请补充信息。"),
    ) == "已为您调整车窗儿童锁状态。关于「温度调高」我还需要确认一下，请补充信息。"


def test_golden_single_temperature_no_position():
    assert _reply(_executed("set_temperature", {"temperature": 25})) == \
        "已将当前区域温度设置为25°C。"


def test_golden_single_temperature_with_position():
    assert _reply(_executed("set_temperature", {"temperature": 26, "position": "passenger"})) == \
        "已将副驾温度设置为26°C。"


def test_golden_low_confidence_only():
    assert _reply(_asked("抱歉，我不太确定您的意思，可以换个说法吗？")) == \
        "抱歉，我不太确定您的意思，可以换个说法吗？"


def test_golden_duplicate_actions_deduped():
    assert _reply(
        _executed("set_sunroof_position", {"percent": 50}),
        _executed("set_sunroof_position", {"percent": 50}),
    ) == "已将天窗开度调整到50%。"


def test_golden_hard_failure_only():
    assert _reply(_failed()) == "抱歉，这个操作没能完成。"


def test_golden_nothing_acted():
    assert _reply(ClauseResult(clause="x", decision=Decision(Band.LOW, None, []))) == "好的。"
