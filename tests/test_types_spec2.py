from t2f.types import LLMResult, ToolCall
from research.dialog import SessionState, PendingState

def test_llmresult_defaults():
    r = LLMResult()
    assert r.tool_call is None and r.clarification is None and r.error is None and r.raw == ""
    r2 = LLMResult(tool_call=ToolCall("set_volume", {"level": 3}))
    assert r2.tool_call.name == "set_volume"

def test_sessionstate_defaults():
    s = SessionState()
    assert s.pending is None and s.turn_count == 0
    s2 = SessionState(pending=PendingState("f", {"a": 1}, ["b"]), turn_count=1)
    assert s2.pending.pending_function == "f" and s2.turn_count == 1
