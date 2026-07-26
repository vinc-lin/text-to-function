from t2f.types import FunctionCard, ParamSpec
from research.dialog import PendingState, SessionState
from t2f.lexical import extract_features
from research.dialog import FollowUpResolver

CARD = FunctionCard("set_temperature", "climate", "温度",
    params=[ParamSpec("temperature", "number", required=True, minimum=16, maximum=32),
            ParamSpec("position", "enum", required=True, enum=["driver", "passenger"])],
    response_template="已将{position}温度设置为{temperature}°C。")
R = FollowUpResolver({"set_temperature": CARD})

def _pending():
    return SessionState(pending=PendingState("set_temperature", {"temperature": 25}, ["position"]))

def test_is_followup_true_for_short_answer():
    assert R.is_followup(_pending(), "副驾") is True
    assert R.is_followup(SessionState(), "副驾") is False

def test_resolve_completes_and_executes():
    res, sess = R.resolve(_pending(), "副驾", extract_features("副驾"))
    assert res.tool_call is not None
    assert res.tool_call.parameters == {"temperature": 25, "position": "passenger"}
    assert res.response is not None and sess.pending is None

def test_resolve_reclarifies_when_still_missing():
    res, sess = R.resolve(_pending(), "嗯", extract_features("嗯"))   # no position extractable
    assert res.clarification is not None and sess.pending is not None and sess.turn_count == 1
