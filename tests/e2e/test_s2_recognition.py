"""S2 — segmented intent recognition. Utterance -> spans -> function(s) + parameters.

Expected replies and dispatches are MEASURED against the fixture catalog, not assumed.
"""
import pytest
from .conftest import build_pipeline

WINDOW = "已为您打开当前区域车窗。"
WINDOW_CLOSED = "已为您关闭当前区域车窗。"   # the confirmation now states the direction
TEMP25 = "已将当前区域温度设置为25°C。"
TEMP22 = "已将当前区域温度设置为22°C。"
FAN3 = "已将当前区域风速设置为3档。"
REJECT = "抱歉，我不太确定您的意思，可以换个说法吗？"

# (case id, utterance, expected reply, expected dispatches)
GREEN = [
    ("S2-01", "把空调调到25度", TEMP25, [("set_temperature", {"temperature": 25.0})]),
    ("S2-02", "温度设成22度", TEMP22, [("set_temperature", {"temperature": 22.0})]),
    ("S2-03", "风速调到三档", FAN3, [("set_fan_speed", {"level": 3})]),
    ("S2-04", "开车窗", WINDOW, [("open_window", {"is_open": True})]),
    ("S2-05", "关闭车窗", WINDOW_CLOSED, [("open_window", {"is_open": False})]),
    ("S2-06", "开车窗,温度调到25度", WINDOW + TEMP25,
     [("open_window", {"is_open": True}), ("set_temperature", {"temperature": 25.0})]),
    ("S2-08", "开车窗,风速调到三档,温度调到25度", WINDOW + FAN3 + TEMP25,
     [("open_window", {"is_open": True}), ("set_fan_speed", {"level": 3}),
      ("set_temperature", {"temperature": 25.0})]),
    ("S2-13", "今天天气怎么样", REJECT, []),
]


@pytest.mark.parametrize("case_id,utterance,expected_reply,expected_calls", GREEN,
                         ids=[c[0] for c in GREEN])
def test_s2_recognition(case_id, utterance, expected_reply, expected_calls):
    pipe, ex = build_pipeline()
    result = pipe.route(utterance)
    assert result.reply == expected_reply
    assert ex.dispatched == expected_calls


def test_s2_07_context_is_suppressed_and_absent_from_reply():
    """Narration must not act, and must not be spoken back."""
    pipe, ex = build_pipeline()
    result = pipe.route("我有点热,温度调到25度")
    assert result.reply == TEMP25
    assert "我有点热" not in result.reply
    assert ex.dispatched == [("set_temperature", {"temperature": 25.0})]
    assert result.plan is not None          # context present -> plan path


def test_s2_09_relative_without_value_does_not_execute():
    """A relative op with no state reaches MEDIUM with no LLM: it must not execute, and it
    must not silently succeed either. (Spec 5's falsely-affirmative fix.)

    It used to answer 抱歉，这个操作没能完成。 The medium band now asks for the slot it could
    name, which executes nothing and is something the driver can actually answer.
    """
    pipe, ex = build_pipeline()
    result = pipe.route("温度调高一点")
    assert ex.dispatched == []
    assert result.reply == "您想设置到多少度？"


def test_s2_11_negation_must_not_invert_the_action():
    """别关车窗 is "don't close the window" — which is not "open the window".

    This was red: polarity was pure substring matching with no negation awareness, so the
    system closed the window it was told not to close. It now declines to guess and asks,
    because a negated instruction says what the driver does NOT want and leaves what they
    do want unstated.
    """
    pipe, ex = build_pipeline()
    result = pipe.route("别关车窗")
    assert ("open_window", {"is_open": False}) not in ex.dispatched
    assert ex.dispatched == []                       # nor the inverse — nothing is guessed
    assert result.reply == "您想打开还是关闭车窗？"
