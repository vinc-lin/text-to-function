"""S4a — on success, inform the user of the completed action."""
import pytest
from .conftest import build_pipeline

WINDOW = "已为您调整当前区域车窗状态。"
TEMP25 = "已将当前区域温度设置为25°C。"
FAN3 = "已将当前区域风速设置为3档。"


def test_s4a_01_confirmation_states_the_value_set():
    pipe, _ = build_pipeline()
    assert pipe.route("把空调调到25度").reply == TEMP25


def test_s4a_02_confirmation_states_a_level():
    pipe, _ = build_pipeline()
    assert pipe.route("风速调到三档").reply == FAN3


def test_s4a_03_two_confirmations_are_sentence_joined():
    pipe, _ = build_pipeline()
    reply = pipe.route("开车窗,温度调到25度").reply
    assert reply == WINDOW + TEMP25
    assert reply.count(WINDOW) == 1


def test_s4a_04_every_dispatched_call_is_mentioned():
    """Coverage invariant: nothing is actuated silently.

    Distinct from S2-08, which pins the exact composed string. This one pins the exact
    dispatch list and then asserts each call has a corresponding confirmation — the
    property that would break if a fourth action were dispatched without being spoken.
    """
    pipe, ex = build_pipeline()
    result = pipe.route("开车窗,风速调到三档,温度调到25度")
    assert ex.dispatched == [("open_window", {"is_open": True}),
                             ("set_fan_speed", {"level": 3}),
                             ("set_temperature", {"temperature": 25.0})]
    for confirmation in (WINDOW, FAN3, TEMP25):
        assert confirmation in result.reply


@pytest.mark.xfail(strict=True,
                   reason="gap 4: render_response humanizes only `position`, so 43 of 92 catalog "
                          "cards confirm an action without stating the value. Opening and closing "
                          "the window produce byte-identical replies.")
def test_s4a_07_boolean_action_states_on_or_off():
    pipe, _ = build_pipeline()
    opened = pipe.route("开车窗").reply
    pipe2, _ = build_pipeline()
    closed = pipe2.route("关闭车窗").reply
    assert opened != closed
