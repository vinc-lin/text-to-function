"""The session arbitrates two speakers: the driver's turn always finishes first."""
import pytest

from cli.session import Session


@pytest.fixture
def session():
    return Session.build(fake=True, llm=False, gate="permissive")


def test_a_scene_event_produces_a_question_and_moves_nothing(session):
    turn = session.observe("rear_occupant", "child", confidence=0.9)
    assert turn.reply == "后排有小孩，要打开儿童锁吗？"
    assert session.changed_signals() == []


def test_consent_moves_the_signal(session):
    session.observe("rear_occupant", "child", confidence=0.9)
    turn = session.handle("好")
    assert turn.reply == "已为您打开车窗儿童锁。"
    assert ("window.all", "window_child_lock", True) in session.changed_signals()


def test_a_command_after_the_question_is_routed_not_consented(session):
    """The driver ignored the question. It is abandoned and the words are routed."""
    session.observe("rear_occupant", "child", confidence=0.9)
    turn = session.handle("把主驾温度调到25度")
    assert "儿童锁" not in turn.reply
    assert not any(a == "window_child_lock" for _, a, _ in session.changed_signals())


def test_an_unknown_scene_key_is_silent_without_a_model(session):
    turn = session.observe("driver_attention", "drowsy", confidence=0.9)
    assert turn.reply == ""


def test_a_scene_llm_can_be_injected(session):
    """The eval arms need to drive S_llm without loading a second copy of Qwen3-0.6B."""
    from scene.llm import FakeSceneLLM
    s = Session.build(fake=True, llm=False, gate="permissive",
                      scene_llm=FakeSceneLLM([{"decision": "notify", "scene": "unmatched",
                                               "reason": "x",
                                               "reply_intent": "notify_driver_fatigue"}]))
    assert s.observe("driver_attention", "drowsy", 0.9).reply == "您看起来有些疲劳，请注意休息。"


def test_the_scene_engine_shares_the_session_car(session):
    """Not a copy. A scene must see what the driver's own commands did."""
    session.handle("打开车窗儿童锁")
    assert session.observe("rear_occupant", "child", 0.9).reply == "", \
        "the lock is already on, so there is nothing to ask about"
