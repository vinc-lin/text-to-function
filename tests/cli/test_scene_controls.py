"""The four controls the session has to expose before the CLI can offer them."""
import pytest

from cli.session import Session


@pytest.fixture
def session():
    return Session.build(fake=True, llm=False, gate="permissive")


def test_a_namespaced_key_is_taken_as_written(session):
    session.observe("outside.weather", "rain", 0.9)
    assert any(r.key == "outside.weather" for r in session.context_rows())


def test_a_bare_key_still_gets_the_cabin_namespace(session):
    """Backwards compatible: /scene rear_occupant=child kept working through this change."""
    session.observe("rear_occupant", "child", 0.9)
    assert any(r.key == "inside.rear_occupant" for r in session.context_rows())


def test_context_rows_carry_what_the_display_needs(session):
    session.observe("rear_occupant", "child", 0.87, source="cabin_cam")
    row = next(r for r in session.context_rows() if r.key == "inside.rear_occupant")
    assert row.value == "child" and row.confidence == 0.87
    assert row.source == "cabin_cam" and row.expires_in > 0


def test_a_custom_ttl_is_honoured(session):
    session.observe("rear_occupant", "child", 0.9, ttl=10.0)
    assert next(r for r in session.context_rows()).expires_in <= 10.0


def test_the_clock_offset_expires_an_observation(session):
    """A 300s ttl cannot be demonstrated by hand without this."""
    session.observe("rear_occupant", "child", 0.9, ttl=30.0)
    assert session.context_rows()
    session.advance_clock(31.0)
    assert session.context_rows() == []


def test_the_clock_offset_elapses_a_cooldown(session):
    assert session.observe("rear_occupant", "child", 0.9).reply
    session.handle("不用")
    assert session.observe("rear_occupant", "child", 0.9).reply == ""
    session.advance_clock(121.0)
    assert session.observe("rear_occupant", "child", 0.9).reply == "后排有小孩，要打开儿童锁吗？"


def test_the_clock_offset_accumulates(session):
    session.advance_clock(10.0)
    session.advance_clock(5.0)
    assert session.clock_offset == 15.0


def test_a_scene_fallback_can_be_attached(session):
    from scene.llm import FakeSceneLLM
    session.attach_scene_llm(FakeSceneLLM([{"decision": "notify", "scene": "unmatched",
                                            "reason": "x",
                                            "reply_intent": "notify_driver_fatigue"}]))
    assert session.observe("driver_state", "drowsy", 0.9).reply == "您看起来有些疲劳，请注意休息。"


def test_detaching_the_fallback_returns_to_silence(session):
    from scene.llm import FakeSceneLLM
    session.attach_scene_llm(FakeSceneLLM([{"decision": "notify", "scene": "unmatched",
                                            "reason": "x",
                                            "reply_intent": "notify_driver_fatigue"}]))
    session.attach_scene_llm(None)
    assert session.observe("driver_state", "drowsy", 0.9).reply == ""
    assert "no model attached" in session.scene.fallback_note()


def test_attaching_a_fallback_keeps_the_car_and_the_context(session):
    """Same principle as /llm and /gate: switching a mode must not reset the vehicle."""
    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    before = session.changed_signals()
    session.attach_scene_llm(None)
    assert session.changed_signals() == before
    assert before != []


def test_a_turn_carries_the_rule_reports(session):
    turn = session.observe("rear_occupant", "child", 0.6)
    assert turn.rules and turn.rules[0].verdict == "near_miss"
    assert "0.60" in turn.rules[0].reason
