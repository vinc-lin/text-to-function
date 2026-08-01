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
    from scene.rules import RULES
    turn = session.observe("rear_occupant", "child", 0.6)
    # Every rule's report travels, not just the one this observation is about — a rule missing
    # from the turn is a silence the CLI cannot explain. Selected by id rather than position,
    # which quietly started describing a different rule the moment a second one shipped.
    assert [r.rule_id for r in turn.rules] == [r.id for r in RULES]
    child = next(r for r in turn.rules if r.rule_id == "rear_child_window_lock")
    assert child.verdict == "near_miss" and "0.60" in child.reason


# --- the /scene command's own argument handling -------------------------------------------

def _run(session, line):
    """Drive the command dispatcher the way a typed line reaches it."""
    from cli.__main__ import _scene
    _scene(session, line.split()[1:])


def test_ttl_typed_at_the_prompt_reaches_the_observation(session, capsys):
    """It used to be accepted in silence and ignored, so the guide documented an expiry
    demonstration that never expired."""
    _run(session, "/scene rear_occupant=child conf=0.9 ttl=30")
    capsys.readouterr()
    assert next(r for r in session.context_rows()).expires_in <= 30.0


def test_an_unknown_option_is_refused_not_dropped(session, capsys):
    """A command that quietly discards what you typed is worse than one that cannot do the
    thing — you cannot tell the difference between ignored and unsupported."""
    _run(session, "/scene rear_occupant=child confidence=0.9")
    out = capsys.readouterr().out
    assert "unknown option" in out and "usage:" in out
    assert session.context_rows() == [], "nothing should have been observed"


def test_a_mistyped_number_prints_usage_and_survives(session, capsys):
    _run(session, "/scene rear_occupant=child conf=abc")
    assert "usage:" in capsys.readouterr().out
    assert session.context_rows() == []


# --- what an observation may not be -------------------------------------------------------

@pytest.mark.parametrize("kwargs,why", [
    ({"key": "", "value": "child", "confidence": 0.9}, "empty key"),
    ({"key": "   ", "value": "child", "confidence": 0.9}, "whitespace key"),
    ({"key": "k", "value": "", "confidence": 0.9}, "empty value"),
    ({"key": "k", "value": "v", "confidence": 7.0}, "confidence above 1"),
    ({"key": "k", "value": "v", "confidence": -0.5}, "negative confidence"),
    ({"key": "k", "value": "v", "confidence": 0.9, "ttl": 0}, "zero ttl"),
    ({"key": "k", "value": "v", "confidence": 0.9, "ttl": -5}, "negative ttl"),
])
def test_a_malformed_observation_is_refused(session, kwargs, why):
    """Validated on Session so the terminal and the browser cannot disagree.

    The confidence range carries the weight: every rule's floor and threshold live in
    [0, 1], so an observation at 7.0 clears any band trivially and the instrument would
    report a confident MATCH built on a number the design has no meaning for.
    """
    with pytest.raises(ValueError):
        session.observe(**kwargs)
    assert session.context_rows() == [], why


def test_the_boundary_values_are_allowed(session):
    """0.0 and 1.0 are legitimate confidences, and a rule floor of 0.0 depends on it."""
    session.observe("a", "x", 0.0)
    session.observe("b", "y", 1.0)
    assert len(session.context_rows()) == 2


def test_the_session_clock_agrees_with_the_car(session):
    """The blocker that would have made staleness a silent no-op.

    SqliteVehicle stamps updated_at with time.time(); a session on monotonic subtracted
    ~756 thousand from ~1.79 billion and got an age of about minus 1.78 billion seconds.
    Negative is under every max_age, so every signal read LIVE and the discipline never
    fired -- failing open, in the one direction that looks wired while doing nothing.
    """
    session.set_signal("vehicle.all", "speed_kph", 45.0)
    age = session.car.signal_age("vehicle.all", "speed_kph", session._now())
    assert age is not None
    assert 0 <= age < 5.0, f"session clock and car clock disagree: age {age}"


def test_a_stale_signal_reads_as_absent_through_the_world(session):
    """End to end on the session's own clock, which is what Task 4 will rely on."""
    from intake.hub import WorldView
    session.set_signal("vehicle.all", "speed_kph", 45.0)
    world = WorldView(session.scene.context, session.car)
    assert world.signal("vehicle.all", "speed_kph", session._now()) == 45.0
    session.advance_clock(5.0)
    assert world.signal("vehicle.all", "speed_kph", session._now()) is None
    status, why = world.signal_status("vehicle.all", "speed_kph", session._now())
    assert status == "stale" and "2.0" in why
