"""The snapshot is the whole contract between the engine and the page."""
import pytest

from cli.session import Session
from ui.state import snapshot


@pytest.fixture
def session():
    return Session.build(fake=True, llm=False, gate="permissive")


def _rule(session, rule_id="rear_child_window_lock"):
    """The pane row FOR a rule, never `["rules"][0]`. These tests drive the child-lock rule,
    and positional access silently started describing a different rule the moment a second
    one shipped."""
    return next(r for r in snapshot(session)["rules"] if r["rule_id"] == rule_id)


def test_a_fresh_session_snapshots_cleanly(session):
    s = snapshot(session)
    assert s["perception"] == [] and s["rules"] == [] and s["car"] == []
    assert s["pending"] is None
    assert "S" in s["mode"]


def test_perception_carries_what_a_draining_bar_needs(session):
    session.observe("rear_occupant", "child", 0.9, ttl=30.0)
    row = snapshot(session)["perception"][0]
    assert row["key"] == "inside.rear_occupant" and row["value"] == "child"
    assert row["confidence"] == 0.9 and row["ttl"] == 30.0
    assert 0 < row["expires_in"] <= 30.0


def test_the_rules_pane_shows_every_rule_with_its_reason(session):
    from scene.rules import RULES
    session.observe("rear_occupant", "child", 0.6)
    # Every rule, because the pane's job is that a silence always has a reason beside it.
    assert [r["rule_id"] for r in snapshot(session)["rules"]] == [r.id for r in RULES]
    rule = _rule(session)
    assert rule["verdict"] == "near_miss" and "0.60" in rule["reason"]


def test_a_pending_question_carries_its_sentence_and_a_countdown(session):
    """PendingConsent stores the scene and the proposal, never the sentence — it is
    recovered from the rule's intent so there is only ever one copy of it."""
    session.observe("rear_occupant", "child", 0.9)
    pending = snapshot(session)["pending"]
    assert pending["scene"] == "rear_child_window_lock"
    assert pending["question"] == "后排有小孩，要打开儿童锁吗？"
    assert 0 < pending["expires_in"] <= 30.0


def test_the_pending_question_clears_once_answered(session):
    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    assert snapshot(session)["pending"] is None


def test_the_car_pane_shows_only_what_moved(session):
    assert snapshot(session)["car"] == []
    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    car = snapshot(session)["car"]
    assert {"entity": "window.all", "attribute": "window_child_lock", "value": True} in car


def _speed(session):
    return next(r for r in snapshot(session)["sensed"] if r["attribute"] == "speed_kph")


def test_the_sensed_pane_shows_a_signal_at_rest(session):
    """`car` means 'changed from seeded' and must keep meaning it. A speed of 0.0 is not a
    change, and it is also the entire answer to why the animal rule said nothing — so it
    lives under its own key rather than being squeezed into a pane that answers the other
    question."""
    s = snapshot(session)
    assert s["car"] == []
    assert {"entity": "vehicle.all", "attribute": "speed_kph", "value": 0.0,
            "unit": "kph", "min": 0.0, "max": 240.0}.items() <= _speed(session).items()


def test_the_sensed_pane_follows_the_control(session):
    session.set_signal("vehicle.all", "speed_kph", 45)
    assert _speed(session)["value"] == 45.0


def test_every_sensed_field_the_page_draws_is_present(session):
    """The page bounds a control by `min` and `max`, labels it with `unit`, prints `age` and
    reads `stale` to decide whether to show the value at all. A field named something else
    here renders an empty control forever and nothing raises."""
    for row in snapshot(session)["sensed"]:
        assert set(row) == {"entity", "attribute", "value", "unit", "min", "max",
                            "age", "stale"}


def test_a_sensed_row_carries_its_age(session):
    """A number, not a phrase: the pane renders it and the terminal does not, so a string
    formatted here would be a second opinion about how an age reads."""
    session.set_signal("vehicle.all", "speed_kph", 45)
    row = _speed(session)
    assert isinstance(row["age"], float) and row["age"] < 1.0
    assert row["stale"] is False


def test_a_stale_row_says_so(session):
    """The pane must be able to read as STALE rather than as a value. A stale 45 drawn as
    '45 kph' is exactly the lie this discipline exists to remove — no rule can read it."""
    session.set_signal("vehicle.all", "speed_kph", 45)
    session.set_bus(False)
    session.advance_clock(40.0)
    row = _speed(session)
    assert row["stale"] is True
    assert row["age"] >= 40.0
    # The value itself is still the last thing the bus said, and the control is still bounded
    # by it. `stale` is what says nothing reads it.
    assert row["value"] == 45.0


def test_the_pane_and_the_rules_agree_about_staleness(session):
    """One world, one answer. A pane comparing ages for itself would be a second belief about
    one bus, and the one nothing consults is the one that is wrong."""
    session.set_signal("vehicle.all", "speed_kph", 45)
    session.set_bus(False)
    session.advance_clock(40.0)
    turn = session.observe("outside.front_object", "animal", 0.9)
    report = next(r for r in turn.rules if r.rule_id == "animal_ahead")
    assert "stale" in report.reason
    assert _speed(session)["stale"] is True


def test_a_live_bus_never_shows_a_stale_row(session):
    """Ten minutes forward with the bus running. A real bus does not go quiet because time
    passed, and the pane must not claim it did."""
    session.set_signal("vehicle.all", "speed_kph", 45)
    session.advance_clock(600.0)
    session.pump()
    row = _speed(session)
    assert row["stale"] is False and row["age"] < 1.0


def test_the_bus_state_is_in_the_snapshot(session):
    """The toggle needs the state, and it cannot be derived from the rows: a stopped bus whose
    values are still inside their max age looks exactly like a running one."""
    assert snapshot(session)["bus"] is True
    session.set_bus(False)
    assert snapshot(session)["bus"] is False


def test_the_sensed_pane_returns_to_rest_on_a_reset(session):
    session.set_signal("vehicle.all", "speed_kph", 45)
    session.reset()
    row = next(r for r in snapshot(session)["sensed"] if r["attribute"] == "speed_kph")
    assert row["value"] == 0.0


def test_the_log_carries_the_cause_of_a_refusal(session):
    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    session.handle("开车窗")
    refused = [r for r in snapshot(session)["log"] if r["outcome"] == "refused"]
    assert refused and refused[0]["detail"] == "车窗儿童锁已开启"


def test_the_record_pane_is_the_history_the_terminal_prints(session):
    """One session, two doors, one history. The pane is `Session.recent_turns` unreshaped —
    a page that re-derived it could show a different account of the same run."""
    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    assert snapshot(session)["store"] == session.recent_turns()


def test_the_record_carries_the_four_things_it_draws(session):
    """heard · decided · did · said. A field named something else renders an empty line
    forever and nothing raises — the pane swallows its own exceptions."""
    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    consent = next(t for t in snapshot(session)["store"] if t["kind"] == "consent")
    assert set(consent) == {"id", "at", "kind", "reply", "source", "error", "heard",
                            "decisions", "operations"}
    assert consent["reply"] == "已为您打开车窗儿童锁。"
    assert [op["function"] for op in consent["operations"]] == ["set_window_child_lock"]


def test_the_record_pane_is_read_only(session):
    """It has no entry in either table, and that is the point: the record is written by the
    door as each input goes through it, never by the page."""
    from ui.actions import ACTIONS, CONTROLS
    assert not [name for name in list(ACTIONS) + list(CONTROLS) if "store" in name]


def test_the_clock_offset_is_reported(session):
    session.advance_clock(45.0)
    assert snapshot(session)["clock_offset"] == 45.0


def test_the_snapshot_is_json_serialisable(session):
    """It crosses the wire as JSON; a stray dataclass or Enum would 500 the page."""
    import json
    session.observe("rear_occupant", "child", 0.9)
    json.dumps(snapshot(session))


def test_the_snapshot_never_raises_on_a_half_built_session(session):
    """The page polls several times a second, including during a reset. One exception
    turns the whole instrument into a 500 and hides the state it exists to show."""
    session.scene = None
    s = snapshot(session)
    assert s["perception"] == [] and s["rules"] == []


def test_a_rule_carries_the_bands_the_page_draws(session):
    """The page shows a near-miss as a position between the floor and the threshold. The
    alternative was regexing `reason`, which scene/rules.py documents as diagnostics for a
    developer at a terminal and is therefore free to change wording without warning."""
    session.observe("rear_occupant", "child", 0.6)
    rule = _rule(session)
    assert rule["floor"] == 0.5 and rule["threshold"] == 0.8
    assert rule["observed_keys"] == ["inside.rear_occupant"]


def test_a_report_with_no_matching_rule_invents_no_bands(session):
    """observe() records a '—' row when the engine itself raised. It has no bands to draw."""
    session.scene.world = None            # make evaluate() blow up inside observe()
    session.observe("rear_occupant", "child", 0.9)
    rule = snapshot(session)["rules"][0]
    assert rule["verdict"] == "error"
    assert rule["threshold"] is None and rule["observed_keys"] == []


def test_the_scene_fallback_flag_is_structured_not_parsed(session):
    """`mode` joins its parts with ' · ', so a page testing it for 'S_llm' also matches 'S'
    — the toggle would read as on at exactly the moment the fallback is off."""
    from scene.llm import FakeSceneLLM
    assert snapshot(session)["scene_llm"] is False
    session.attach_scene_llm(FakeSceneLLM([]))
    assert snapshot(session)["scene_llm"] is True
