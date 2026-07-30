"""The snapshot is the whole contract between the engine and the page."""
import pytest

from cli.session import Session
from ui.state import snapshot


@pytest.fixture
def session():
    return Session.build(fake=True, llm=False, gate="permissive")


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
    session.observe("rear_occupant", "child", 0.6)
    rule = snapshot(session)["rules"][0]
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


def test_the_log_carries_the_cause_of_a_refusal(session):
    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    session.handle("开车窗")
    refused = [r for r in snapshot(session)["log"] if r["outcome"] == "refused"]
    assert refused and refused[0]["detail"] == "车窗儿童锁已开启"


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
    rule = snapshot(session)["rules"][0]
    assert rule["floor"] == 0.5 and rule["threshold"] == 0.8
    assert rule["observed_keys"] == ["inside.rear_occupant"]


def test_a_report_with_no_matching_rule_invents_no_bands(session):
    """observe() records a '—' row when the engine itself raised. It has no bands to draw."""
    session.scene.facts = None            # make evaluate() blow up inside observe()
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
