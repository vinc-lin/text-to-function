"""`GET /trace/<id>` — one input, and every row it became.

The record pane answers "what has been happening". This answers the other question: which raw
row an input was, what that row was parsed into, and what followed. Two halves are under test.
`Store.trace` walks the five tables with every join LEFT, because a turn outlives what
triggered it; `Session.trace` adds `operation_log`, which the car owns and the store does not.

The case most worth having is the swept one. Retention deletes the raw row an hour later and
the schema SET NULLs every pointer at it, so the trace of an old turn is mostly holes — and a
view that rendered those as blanks would say "nothing arrived" about an input that did.
"""
import json

import pytest

from cli.session import Session
from ui.actions import ACTIONS, CONTROLS
from ui.server import handle_request


@pytest.fixture
def session():
    return Session.build(fake=True, llm=False, gate="permissive")


def _get(session, path):
    status, _ctype, body = handle_request(session, "GET", path, b"")
    return status, json.loads(body)


def _trace(session, turn_id):
    status, payload = _get(session, f"/trace/{turn_id}")
    assert status == 200, payload
    return payload


def _turn_of(session, kind):
    """The newest turn of one kind. Never `recent_turns()[0]`: these tests drive three inputs
    in a row, so positional access silently starts describing a different turn the moment the
    order changes."""
    return next(t["id"] for t in session.recent_turns(20) if t["kind"] == kind)


# --- the three kinds of turn -----------------------------------------------------------------

def test_a_voice_turn_traces_to_its_row_its_words_its_decision_and_its_operation(session):
    """The whole chain, on the path that has all four links. `utterance` is the parsed layer
    here — the same sentence as the payload, written a second time bare — and the operation
    comes from the car, which is the one table `intake.store` does not own."""
    session.handle("开车窗")
    tr = _trace(session, _turn_of(session, "route"))

    assert tr["turn"]["kind"] == "route"
    assert tr["raw"]["source"] == "mic" and "开车窗" in tr["raw"]["payload"]
    assert tr["raw"]["processed_at"] is not None
    assert [u["text"] for u in tr["utterance"]] == ["开车窗"]
    assert [d["chosen"] for d in tr["decisions"]] == ["open_window"]
    assert [(op["function"], op["outcome"]) for op in tr["operations"]] == [
        ("open_window", "executed")]


def test_a_scene_turn_traces_to_the_belief_it_produced_and_no_utterance(session):
    """A camera frame is parsed into a `perception` row and nothing else — nobody spoke.

    The belief is written by `SceneContext.update`, which has only an `Observation` by then and
    cannot know which row arrived, so `Intake._observe` applies the id afterwards off a
    watermark. Without that this row comes back unattributed and the trace of a scene turn
    shows a frame that produced nothing, with the belief sitting in the next table along.
    """
    session.observe("rear_occupant", "child", 0.62)
    tr = _trace(session, _turn_of(session, "scene"))

    assert tr["utterance"] == [], "nobody said anything"
    assert [(p["key"], p["value"]) for p in tr["perception"]] == [
        ("inside.rear_occupant", "child")]
    assert tr["perception"][0]["raw_id"] == tr["raw"]["id"], "the belief names its frame"
    assert any(d["subject"] == "rear_child_window_lock" for d in tr["decisions"])


def test_a_consent_turn_traces_to_the_operation_it_authorised(session):
    """好 is an answer, not a command: `Session._record_consent` writes a raw row and a turn
    and nothing in between, so the parsed layer is legitimately empty. What the turn is FOR is
    the operation it let through, and that is the link this trace has to carry."""
    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    tr = _trace(session, _turn_of(session, "consent"))

    assert tr["perception"] == [] and tr["utterance"] == [], "nothing was parsed"
    assert [d["verdict"] for d in tr["decisions"]] == ["yes"]
    assert [(op["function"], op["outcome"]) for op in tr["operations"]] == [
        ("set_window_child_lock", "executed")]


def test_a_turn_that_moved_nothing_traces_with_an_empty_operation_list(session):
    """Silence is a fact and it has to be a different one from ignorance. An observation that
    only asked a question executed nothing, and `[]` is that — not a missing key, and not an
    absent turn."""
    session.observe("rear_occupant", "child", 0.62)
    tr = _trace(session, _turn_of(session, "scene"))
    assert tr["operations"] == [] and tr["decisions"]


# --- what retention leaves behind --------------------------------------------------------------

def test_a_trace_survives_the_sweep_with_the_raw_row_marked_absent(session):
    """The case this whole read exists for, and the one most likely to be got wrong.

    An hour on, retention deletes the frame and `ON DELETE SET NULL` empties every pointer at
    it. The turn, the decisions and the operation stay — that is the point of the raw/parsed
    split — so the trace must still answer, with the raw row honestly absent rather than the
    whole turn vanishing behind an inner join.
    """
    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    turn_id = _turn_of(session, "consent")
    assert _trace(session, turn_id)["raw"] is not None, "there before the sweep"

    # Through the session's own clock and its own pump, which is the path that really applies
    # retention: `Store.apply_retention` rides on `Intake.pump`, so a sweep called by hand here
    # would be testing a call no loop makes.
    session.advance_clock(3700.0)
    session.pump()

    tr = _trace(session, turn_id)
    assert tr["turn"]["id"] == turn_id and tr["turn"]["reply"], "the turn outlived the words"
    assert tr["raw"] is None, "the frame is gone"
    assert tr["perception"] == [] and tr["utterance"] == [], "and so is the link to them"
    assert [d["verdict"] for d in tr["decisions"]] == ["yes"], "the reasoning is not"
    assert [op["function"] for op in tr["operations"]] == ["set_window_child_lock"], \
        "and neither is what the car did"


def test_the_swept_words_are_gone_from_the_trace_and_not_only_from_the_raw_row(session):
    """The sweep scrubs `decision.subject` on a route turn as well as the payload, because the
    clause IS the sentence. A trace that kept it would hand back the words retention deleted."""
    session.handle("开车窗")
    turn_id = _turn_of(session, "route")
    session.advance_clock(3700.0)
    session.pump()

    tr = _trace(session, turn_id)
    assert tr["raw"] is None
    assert [d["subject"] for d in tr["decisions"]] == [""], "the clause went with the payload"
    assert [d["chosen"] for d in tr["decisions"]] == ["open_window"], "the choice did not"


def test_a_session_that_never_wrote_the_words_down_traces_the_same_way(session):
    """`--no-raw-capture` leaves '' where the sweep leaves NULL, and the store answers `None`
    for both: two spellings of one fact, and a display must not have to tell them apart."""
    quiet = Session.build(fake=True, llm=False, gate="permissive", raw_capture=False)
    quiet.handle("开车窗")
    tr = _trace(quiet, _turn_of(quiet, "route"))
    assert tr["raw"] is not None and tr["raw"]["payload"] is None
    assert [u["text"] for u in tr["utterance"]] == [None]


# --- the route -------------------------------------------------------------------------------

def test_an_unknown_turn_is_404_and_not_a_500(session):
    status, payload = _get(session, "/trace/9999")
    assert status == 404 and "9999" in payload["error"]


def test_an_id_that_is_not_a_number_is_the_same_404(session):
    """One shape for "there is nothing there". A caller should not have to tell a malformed id
    from a missing one to know it got nothing."""
    assert _get(session, "/trace/abc")[0] == 404
    assert _get(session, "/trace/")[0] == 404
    assert _get(session, "/trace/../../etc/passwd")[0] == 404


def test_a_trace_ships_nothing_on_the_state_poll(session):
    """The snapshot goes out every 400ms. Table contents must not ride along with it, or the
    poll's cost grows with the record whether or not anyone opened a turn."""
    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    _trace(session, _turn_of(session, "consent"))

    state = _get(session, "/state")[1]
    assert "trace" not in state
    # The record pane's entries are what they were: four lines about a turn, never its rows.
    assert set(state["store"][0]) == {"id", "at", "kind", "reply", "source", "error",
                                      "heard", "decisions", "operations"}
    assert all("raw" not in entry for entry in state["store"])


def test_the_trace_is_a_read_and_has_no_entry_in_either_table(session):
    """`ACTIONS` and `CONTROLS` are the page's routes to the car and they exist for writes.
    A read in either would widen the surface they are there to bound — so the trace is a GET
    on its own route, reachable through neither table, and the two are still disjoint."""
    assert "trace" not in ACTIONS and "trace" not in CONTROLS
    assert set(ACTIONS) == {"observe", "say", "clock", "reset", "scene_llm"}
    assert set(CONTROLS) == {"set_signal", "set_bus"}
    assert set(ACTIONS) & set(CONTROLS) == set()
    assert handle_request(session, "POST", "/action/trace", b"{}")[0] == 404
    assert handle_request(session, "POST", "/control/trace", b"{}")[0] == 404


def test_a_trace_does_not_pump_the_bus(session):
    """The poll is this door's loop and pumps because it IS the loop. A read of history that
    re-stamped the bus would make looking at the past a way of changing the present — a stopped
    bus would come back to life every time somebody opened a turn."""
    session.handle("开车窗")
    turn_id = _turn_of(session, "route")
    session.set_bus(False)
    session.advance_clock(40.0)
    _trace(session, turn_id)
    assert all(row["stale"] for row in session.sensed_rows()), "still aging"
