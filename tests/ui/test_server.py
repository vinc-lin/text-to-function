"""Routing, encoding, and refusing what should be refused."""
import json

import pytest

from cli.session import Session
from ui.server import handle_request


@pytest.fixture
def session():
    return Session.build(fake=True, llm=False, gate="permissive")


def test_state_returns_the_snapshot(session):
    status, ctype, body = handle_request(session, "GET", "/state", b"")
    assert status == 200 and "json" in ctype
    assert "perception" in json.loads(body)


def test_the_root_serves_the_page(session):
    status, ctype, body = handle_request(session, "GET", "/", b"")
    assert status == 200 and "html" in ctype
    assert b"<" in body


def test_an_action_posts_and_returns_the_new_state(session):
    body = json.dumps({"key": "rear_occupant", "value": "child", "confidence": 0.9}).encode()
    status, _, out = handle_request(session, "POST", "/action/observe", body)
    assert status == 200
    assert json.loads(out)["state"]["pending"]["scene"] == "rear_child_window_lock"


def test_the_poll_pumps_the_bus(session):
    """The page's loop is the poll, and the bus is pumped rather than threaded — so without
    this every sensed signal would age past its max a second after load and the instrument
    would report a dead bus on a session whose bus is running."""
    session.set_signal("vehicle.all", "speed_kph", 45)
    session.advance_clock(600.0)
    row = json.loads(handle_request(session, "GET", "/state", b"")[2])["sensed"][0]
    assert row["stale"] is False and row["age"] < 1.0


def test_the_poll_cannot_make_a_stopped_bus_look_alive(session):
    """The pump is what a running bus does. A stopped one must age, or the toggle would be
    decoration and `/bus off` would mean nothing on this door."""
    session.set_signal("vehicle.all", "speed_kph", 45)
    session.set_bus(False)
    session.advance_clock(40.0)
    row = json.loads(handle_request(session, "GET", "/state", b"")[2])["sensed"][0]
    assert row["stale"] is True and row["age"] >= 40.0


def test_a_poll_that_cannot_pump_still_serves_the_page(session):
    """One failure an instrument may not have is going blank. A pump that raised on every
    poll would 500 the whole page; the cost of swallowing it is visible instead — the rows it
    did not re-stamp age, and each one says stale."""
    session.intake = None
    status, _, body = handle_request(session, "GET", "/state", b"")
    assert status == 200 and "sensed" in json.loads(body)


def test_the_bus_toggle_posts_as_a_control(session):
    status, _, out = handle_request(session, "POST", "/control/set_bus", b'{"on": false}')
    assert status == 200
    assert json.loads(out)["state"]["bus"] is False
    assert handle_request(session, "POST", "/action/set_bus", b'{"on": true}')[0] == 404


def test_an_unknown_path_is_404_not_a_traceback(session):
    assert handle_request(session, "GET", "/../../etc/passwd", b"")[0] == 404


def test_an_unknown_action_is_404(session):
    assert handle_request(session, "POST", "/action/execute", b"{}")[0] == 404


def test_a_control_posts_and_returns_the_new_state(session):
    """/control/ is its own route onto its own table — see ui/actions.py for why the
    simulator's world is not a sixth action."""
    body = json.dumps({"entity": "vehicle.all", "attribute": "speed_kph",
                       "value": 45}).encode()
    status, _, out = handle_request(session, "POST", "/control/set_signal", body)
    assert status == 200
    sensed = json.loads(out)["state"]["sensed"]
    assert {"entity": "vehicle.all", "attribute": "speed_kph"}.items() <= sensed[0].items()
    assert sensed[0]["value"] == 45.0


def test_an_unknown_control_is_404(session):
    assert handle_request(session, "POST", "/control/execute", b"{}")[0] == 404


def test_an_action_is_not_reachable_through_the_control_route(session):
    """Two tables, two routes, and neither can reach the other's names — otherwise the
    separation would be a naming convention rather than a boundary."""
    body = json.dumps({"utterance": "好"}).encode()
    assert handle_request(session, "POST", "/control/say", body)[0] == 404


def test_a_control_is_not_reachable_through_the_action_route(session):
    body = json.dumps({"entity": "vehicle.all", "attribute": "speed_kph",
                       "value": 45}).encode()
    assert handle_request(session, "POST", "/action/set_signal", body)[0] == 404


def test_a_refused_control_is_400_and_keeps_the_session(session):
    body = json.dumps({"entity": "window.all", "attribute": "window_child_lock",
                       "value": True}).encode()
    status, _, out = handle_request(session, "POST", "/control/set_signal", body)
    assert status == 400 and "not a sensed signal" in json.loads(out)["error"]
    assert handle_request(session, "GET", "/state", b"")[0] == 200
    assert session.changed_signals() == []


def test_malformed_json_is_400_not_a_500(session):
    assert handle_request(session, "POST", "/action/observe", b"{not json")[0] == 400


def test_an_action_that_raises_returns_400_and_keeps_the_session(session):
    """A bad input must not kill an instrument holding a minute of loaded models."""
    body = json.dumps({"key": "k", "value": "v", "confidence": "not a number"}).encode()
    status, _, _ = handle_request(session, "POST", "/action/observe", body)
    assert status == 400
    assert handle_request(session, "GET", "/state", b"")[0] == 200


def test_the_server_is_not_threaded():
    """Locked in by a test because the failure is SILENT, not loud.

    sim/vehicle.py opens SQLite with default thread affinity, so a handler on another
    thread cannot read the car. But ui/state.py wraps each pane defensively, so the
    exception is swallowed per pane: a threaded server would serve a snapshot showing an
    empty car and an empty log while perception and rules rendered fine. It would lie
    rather than break, and nobody would think to look at the server class.
    """
    from http.server import HTTPServer, ThreadingHTTPServer

    from ui.server import make_server

    server = make_server(None, port=0)
    try:
        assert isinstance(server, HTTPServer)
        assert not isinstance(server, ThreadingHTTPServer), \
            "a threaded server silently reports a car that never moved"
    finally:
        server.server_close()


def test_a_snapshot_from_another_thread_loses_the_car(session):
    """The evidence for the test above. This is what a threaded server would serve."""
    import threading

    from ui.state import snapshot

    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    assert snapshot(session)["car"], "the signal moved on this thread"

    seen = {}
    t = threading.Thread(target=lambda: seen.update(snapshot(session)))
    t.start()
    t.join()
    assert seen["car"] == [] and seen["log"] == [], "sqlite refused the other thread"
    assert seen["rules"], "pure-Python panes still render, which is what makes it a lie"
