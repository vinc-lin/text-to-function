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


def test_an_unknown_path_is_404_not_a_traceback(session):
    assert handle_request(session, "GET", "/../../etc/passwd", b"")[0] == 404


def test_an_unknown_action_is_404(session):
    assert handle_request(session, "POST", "/action/execute", b"{}")[0] == 404


def test_malformed_json_is_400_not_a_500(session):
    assert handle_request(session, "POST", "/action/observe", b"{not json")[0] == 400


def test_an_action_that_raises_returns_400_and_keeps_the_session(session):
    """A bad input must not kill an instrument holding a minute of loaded models."""
    body = json.dumps({"key": "k", "value": "v", "confidence": "not a number"}).encode()
    status, _, _ = handle_request(session, "POST", "/action/observe", body)
    assert status == 400
    assert handle_request(session, "GET", "/state", b"")[0] == 200
