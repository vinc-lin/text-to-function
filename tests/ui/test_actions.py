"""Six actions, each a call the CLI already makes."""
import pytest

from cli.session import Session
from ui.actions import ACTIONS, perform


@pytest.fixture
def session():
    return Session.build(fake=True, llm=False, gate="permissive")


def test_observe_records_a_perception_event(session):
    perform(session, "observe", {"key": "rear_occupant", "value": "child",
                                 "confidence": 0.9, "ttl": 30.0})
    assert session.context_rows()[0].expires_in <= 30.0


def test_say_routes_an_utterance(session):
    out = perform(session, "say", {"utterance": "打开车窗儿童锁"})
    assert out["reply"]


def test_consent_goes_through_the_same_lexicon_as_the_cli(session):
    """The Yes button submits 好 as an utterance. It does not call the executor, and it
    does not reach into the engine — a second route to the car with different rules is
    exactly what this design forbids."""
    perform(session, "observe", {"key": "rear_occupant", "value": "child", "confidence": 0.9})
    perform(session, "say", {"utterance": "好"})
    assert ("window.all", "window_child_lock", True) in session.changed_signals()


def test_clock_advances(session):
    perform(session, "clock", {"seconds": 30.0})
    assert session.clock_offset == 30.0


def test_reset_clears_the_car_and_the_context(session):
    perform(session, "observe", {"key": "rear_occupant", "value": "child", "confidence": 0.9})
    perform(session, "say", {"utterance": "好"})
    perform(session, "reset", {})
    assert session.changed_signals() == [] and session.context_rows() == []


def test_an_unknown_action_is_refused_not_guessed(session):
    with pytest.raises(KeyError):
        perform(session, "execute_tool_call", {"name": "unlock_doors"})


def test_the_action_table_is_the_whole_surface(session):
    """Anything reachable from the page is in this table. Adding a route without adding
    an entry is how a second path to the car gets built by accident."""
    assert set(ACTIONS) == {"observe", "say", "clock", "reset", "scene_llm"}
