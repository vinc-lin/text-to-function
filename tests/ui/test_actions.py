"""Five actions and one control, each a call the CLI already makes."""
import pytest

from cli.session import Session
from ui.actions import ACTIONS, CONTROLS, perform, perform_control


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


# --- the controls, which are not actions --------------------------------------------------

def test_the_control_table_holds_the_two_world_controls():
    """Both entries are the simulated world being told what it is doing, never the Central
    Model being asked to do something. Setting a sensed signal says what the car is doing;
    setting the bus says whether anything is still saying it. Neither is in the catalog and
    neither goes near the executor — a third entry needs the same argument made again."""
    assert set(CONTROLS) == {"set_signal", "set_bus"}


def test_the_two_tables_are_disjoint():
    """The separation is the design: an action is something the model performs under every
    check the executor applies; a control is the world changing around it. A name in both
    would mean the page had one route wearing two justifications."""
    assert set(ACTIONS) & set(CONTROLS) == set()


def test_set_signal_reaches_the_car(session):
    perform_control(session, "set_signal", {"entity": "vehicle.all",
                                            "attribute": "speed_kph", "value": 45})
    assert session.car.get_signal("vehicle.all", "speed_kph") == 45.0


def test_a_control_refuses_an_actuated_signal(session):
    """The page cannot be a laxer door onto the car than the terminal is: the same
    `Session.set_signal` refuses both."""
    with pytest.raises(ValueError, match="not a sensed signal"):
        perform_control(session, "set_signal", {"entity": "window.all",
                                                "attribute": "window_child_lock",
                                                "value": True})
    assert session.changed_signals() == []


def test_a_control_refuses_a_value_out_of_range(session):
    with pytest.raises(ValueError, match="240"):
        perform_control(session, "set_signal", {"entity": "vehicle.all",
                                                "attribute": "speed_kph", "value": 300})
    assert session.car.get_signal("vehicle.all", "speed_kph") == 0.0


def test_the_bus_control_stops_and_starts_the_publisher(session):
    assert session.bus_publishing() is True
    perform_control(session, "set_bus", {"on": False})
    assert session.bus_publishing() is False
    perform_control(session, "set_bus", {"on": True})
    assert session.bus_publishing() is True


def test_stopping_the_bus_from_the_page_makes_a_signal_go_stale(session):
    """The same story the terminal tells with /bus off, through the other door. Nothing here
    changes the value — only whether anything is still saying it."""
    perform_control(session, "set_signal", {"entity": "vehicle.all",
                                            "attribute": "speed_kph", "value": 45})
    perform_control(session, "set_bus", {"on": False})
    session.advance_clock(40.0)
    turn = session.observe("outside.front_object", "animal", 0.9)
    report = next(r for r in turn.rules if r.rule_id == "animal_ahead")
    assert report.reason == "vehicle.all/speed_kph is stale (40.0s > 2.0s)"
    assert session.car.get_signal("vehicle.all", "speed_kph") == 45.0


def test_the_bus_control_is_not_reachable_as_an_action(session):
    """It is the world being described, not the car being commanded, and the tables are what
    keep that from becoming a naming convention."""
    with pytest.raises(KeyError):
        perform(session, "set_bus", {"on": False})


def test_an_action_name_is_not_reachable_as_a_control(session):
    """The tables are looked up separately, so `/control/say` is not a route to `say`."""
    with pytest.raises(KeyError):
        perform_control(session, "say", {"utterance": "好"})


def test_a_control_name_is_not_reachable_as_an_action(session):
    with pytest.raises(KeyError):
        perform(session, "set_signal", {"entity": "vehicle.all",
                                        "attribute": "speed_kph", "value": 45})
