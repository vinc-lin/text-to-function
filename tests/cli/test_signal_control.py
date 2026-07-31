"""Setting a signal the car senses — a simulator control, and what it must refuse.

The refusals carry most of the weight here. `/signal` exists so the animal rule is reachable
by hand; it must not become a way to command the car that answers to none of the rules
commanding it answers to.
"""
import pytest

from cli.session import Session


@pytest.fixture
def session():
    return Session.build(fake=True, llm=False, gate="permissive")


def test_setting_the_speed_reaches_the_car(session):
    assert session.set_signal("vehicle.all", "speed_kph", 45) == 45.0
    assert session.car.get_signal("vehicle.all", "speed_kph") == 45.0


def test_a_string_from_a_prompt_or_a_form_is_taken_as_a_number(session):
    """Both doors hand this a string: the CLI splits a typed line, the browser posts a form
    field. Neither should have to know that the car stores a float."""
    session.set_signal("vehicle.all", "speed_kph", "45")
    assert session.car.get_signal("vehicle.all", "speed_kph") == 45.0


def test_an_actuated_signal_is_refused(session):
    """The point of the whole control. `window_child_lock` is written by open_window and
    guarded by a precondition; set here it would land in the car with the availability check,
    the precondition and the physical limit all skipped."""
    with pytest.raises(ValueError, match="not a sensed signal"):
        session.set_signal("window.all", "window_child_lock", True)


def test_a_refused_signal_leaves_the_car_untouched(session):
    before = session.car.get_signal("window.all", "window_child_lock")
    with pytest.raises(ValueError):
        session.set_signal("window.all", "window_child_lock", True)
    assert session.car.get_signal("window.all", "window_child_lock") == before
    assert session.changed_signals() == []


def test_a_signal_the_car_does_not_hold_at_all_is_refused(session):
    with pytest.raises(ValueError, match="not a sensed signal"):
        session.set_signal("vehicle.all", "altitude_m", 900)


def test_a_value_above_the_limit_is_refused(session):
    with pytest.raises(ValueError, match="240"):
        session.set_signal("vehicle.all", "speed_kph", 300)


def test_a_value_below_the_limit_is_refused(session):
    with pytest.raises(ValueError):
        session.set_signal("vehicle.all", "speed_kph", -1)


def test_an_out_of_range_value_leaves_the_car_untouched(session):
    """Refused before the write, not written and then complained about."""
    session.set_signal("vehicle.all", "speed_kph", 45)
    with pytest.raises(ValueError):
        session.set_signal("vehicle.all", "speed_kph", 300)
    assert session.car.get_signal("vehicle.all", "speed_kph") == 45.0


def test_the_limits_themselves_are_allowed(session):
    session.set_signal("vehicle.all", "speed_kph", 0)
    assert session.car.get_signal("vehicle.all", "speed_kph") == 0.0
    session.set_signal("vehicle.all", "speed_kph", 240)
    assert session.car.get_signal("vehicle.all", "speed_kph") == 240.0


def test_a_value_that_is_not_a_number_says_which_signal(session):
    """`float()`'s own complaint does not name the signal, and both doors print this."""
    with pytest.raises(ValueError, match="speed_kph"):
        session.set_signal("vehicle.all", "speed_kph", "quickly")


def test_the_refusal_of_an_actuated_signal_does_not_depend_on_the_value(session):
    """Membership is checked first: 'that is not a sensed signal' is the true reason, and a
    complaint about `true` not being a number would answer a question nobody asked."""
    with pytest.raises(ValueError, match="not a sensed signal"):
        session.set_signal("window.all", "window_child_lock", "true")


def test_setting_the_speed_does_not_write_the_operation_log(session):
    """Nothing performed an operation. The log is what the executor did, and an entry here
    would put the world's own movement in the car's record of its commands."""
    session.set_signal("vehicle.all", "speed_kph", 45)
    assert session.car.recent_operations(5) == []


# --- what the panes read ------------------------------------------------------------------

def test_sensed_rows_show_every_declared_signal_whatever_its_value(session):
    """Always all of them. A speed resting at 0.0 is the answer to 'why did the animal rule
    say nothing', and /car would never show it because it has not changed."""
    from sim.seed import sensed_signals
    rows = session.sensed_rows()
    assert [(r["entity"], r["attribute"]) for r in rows] == \
        [(e, a) for e, a, *_ in sensed_signals()]


def test_sensed_rows_carry_the_value_the_unit_and_the_limits(session):
    session.set_signal("vehicle.all", "speed_kph", 45)
    row = next(r for r in session.sensed_rows()
               if (r["entity"], r["attribute"]) == ("vehicle.all", "speed_kph"))
    assert row["value"] == 45.0 and row["unit"] == "kph"
    assert row["min"] == 0.0 and row["max"] == 240.0


def test_a_sensed_signal_at_rest_is_not_a_change(session):
    """The two panes answer different questions and must keep answering different ones."""
    assert session.sensed_rows() and session.changed_signals() == []


# --- the /signal command's own argument handling ------------------------------------------

def _run(session, line):
    """Drive the command dispatcher the way a typed line reaches it."""
    from cli.__main__ import _signal
    _signal(session, line.split()[1:])


def test_the_typed_command_reaches_the_car(session, capsys):
    _run(session, "/signal vehicle.all/speed_kph=45")
    assert "45" in capsys.readouterr().out
    assert session.car.get_signal("vehicle.all", "speed_kph") == 45.0


@pytest.mark.parametrize("line", [
    "/signal",
    "/signal vehicle.all/speed_kph",
    "/signal speed_kph=45",
    "/signal =45",
    "/signal vehicle.all/=45",
    "/signal vehicle.all/speed_kph=",
])
def test_a_malformed_argument_prints_usage_and_survives(session, capsys, line):
    """Same reason a mistyped conf does not raise: the session holds the car and the models,
    and losing it to a typo costs a 60-second reload."""
    _run(session, line)
    assert "usage:" in capsys.readouterr().out
    assert session.changed_signals() == []


def test_a_mistyped_number_is_refused_by_name_not_by_traceback(session, capsys):
    _run(session, "/signal vehicle.all/speed_kph=quickly")
    assert "speed_kph" in capsys.readouterr().out
    assert session.car.get_signal("vehicle.all", "speed_kph") == 0.0


def test_the_command_refuses_an_actuated_signal_with_the_true_reason(session, capsys):
    _run(session, "/signal window.all/window_child_lock=true")
    assert "not a sensed signal" in capsys.readouterr().out
    assert session.changed_signals() == []


def test_the_command_refuses_a_value_out_of_range(session, capsys):
    _run(session, "/signal vehicle.all/speed_kph=300")
    assert "240" in capsys.readouterr().out
    assert session.car.get_signal("vehicle.all", "speed_kph") == 0.0


def test_the_command_is_in_the_help(session):
    """Undiscoverable is the same as absent for a control that exists to be typed by hand."""
    from cli.__main__ import HELP
    assert "/signal" in HELP


def test_the_dispatcher_routes_the_command(session):
    """Through `_command`, as a typed line actually arrives — a helper nothing dispatches to
    is a command that does not exist."""
    from cli.__main__ import _command
    assert _command(session, "/signal vehicle.all/speed_kph=45") is True
    assert session.car.get_signal("vehicle.all", "speed_kph") == 45.0
