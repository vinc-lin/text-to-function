"""The bus, and what stopping it does.

A dead bus and a stationary car used to be indistinguishable to every rule: `updated_at` was
written on every signal write and read by nothing, so `SignalAbove("vehicle.all", "speed_kph",
above=5.0)` fired the animal warning off a speed frozen ten minutes ago exactly as readily as
off a live one. These tests are that distinction, driven from the door a person actually types
at.

The shape of every one of them is the same and it is the point: nothing here makes a signal go
stale by waiting. `/clock` cannot do it while the bus is running, because the pump stamps on the
session's own clock and the stamps move with it. Only stopping the bus does — which is what a
stale reading means in a car.
"""
import pytest

from cli.session import Session

WARNING = "前方有动物，请注意。"


@pytest.fixture
def session():
    return Session.build(fake=True, llm=False, gate="permissive")


def _animal(session):
    """One perception event the animal rule is about, through the session's own door."""
    return session.observe("outside.front_object", "animal", 0.9)


def _report(turn, rule_id="animal_ahead"):
    return next(r for r in turn.rules if r.rule_id == rule_id)


# --- the story ------------------------------------------------------------------------------

def test_a_running_bus_keeps_the_warning_alive(session):
    session.set_signal("vehicle.all", "speed_kph", 45.0)
    turn = _animal(session)
    assert turn.reply == WARNING
    assert _report(turn).verdict == "match"


def test_a_stopped_bus_makes_the_rule_reject_and_names_both_ages(session):
    """The rejection must say `stale`, not `not above 5.0`. A bare threshold complaint about a
    frozen value reads as "the car is slow" when the truth is "the bus is quiet", and the two
    call for opposite responses from whoever is reading."""
    session.set_signal("vehicle.all", "speed_kph", 45.0)
    _animal(session)
    session.set_bus(False)
    session.advance_clock(5.0)

    turn = _animal(session)
    report = _report(turn)
    assert report.verdict == "reject"
    assert report.reason == "vehicle.all/speed_kph is stale (5.0s > 2.0s)"
    assert turn.reply == "", "a stale signal must silence the rule, not soften it"


def test_starting_the_bus_again_restores_the_warning(session):
    """The value never changed; only whether anything is still saying it."""
    session.set_signal("vehicle.all", "speed_kph", 45.0)
    assert _animal(session).reply == WARNING
    session.set_bus(False)
    # Past the max age AND past the rule's own 30s cooldown, so the silence that follows is the
    # bus and nothing else, and the warning that returns is a real re-fire.
    session.advance_clock(40.0)
    assert "stale" in _report(_animal(session)).reason

    session.set_bus(True)
    turn = _animal(session)
    assert turn.reply == WARNING
    assert _report(turn).verdict == "match"


def test_a_clock_advance_alone_cannot_manufacture_staleness(session):
    """Ten minutes forward with the bus running, and the speed is still live. A real bus does
    not go quiet because time passed."""
    session.set_signal("vehicle.all", "speed_kph", 45.0)
    session.advance_clock(600.0)
    turn = _animal(session)
    assert _report(turn).verdict == "match"
    assert turn.reply == WARNING


def test_a_session_that_has_touched_nothing_still_has_a_live_bus(session):
    """The seeded resting values are on the bus from the first moment, exactly as a real one
    publishes from power-on. Without that, nothing is held until someone types /signal, and two
    seconds into any session the animal rule would report `speed_kph is stale` with the bus
    running — the instrument blaming the bus for a car that is simply parked."""
    session.advance_clock(60.0)
    report = _report(_animal(session))
    assert report.verdict == "reject"
    assert report.reason == "vehicle.all/speed_kph is 0.0, not above 5.0"


# --- what the bus is, and is not ------------------------------------------------------------

def test_the_bus_is_running_by_default(session):
    assert session.bus_publishing() is True


def test_stopping_the_bus_does_not_change_the_value(session):
    """`/bus off` is not `/signal 0`. The car is still doing 45; nothing is saying so."""
    session.set_signal("vehicle.all", "speed_kph", 45.0)
    session.set_bus(False)
    session.advance_clock(5.0)
    assert session.car.get_signal("vehicle.all", "speed_kph") == 45.0


def test_a_reset_takes_the_old_car_s_reading_off_the_bus(session):
    """A held 45 kph belongs to a vehicle that no longer exists. Republished into the freshly
    seeded one it would arrive looking perfectly live — a car reading as doing 45 while /car
    reports it as it was seeded."""
    session.set_signal("vehicle.all", "speed_kph", 45.0)
    session.reset()
    session.pump()
    assert session.car.get_signal("vehicle.all", "speed_kph") == 0.0
    assert session.changed_signals() == []


def test_the_new_car_is_on_the_bus_after_a_reset(session):
    session.reset()
    session.advance_clock(60.0)
    assert "stale" not in _report(_animal(session)).reason


def test_the_bus_state_survives_a_reset(session):
    """An instrument setting, in the same category as /llm and /gate — not a fact about the
    car, which is the only thing /reset replaces."""
    session.set_bus(False)
    session.reset()
    assert session.bus_publishing() is False


def test_a_mode_switch_reaches_the_door(session):
    """/llm off detaches the model from the session; a door still holding the old pipeline
    would keep routing every utterance through the one that still had it."""
    session.rebuild(gate="shipped")
    assert session.intake.pipeline is session.pipeline


def test_a_mode_switch_leaves_the_bus_running(session):
    """The point of the switch is typing the same words twice against the same car, and a bus
    emptied by it would make the second attempt behave differently for no stated reason."""
    session.set_signal("vehicle.all", "speed_kph", 45.0)
    session.rebuild(gate="shipped")
    session.advance_clock(600.0)
    assert _report(_animal(session)).verdict == "match"


# --- the typed commands ---------------------------------------------------------------------

def _run(session, line):
    from cli.__main__ import _command
    return _command(session, line)


def test_the_command_is_in_the_help():
    """Undiscoverable is the same as absent for a control that exists to be typed by hand."""
    from cli.__main__ import HELP
    assert "/bus" in HELP


def test_the_dispatcher_routes_the_command(session):
    assert _run(session, "/bus off") is True
    assert session.bus_publishing() is False
    assert _run(session, "/bus on") is True
    assert session.bus_publishing() is True


def test_a_malformed_argument_prints_usage_and_changes_nothing(session, capsys):
    """Same reason a mistyped conf does not raise: the session holds the car and the models,
    and losing it to a typo costs a 60-second reload."""
    _run(session, "/bus")
    assert "usage:" in capsys.readouterr().out
    assert session.bus_publishing() is True
    _run(session, "/bus maybe")
    assert "usage:" in capsys.readouterr().out
    assert session.bus_publishing() is True


def test_the_signal_line_says_whether_the_bus_is_live(session, capsys):
    """"45 kph" and "45 kph as of some moment in the past" are different facts, and only one of
    them is what a rule will read."""
    _run(session, "/signal vehicle.all/speed_kph=45")
    out = capsys.readouterr().out
    assert "45" in out and "kph" in out and "publishing" in out

    _run(session, "/bus off")
    capsys.readouterr()
    _run(session, "/signal vehicle.all/speed_kph=45")
    assert "bus off" in capsys.readouterr().out


def test_every_command_pumps(session):
    """A command that only looks still pumps, so a signal's age never depends on how you asked
    to see it. `/car` reads nothing about time and still leaves the bus fresh behind it."""
    session.set_signal("vehicle.all", "speed_kph", 45.0)
    session.advance_clock(600.0)
    _run(session, "/car")
    assert session.car.signal_age("vehicle.all", "speed_kph", session._now()) < 2.0


def test_the_typed_story_end_to_end(session, capsys):
    """The whole discipline, in the order a person types it."""
    _run(session, "/signal vehicle.all/speed_kph=45")
    _run(session, "/scene outside.front_object=animal conf=0.9")
    assert WARNING in capsys.readouterr().out

    _run(session, "/bus off")
    _run(session, "/clock +40")
    _run(session, "/scene outside.front_object=animal conf=0.9")
    out = capsys.readouterr().out
    assert "is stale (40.0s > 2.0s)" in out
    assert WARNING not in out

    _run(session, "/bus on")
    _run(session, "/scene outside.front_object=animal conf=0.9")
    assert WARNING in capsys.readouterr().out
