"""One door in, and a bus that keeps publishing.

Two properties, and they pull in opposite directions. Dispatch must reach the module that owns
the decision and nothing else -- no rule, no threshold, no phrasing lives here -- while the
publisher has to be real enough that a stopped bus actually goes stale. The tests below are
split accordingly: the first group proves each payload type lands where it should, the second
proves a held value stays fresh while it is published and ages the moment it is not.

The clock test at the end is the one that has already bitten. `SqliteVehicle.set_signal` stamps
with `time.time()` and a session reads at `time.time() + offset`, so a pump that stamps on the
wall clock publishes values that are stale on arrival the moment anyone touches `/clock`. It
fails OPEN in the sense that nothing raises -- the bus looks wired and every rule falls silent.
"""
import time

import pytest

from intake.envelope import Input, Percept, SignalWrite, Utterance
from intake.ingest import Intake
from intake.sources import SOURCES
from sim.vehicle import SqliteVehicle


class SpyPipeline:
    def __init__(self):
        self.routed = []

    def route(self, text):
        self.routed.append(text)
        return f"routed:{text}"


class SpyEngine:
    """Duck-typed, and `world` is None so the construction check below stays out of the way of
    the tests that are not about it."""
    world = None

    def __init__(self):
        self.seen = []

    def observe(self, obs, now, **kw):
        self.seen.append((obs, now))
        return f"observed:{obs.key}"


@pytest.fixture
def car():
    v = SqliteVehicle(":memory:")
    v.init_schema()
    return v


def _intake(car, pipeline=None, engine=None, world=None):
    return Intake(pipeline or SpyPipeline(), engine or SpyEngine(), car, world)


# --- dispatch ------------------------------------------------------------------------------

def test_an_utterance_reaches_the_router(car):
    pipe = SpyPipeline()
    out = _intake(car, pipeline=pipe).ingest(Input("mic", 100.0, Utterance("开车窗")))
    assert pipe.routed == ["开车窗"]
    assert out == "routed:开车窗"


def test_a_percept_reaches_the_engine(car):
    engine = SpyEngine()
    out = _intake(car, engine=engine).ingest(
        Input("cabin_cam", 100.0, Percept("inside.rear_occupant", "child", 0.9, 300.0)))
    assert out == "observed:inside.rear_occupant"
    obs, now = engine.seen[0]
    assert (obs.key, obs.value, obs.confidence, obs.ttl) == \
        ("inside.rear_occupant", "child", 0.9, 300.0)
    assert now == 100.0


def test_a_percept_carries_the_envelope_s_provenance_into_perception(car):
    """The whole reason `source` stopped being decoration. `Observation.source` is filled from
    the Input, not from a default the caller forgot to override, so a belief in the store can
    always be traced to something that was allowed to produce it."""
    engine = SpyEngine()
    _intake(car, engine=engine).ingest(
        Input("front_cam", 100.0, Percept("outside.front_object", "animal", 0.9, 300.0)))
    obs, _ = engine.seen[0]
    assert obs.source == "front_cam"
    assert obs.at == 100.0


def test_a_signal_write_reaches_the_car(car):
    out = _intake(car).ingest(Input("can0", 100.0, SignalWrite("vehicle.all", "speed_kph", 45.0)))
    assert car.get_signal("vehicle.all", "speed_kph") == 45.0
    assert out == 45.0


def test_a_signal_write_stamps_at_the_input_s_own_time(car):
    """The envelope already carries WHEN, so the car must not re-invent it. A write stamped at
    wall time and read on an offset clock is the same defect as a pump on the wrong clock, one
    step earlier in the pipe."""
    at = time.time() + 500.0
    _intake(car).ingest(Input("can0", at, SignalWrite("vehicle.all", "speed_kph", 45.0)))
    assert car.signal_age("vehicle.all", "speed_kph", at) == pytest.approx(0.0, abs=0.5)


def test_every_declared_source_round_trips(car):
    """Each source's declared payload type reaches a handler. A source declared in the registry
    with nothing behind it would accept an Input and then quietly do nothing."""
    samples = {
        Utterance: Utterance("开车窗"),
        Percept: Percept("inside.rear_occupant", "child", 0.9, 300.0),
        SignalWrite: SignalWrite("vehicle.all", "speed_kph", 45.0),
    }
    door = _intake(car)
    for name, src in SOURCES.items():
        assert src.accepts in samples, f"{name} accepts a payload nothing here can build"
        assert door.ingest(Input(name, 100.0, samples[src.accepts])) is not None


def test_an_undeclared_source_never_reaches_ingest(car):
    """Not refused BY ingest -- refused before an Input naming it can exist at all. Validating
    at construction means the door has no undeclared case to handle and cannot grow one."""
    with pytest.raises(ValueError, match="nowhere"):
        Input("nowhere", 100.0, Utterance("hi"))


def test_a_payload_type_with_no_handler_is_refused_loudly(car):
    """Silence is the safe default everywhere in this system EXCEPT here. A payload the door
    cannot place is a wiring mistake, and swallowing it would drop an input with nothing said."""
    class Rumour:
        pass

    door = _intake(car)
    # Forced past the envelope's own check, because that check is what makes this unreachable
    # by any legitimate route. The branch still has to exist: a fourth payload type added to
    # envelope.py and forgotten here would otherwise vanish silently.
    bad = Input("mic", 100.0, Utterance("x"))
    object.__setattr__(bad, "payload", Rumour())
    with pytest.raises(TypeError, match="Rumour"):
        door.ingest(bad)


def test_the_door_and_the_engine_must_read_one_world(car):
    """The same silent failure `SceneEngine.reads` exists to catch, one level up: a door writing
    into stores nothing reads leaves every rule looking at an empty world, with nothing raising
    and the system merely quiet."""
    engine = SpyEngine()
    engine.world = object()
    with pytest.raises(ValueError, match="same world"):
        Intake(SpyPipeline(), engine, car, object())


# --- the publisher -------------------------------------------------------------------------

def _publish(door, at, value=45.0):
    door.ingest(Input("can0", at, SignalWrite("vehicle.all", "speed_kph", value)))


def test_a_held_value_stays_fresh_across_pumps(car):
    """What a live bus IS. The value never changes; only its stamp does, and that is the whole
    difference between a car doing 45 and a car that said 45 once."""
    door = _intake(car)
    at = time.time()
    _publish(door, at)
    for elapsed in (1.0, 5.0, 60.0, 600.0):
        door.pump(at + elapsed)
        assert car.signal_age("vehicle.all", "speed_kph", at + elapsed) == pytest.approx(0.0)


def test_a_pump_re_stamps_without_changing_the_value(car):
    door = _intake(car)
    at = time.time()
    _publish(door, at, 45.0)
    door.pump(at + 10.0)
    assert car.get_signal("vehicle.all", "speed_kph") == 45.0


def test_a_stopped_bus_lets_its_value_age(car):
    """The only way age accumulates. Stopping the publisher is what makes a signal stale, not
    the passage of time -- which is true of a real bus too."""
    door = _intake(car)
    at = time.time()
    _publish(door, at)
    door.set_publishing("can0", False)
    door.pump(at + 10.0)
    assert car.signal_age("vehicle.all", "speed_kph", at + 10.0) == pytest.approx(10.0)


def test_starting_the_bus_again_makes_the_value_young_again(car):
    door = _intake(car)
    at = time.time()
    _publish(door, at)
    door.set_publishing("can0", False)
    door.pump(at + 10.0)
    door.set_publishing("can0", True)
    door.pump(at + 11.0)
    assert car.signal_age("vehicle.all", "speed_kph", at + 11.0) == pytest.approx(0.0)


def test_a_value_written_while_the_bus_was_off_is_published_when_it_starts(car):
    """Held on the DECLARATION, published on the current state. Otherwise `/bus on` would
    republish nothing until someone happened to set a signal again, and the toggle would look
    broken in exactly the case it exists for."""
    door = _intake(car)
    at = time.time()
    door.set_publishing("can0", False)
    _publish(door, at, 45.0)
    door.set_publishing("can0", True)
    door.pump(at + 10.0)
    assert car.signal_age("vehicle.all", "speed_kph", at + 10.0) == pytest.approx(0.0)


def test_a_pump_with_nothing_held_writes_nothing(car):
    assert _intake(car).pump(time.time()) == 0


def test_a_non_publishing_source_puts_nothing_on_the_bus(car):
    """`pump` over a source that does not publish is a no-op, because nothing it produces is a
    level to re-stamp: an utterance is an event and a percept expires on its own terms."""
    door = _intake(car)
    door.ingest(Input("mic", 100.0, Utterance("开车窗")))
    door.ingest(Input("cabin_cam", 100.0, Percept("inside.rear_occupant", "child", 0.9, 300.0)))
    assert door.pump(200.0) == 0


def test_a_source_that_cannot_publish_cannot_be_switched_on(car):
    """Refused rather than accepted and ignored. `set_publishing("mic", True)` that returned
    quietly would report a bus running for a source with nothing to run."""
    door = _intake(car)
    with pytest.raises(ValueError, match="mic"):
        door.set_publishing("mic", True)
    with pytest.raises(ValueError, match="nowhere"):
        door.set_publishing("nowhere", True)


def test_publishing_starts_from_the_declaration(car):
    door = _intake(car)
    assert door.publishing("can0") is True
    assert door.publishing("mic") is False


def test_forgetting_the_car_forgets_what_was_held(car):
    """A reset replaces the vehicle. Republishing a speed measured on the car that no longer
    exists is the same mistake as answering a question that was asked about it -- and worse
    here, because the value would land in a freshly seeded car looking live."""
    door = _intake(car)
    at = time.time()
    _publish(door, at, 45.0)
    door.forget()
    car.set_signal("vehicle.all", "speed_kph", 0.0, at=at)
    door.pump(at + 10.0)
    assert car.get_signal("vehicle.all", "speed_kph") == 0.0
    assert door.publishing("can0") is True, "the bus is an instrument setting, not car state"


# --- the clock ------------------------------------------------------------------------------

def test_the_pump_stamps_at_the_clock_it_was_given(car):
    """The trap this design already fell into once.

    `set_signal` stamps `time.time()`; a session reads at `time.time() + offset`. A pump that
    stamps on the wall clock therefore publishes a value that is `offset` seconds old the
    instant it is written, so one `/clock +5` makes every pumped signal stale immediately --
    for a reason that has nothing to do with the bus, with nothing raising and every rule
    falling silent.

    Written so it FAILS if the pump ever stamps `time.time()` itself: the offset here is far
    larger than any max_age, so a wall-clock stamp reads as 500 seconds old rather than zero.
    """
    door = _intake(car)
    offset_now = time.time() + 500.0                  # a session that has typed /clock +500
    _publish(door, offset_now)
    door.pump(offset_now + 1.0)
    assert car.signal_age("vehicle.all", "speed_kph", offset_now + 1.0) == \
        pytest.approx(0.0, abs=0.5)


def test_the_pump_does_not_read_a_clock_of_its_own(car):
    """A pump on the wrong clock is invisible in a session with no offset, which is why the
    test above uses one. This states the same property from the other side: the stamp is a
    function of the argument alone, so two pumps at the same `now` produce the same age."""
    door = _intake(car)
    at = time.time() + 500.0
    _publish(door, at)
    door.pump(at + 3.0)
    first = car.signal_age("vehicle.all", "speed_kph", at + 3.0)
    door.pump(at + 3.0)
    assert car.signal_age("vehicle.all", "speed_kph", at + 3.0) == pytest.approx(first)
    assert first == pytest.approx(0.0, abs=0.5)
