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

A third group follows: what the door WRITES DOWN, and the queue a producer can reach it
through. Those tests read the tables directly rather than through any accessor, because the
claim being made is about the record itself -- "the database is the record" is a property of
rows, and a helper that read them for us could be wrong in the same direction as the writer.
"""
import time

import pytest

from intake.envelope import Input, Percept, SignalWrite, Utterance
from intake.ingest import Intake, encode_payload, input_from_row
from intake.sources import SOURCES
from intake.store import Store
from scene.engine import RuleReport, SceneOutcome
from sim.vehicle import SqliteVehicle
from t2f.types import Band, ClauseResult, Decision, RouteResult


class SpyPipeline:
    """Returns a real `RouteResult`, because the door now reads one.

    It used to hand back a string, which was enough while dispatch was all that happened here.
    The door records what the router decided, so a double that cannot be asked what it decided
    would be testing this door with its recording switched off.
    """

    def __init__(self, clauses=()):
        self.routed = []
        self.clauses = list(clauses)

    def route(self, text):
        self.routed.append(text)
        return RouteResult(utterance=text, clauses=self.clauses, reply=f"routed:{text}")


class SpyEngine:
    """Duck-typed, and `world` is None so the construction check below stays out of the way of
    the tests that are not about it.

    Returns a real `SceneOutcome` and answers `explain()` for the same reason SpyPipeline
    returns a RouteResult: those two are what the door writes down.
    """
    world = None

    def __init__(self, reports=()):
        self.seen = []
        self.reports = list(reports)

    def observe(self, obs, now, **kw):
        self.seen.append((obs, now))
        return SceneOutcome("notify", f"observed:{obs.key}", "后排有小孩。", None, "rule", "")

    def explain(self):
        return self.reports


def _clause(text, band=Band.HIGH, chosen="open_window"):
    return ClauseResult(clause=text, decision=Decision(band=band, chosen=chosen, candidates=[]))


@pytest.fixture
def car():
    v = SqliteVehicle(":memory:")
    v.init_schema()
    return v


@pytest.fixture
def store(car):
    """A second `Store` over the SAME connection the door builds one over. Reading through a
    different object than the writer used is deliberate -- it is the closest a test gets to
    asking the database rather than asking the code."""
    return Store(car.conn)


def _intake(car, pipeline=None, engine=None, world=None):
    return Intake(pipeline or SpyPipeline(), engine or SpyEngine(), car, world)


def _rows(car, table):
    return [dict(r) for r in car.conn.execute(f"SELECT * FROM {table} ORDER BY id")]


# --- dispatch ------------------------------------------------------------------------------

def test_an_utterance_reaches_the_router(car):
    pipe = SpyPipeline()
    out = _intake(car, pipeline=pipe).ingest(Input("mic", 100.0, Utterance("开车窗")))
    assert pipe.routed == ["开车窗"]
    assert out.reply == "routed:开车窗"


def test_a_percept_reaches_the_engine(car):
    engine = SpyEngine()
    out = _intake(car, engine=engine).ingest(
        Input("cabin_cam", 100.0, Percept("inside.rear_occupant", "child", 0.9, 300.0)))
    assert out.scene == "observed:inside.rear_occupant"
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


# --- what the door writes down ---------------------------------------------------------------

_SAMPLES = [
    ("mic", Utterance("开车窗")),
    ("cabin_cam", Percept("inside.rear_occupant", "child", 0.9, 300.0)),
    ("can0", SignalWrite("vehicle.all", "speed_kph", 45.0)),
]


@pytest.mark.parametrize("source,payload", _SAMPLES, ids=lambda p: getattr(p, "__name__", p))
def test_every_input_leaves_a_raw_row(car, source, payload):
    """Whatever the payload, and before anything is done with it. "What arrived" has to be a
    fact independent of whether anything could be done about it, or the only inputs on record
    are the ones that worked -- precisely the set nobody needs to investigate."""
    _intake(car).ingest(Input(source, 100.0, payload))
    rows = _rows(car, "observation_raw")
    assert len(rows) == 1
    assert rows[0]["source"] == source and rows[0]["at"] == 100.0
    assert rows[0]["payload"] == encode_payload(payload)


@pytest.mark.parametrize("source,payload", _SAMPLES, ids=lambda p: getattr(p, "__name__", p))
def test_a_payload_round_trips_through_the_column(car, store, source, payload):
    """The convention, stated as a test: JSON of the payload's fields, with the TYPE recovered
    from the source's declaration rather than written beside the data. Every producer in Task 7
    writes rows against this, so a change to either half has to fail here."""
    sent = Input(source, 100.0, payload)
    store.put_raw(sent.source, sent.at, encode_payload(sent.payload))
    assert input_from_row(store.pending()[0]) == sent


def test_an_input_that_came_through_the_door_is_not_left_in_the_queue(car):
    """The bug that would matter most and be hardest to see. A raw row still pending after a
    synchronous ingest is one `process_pending` would run a second time -- and for a command
    that reaches the car, twice is not a cosmetic difference."""
    door = _intake(car)
    door.ingest(Input("mic", 100.0, Utterance("开车窗")))
    assert door.store.pending() == []
    assert _rows(car, "observation_raw")[0]["processed_at"] == 100.0


def test_a_voice_input_writes_a_turn_and_one_decision_per_clause(car):
    door = _intake(car, pipeline=SpyPipeline(clauses=[
        _clause("开车窗", Band.HIGH, "open_window"),
        _clause("空调调到22度", Band.MEDIUM, "set_temperature")]))
    door.ingest(Input("mic", 100.0, Utterance("开车窗，空调调到22度")))

    turns = _rows(car, "turn")
    assert len(turns) == 1
    assert turns[0]["kind"] == "route" and turns[0]["reply"] == "routed:开车窗，空调调到22度"
    assert turns[0]["raw_id"] == _rows(car, "observation_raw")[0]["id"]

    decisions = _rows(car, "decision")
    assert [(d["subject"], d["verdict"], d["chosen"]) for d in decisions] == [
        ("开车窗", "high", "open_window"),
        ("空调调到22度", "medium", "set_temperature")]
    assert all(d["turn_id"] == turns[0]["id"] for d in decisions)


def test_a_voice_input_writes_the_words_as_well(car):
    """`utterance` is separate from the raw payload because retention clears one and keeps the
    other -- and because the parsed layer must survive the sweep for a run to stay replayable."""
    _intake(car).ingest(Input("mic", 100.0, Utterance("开车窗")))
    rows = _rows(car, "utterance")
    assert len(rows) == 1 and rows[0]["text"] == "开车窗"
    assert rows[0]["raw_id"] == _rows(car, "observation_raw")[0]["id"]


def test_a_scene_observation_writes_a_turn_and_one_decision_per_rule(car):
    engine = SpyEngine(reports=[
        RuleReport("observed:inside.rear_occupant", "match", "rule matched", ""),
        RuleReport("animal_ahead", "reject", "no animal", ""),
        RuleReport("child_alone", "match", "rule matched", "outranked by rear_occupant")])
    door = _intake(car, engine=engine)
    door.ingest(Input("cabin_cam", 100.0, Percept("inside.rear_occupant", "child", 0.9, 300.0)))

    turns = _rows(car, "turn")
    assert len(turns) == 1
    assert turns[0]["kind"] == "scene" and turns[0]["reply"] == "后排有小孩。"

    decisions = _rows(car, "decision")
    assert [(d["subject"], d["verdict"], d["chosen"], d["suppressed_by"]) for d in decisions] == [
        # `chosen` names the scene only on the rule that produced the outcome. On every row it
        # would say all three chose it; on none of them the table could not answer what was
        # acted on.
        ("observed:inside.rear_occupant", "match", "observed:inside.rear_occupant", ""),
        ("animal_ahead", "reject", None, ""),
        ("child_alone", "match", None, "outranked by rear_occupant")]


def test_a_signal_write_records_the_frame_and_opens_no_turn(car):
    """A signal write is the world moving: no decision to record and nothing said. A turn per
    frame would also mean a turn at 10 Hz in which nothing happened."""
    _intake(car).ingest(Input("can0", 100.0, SignalWrite("vehicle.all", "speed_kph", 45.0)))
    assert len(_rows(car, "observation_raw")) == 1
    assert _rows(car, "turn") == [] and _rows(car, "decision") == []


def test_an_input_the_door_cannot_place_is_still_written_down(car):
    """The TypeError stays loud -- and the arrival stops being invisible. A wiring mistake that
    raises into a caller who swallows it used to leave no trace at all that anything came in."""
    class Rumour:
        pass

    bad = Input("mic", 100.0, Utterance("x"))
    object.__setattr__(bad, "payload", Rumour())
    with pytest.raises(TypeError, match="Rumour"):
        _intake(car).ingest(bad)
    row = _rows(car, "observation_raw")[0]
    assert row["processed_at"] == 100.0 and "Rumour" in row["error"]


# --- the queue --------------------------------------------------------------------------------

def test_process_pending_on_an_empty_queue_is_a_no_op(car):
    door = _intake(car)
    assert door.process_pending(now=100.0) == []
    assert _rows(car, "observation_raw") == [] and _rows(car, "turn") == []


def test_a_row_written_by_a_producer_is_dispatched(car, store):
    """The whole point of inputs being an interface: no binding, no import, just a row."""
    store.put_raw("can0", 100.0, encode_payload(SignalWrite("vehicle.all", "speed_kph", 45.0)))
    door = _intake(car)
    done = door.process_pending(now=101.0)
    assert [(p.raw_id, p.outcome, p.error) for p in done] == [(1, 45.0, "")]
    assert car.get_signal("vehicle.all", "speed_kph") == 45.0
    assert store.pending() == []


def test_a_queued_row_is_processed_at_the_caller_s_clock_not_its_own(car, store):
    """`at` is when the world produced it; `processed_at` is when we got to it. The gap between
    them is the only measure of a queue falling behind, so they must not be the same stamp."""
    store.put_raw("can0", 100.0, encode_payload(SignalWrite("vehicle.all", "speed_kph", 45.0)))
    _intake(car).process_pending(now=175.0)
    row = _rows(car, "observation_raw")[0]
    assert row["at"] == 100.0 and row["processed_at"] == 175.0


def test_the_queue_runs_in_at_then_id_order(car, store):
    """A producer on its own clock inserts out of order, and a replay has to be the world's
    order, not the arrival order. `id` breaks ties so the order is total."""
    payload = encode_payload(SignalWrite("vehicle.all", "speed_kph", 1.0))
    late = store.put_raw("can0", 200.0, payload)
    early = store.put_raw("can0", 100.0, payload)
    tie_a = store.put_raw("can0", 150.0, payload)
    tie_b = store.put_raw("can0", 150.0, payload)
    done = _intake(car).process_pending(now=300.0)
    assert [p.raw_id for p in done] == [early, tie_a, tie_b, late]


def test_a_raising_row_is_marked_processed_with_its_error_and_the_next_one_still_runs(car,
                                                                                     store):
    """It must neither block the queue forever nor vanish. A dropped input is silence, and
    unexplained silence is the failure this system spends the most effort preventing."""
    class Exploding:
        def route(self, text):
            raise RuntimeError("boom")

    store.put_raw("mic", 100.0, encode_payload(Utterance("开车窗")))
    store.put_raw("can0", 200.0, encode_payload(SignalWrite("vehicle.all", "speed_kph", 45.0)))
    done = _intake(car, pipeline=Exploding()).process_pending(now=300.0)

    assert done[0].error == "RuntimeError: boom" and done[0].outcome is None
    assert done[1].outcome == 45.0 and done[1].error == ""
    assert store.pending() == [], "a failed row that stays pending blocks the queue forever"
    assert _rows(car, "observation_raw")[0]["error"] == "RuntimeError: boom"
    assert car.get_signal("vehicle.all", "speed_kph") == 45.0


def test_a_turn_that_raised_still_left_its_row(car, store):
    """Opened before the work and closed after it. Recording only the turns that finished
    records precisely the set nobody needs to investigate."""
    class Exploding:
        def route(self, text):
            raise RuntimeError("boom")

    store.put_raw("mic", 100.0, encode_payload(Utterance("开车窗")))
    _intake(car, pipeline=Exploding()).process_pending(now=300.0)
    turns = _rows(car, "turn")
    assert len(turns) == 1 and turns[0]["kind"] == "route" and turns[0]["reply"] == ""


def test_a_row_that_cannot_become_an_input_is_an_error_not_a_crash(car, store):
    """The decode IS the validation that the row was well formed, and a producer's bug belongs
    on the producer's row -- not raised at whoever happened to call the pump."""
    store.put_raw("mic", 100.0, '{"not_a_field": 1}')
    store.put_raw("cabin_cam", 200.0, "not json at all")
    done = _intake(car).process_pending(now=300.0)
    assert all(p.error and p.outcome is None for p in done)
    assert store.pending() == []
    assert all(r["error"] for r in _rows(car, "observation_raw"))


def test_a_queued_row_travels_the_same_path_as_a_direct_call(car, store):
    """The property Task 7's generator depends on. If the queue had its own dispatch, the
    simulator would be exercising a path nothing in production uses."""
    percept = Percept("inside.rear_occupant", "child", 0.9, 300.0)
    _intake(car).ingest(Input("cabin_cam", 100.0, percept))
    direct = (_rows(car, "turn"), _rows(car, "decision"), _rows(car, "perception"))

    car.conn.executescript("DELETE FROM decision; DELETE FROM turn; "
                           "DELETE FROM perception; DELETE FROM observation_raw;")
    store.put_raw("cabin_cam", 100.0, encode_payload(percept))
    _intake(car).process_pending(now=100.0)
    queued = (_rows(car, "turn"), _rows(car, "decision"), _rows(car, "perception"))

    def _shape(tables):
        # `id` and `raw_id` are the two columns that legitimately differ: AUTOINCREMENT does
        # not rewind over the DELETE above, so the second run's rows are numbered from where
        # the first stopped. Everything else has to match exactly.
        skip = ("id", "raw_id")
        return [[{k: v for k, v in row.items() if k not in skip} for row in t] for t in tables]

    assert _shape(direct) == _shape(queued)


# --- what the car did, tied to why -------------------------------------------------------------

def test_an_operation_logged_during_a_turn_is_attributed_to_it(car):
    """`operation_log.turn_id` is the join between what we decided and what the car did.

    Filled from a watermark taken when the turn opens rather than from an ambient turn id the
    executor carries: a caller that forgets to clear an ambient id does not leave a NULL, it
    leaves the PREVIOUS turn's id on rows belonging to this one, and a wrong answer in the
    table you consult to find out what happened is worse than no answer.
    """
    class Acting:
        def __init__(self, car):
            self.car = car

        def route(self, text):
            self.car.log("open_window", {"position": "driver"}, "executed", None, "")
            return RouteResult(utterance=text, clauses=[_clause(text)], reply="好的。")

    door = _intake(car, pipeline=Acting(car))
    door.ingest(Input("mic", 100.0, Utterance("开车窗")))
    ops = _rows(car, "operation_log")
    assert len(ops) == 1 and ops[0]["turn_id"] == _rows(car, "turn")[0]["id"]


def test_an_operation_from_before_the_turn_is_not_claimed_by_it(car):
    """The watermark's whole job. A `--db` file written before turns existed is full of untagged
    rows, and a bare `WHERE turn_id IS NULL` would let the next session's first turn adopt every
    one of them -- a fact in the record that never happened."""
    car.log("lock_doors", {}, "executed", None, "")
    _intake(car).ingest(Input("mic", 100.0, Utterance("开车窗")))
    assert _rows(car, "operation_log")[0]["turn_id"] is None
