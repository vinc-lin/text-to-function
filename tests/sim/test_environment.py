"""The generator, checked against the one thing that is not the generator.

A simulator can only ever be tested for two things: that it is reproducible, and that it agrees
with something a person wrote by hand. Everything else it could be asked would be asking it
whether it matches itself. So the tests here are the determinism of the seed, the validity of
the rows, and — the one worth the most — `test_a_generated_scenario_reaches_the_gold_outcome`,
which drives each generated scenario through `process_pending` and requires the outcome its
hand-authored row in `data/eval/scenes.jsonl` states. Without that last one this module is a
machine for producing confident numbers about our own assumptions.

The outcome is taken from what `process_pending` returns, and the last non-empty reply wins —
the same rule `eval/run_scene_eval.py::predict` applies for the same reason: a suppressed repeat
must not blank a warning that was already spoken.
"""
from __future__ import annotations
import time

import pytest

from cli.session import Session
from eval.dataset import load_dataset
from intake.envelope import Percept, SignalWrite, Utterance
from intake.ingest import input_from_row
from intake.sources import SOURCES
from sim.environment import (GOLD_SCENARIOS, SCENARIOS, SPEED, Motion, Scenario, Sighting,
                             Speech, frames, write)
from sim.seed import sensed_signals

GOLD = {row["id"]: row for row in load_dataset("data/eval/scenes.jsonl")}

# Every gold scenario whose outcome `process_pending` can reproduce, which is every one that
# does not turn on consent. See `test_the_queue_routes_a_consent_answer_instead_of_resolving_it`
# for the six it excludes and why they are excluded rather than fixed here.
QUEUE_REACHABLE = [s for s in GOLD_SCENARIOS if not s.says]


def _drain(scenario: Scenario, *, seed: int = 7):
    """Generate the scenario, run the queue, return (session, what was spoken).

    A real epoch for `start`, not 0.0, and the reason is the one CLAUDE.md keeps: the car stamps
    `time.time()` when `Session.build` seeds it, so a scenario clock starting at zero would give
    every seeded signal an age of minus 1.8 billion seconds — which is under every max_age, so
    staleness would silently never fire and `animal_stale_bus` would pass for the wrong reason.
    A producer stamps the clock it has; so does this.
    """
    session = Session.build(fake=True, llm=False, gate="permissive")
    start = time.time()
    write(session.intake.store, frames(scenario, seed=seed, start=start))
    spoken = ""
    for done in session.intake.process_pending(now=start + scenario.span + 5.0):
        speech = getattr(done.outcome, "speech", "") or ""
        if speech:
            spoken = speech
    return session, spoken


# --- against something a human agreed to ------------------------------------------------------

def test_every_gold_row_has_a_scenario():
    """The correspondence, asserted rather than declared.

    A row added to `data/eval/scenes.jsonl` without a scenario here is not a test that fails —
    it is coverage that quietly is not there, which is the shape of every gap this store's own
    completeness test exists to prevent one level down.
    """
    assert {s.id for s in GOLD_SCENARIOS} == set(GOLD)


@pytest.mark.parametrize("scenario", QUEUE_REACHABLE, ids=lambda s: s.id)
def test_a_generated_scenario_reaches_the_gold_outcome(scenario):
    """**The test this task is worth doing for.**

    Generated rows go in the raw layer; `process_pending` parses, dispatches and evaluates them;
    what comes out has to be the sentence a person wrote down as correct for that situation. It
    is the only thing tying this generator to anything outside itself — and it ties the WHOLE
    path, because the scenario states a world (a car at 45 kph, a camera 90% sure of an animal)
    and the gold states a reply, with every rule, band and liveness question in between.
    """
    _session, spoken = _drain(scenario)
    assert spoken == GOLD[scenario.id].get("expect", "")


def test_the_queue_routes_a_consent_answer_instead_of_resolving_it():
    """The finding, stated as a test so it cannot be forgotten or quietly fixed the wrong way.

    `SceneEngine.resolve` is called from `cli/session.py`, ABOVE the door — 好 is only an answer
    because a session is holding a question, and nothing in an envelope can know that. So an
    utterance that arrives as a ROW is routed against the catalog like any other sentence, and
    the six `consent_*` gold rows cannot be reproduced through `process_pending`.

    That is a real limit of the queue as an integration surface, not a defect in the generator:
    a producer writing rows cannot answer a question it was never asked. Moving consent into
    intake would close it and is rejected in the design's §11 — it is how intake becomes the
    module every other one imports. What this asserts is that the divergence is exactly where
    it is claimed to be: the words reach the router, so they are recorded and answered, just not
    as consent.
    """
    session, spoken = _drain(SCENARIOS["consent_yes"])
    assert GOLD["consent_yes"]["expect"] == "已为您打开车窗儿童锁。"
    assert spoken == "后排有小孩，要打开儿童锁吗？", \
        "the question was asked and never answered"

    turns = [dict(r) for r in session.car.conn.execute("SELECT * FROM turn ORDER BY id")]
    assert [t["kind"] for t in turns] == ["scene", "route"], \
        "好 arriving as a row is routed, because only a session knows a question is pending"
    # Not silently dropped, which would be the bad version of this limit: the words are on the
    # record with what the router made of them, so the gap is visible in the store itself.
    assert turns[-1]["reply"]


def test_a_quiet_bus_is_the_only_reason_the_stale_scenario_says_nothing():
    """Verified by mutation, because a silent scenario passes for any number of reasons.

    `animal_stale_bus` expects silence, and so does a scenario with the wrong key, a broken
    generator, or a rule that never fires at all. Republishing the same speed for the whole span
    changes exactly one thing — whether the bus went quiet — and the warning has to come back.
    Without this, the row asserts that something did not happen.
    """
    quiet = SCENARIOS["animal_stale_bus"]
    _s, silent = _drain(quiet)
    assert silent == ""

    talking = Scenario(quiet.id, span=quiet.span, motion=Motion(45.0), sees=quiet.sees)
    _s, spoken = _drain(talking)
    assert spoken == "前方有动物，请注意。"


# --- reproducibility --------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", GOLD_SCENARIOS, ids=lambda s: s.id)
def test_the_same_seed_produces_identical_rows(scenario):
    """A failure at volume is worth nothing if it cannot be run again."""
    assert frames(scenario, seed=99, start=1000.0) == frames(scenario, seed=99, start=1000.0)


@pytest.mark.parametrize("scenario", GOLD_SCENARIOS, ids=lambda s: s.id)
def test_a_different_seed_moves_every_scenario(scenario):
    """Every one of them, which is stronger than it looks.

    A generator whose noise only touched the bus would leave the seven cabin scenarios byte
    identical for every seed — reproducible, and reproducing exactly one world. Arrival delay is
    what reaches all of them, so this is the assertion that the seed is wired to the whole
    generator rather than to the part that happened to be tested.
    """
    assert frames(scenario, seed=1, start=1000.0) != frames(scenario, seed=2, start=1000.0)


def _speeds(scenario: Scenario, *, seed: int = 3) -> list:
    """Only the bus frames. A sighting's payload has a `value` too, and reading both would let
    'animal' into a set of speeds — which is how the first version of the test below failed."""
    return [f.payload["value"] for f in frames(scenario, seed=seed) if f.source == "can0"]


def test_speed_noise_scales_with_the_reading():
    """Fractional noise, not absolute, and the three shipped speeds are what shows the difference.

    A stopped car must read exactly 0.0: `animal_stopped` and `animal_moving` differ in nothing
    else, and an absolute jitter would have a parked car reporting -0.14 kph — not a slow car but
    a sensor fault, and a value below the signal's own declared minimum.

    A walking-pace car must stay the safe side of the rule's 5.0 kph threshold whatever the seed.
    Absolute jitter big enough to be visible at 45 kph would be ±2 kph at 3 kph as well, which
    puts `animal_walking_pace` over the line and makes a gold row's outcome a function of the
    seed. Fifteen times the spread at fifteen times the speed is the property that prevents it;
    a factor of five is asserted, because the point is the scaling and not the constant.
    """
    stopped = _speeds(SCENARIOS["animal_stopped"])
    assert stopped and set(stopped) == {0.0}

    fast = _speeds(SCENARIOS["animal_moving"])
    slow = _speeds(SCENARIOS["animal_walking_pace"])
    assert len(set(fast)) > 1, "a bus reporting one exact number is not a measurement"
    assert (max(fast) - min(fast)) > (max(slow) - min(slow)) * 5
    assert all(v < 5.0 for v in slow), "noise must not move a gold row across a rule threshold"


# --- the rows themselves ------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", GOLD_SCENARIOS, ids=lambda s: s.id)
def test_every_generated_row_is_a_valid_input(scenario, tmp_path):
    """Decoded back through the same call `process_pending` makes.

    This is what keeps `sim/environment.py` honest about a registry it is not allowed to import:
    the source names and the payload field names are spelled there by hand, and `input_from_row`
    is where a drift between those spellings and `intake/envelope.py` becomes a failure. It
    checks both directions at once — the source must be declared, and its declared payload type
    must be the one the fields actually build.
    """
    from intake.store import Store
    from sim.vehicle import SqliteVehicle

    car = SqliteVehicle(str(tmp_path / "car.db"))
    car.init_schema()
    store = Store(car.conn)
    write(store, frames(scenario, seed=5, start=1000.0))

    rows = store.pending()
    assert len(rows) == len(frames(scenario, seed=5, start=1000.0))
    for row in rows:
        item = input_from_row(row)                 # raises on an undeclared source or bad fields
        assert isinstance(item.payload, SOURCES[item.source].accepts)
    kinds = {type(input_from_row(r).payload) for r in rows}
    assert kinds <= {Percept, SignalWrite, Utterance}


def test_the_rows_come_out_in_the_order_the_world_happened():
    """Sorted by `at`, so a store read by id reads chronologically.

    Not a correctness requirement — `Store.pending` orders by `(at, id)` — which is exactly why
    it is worth pinning: nothing downstream would fail if this quietly stopped, and the person
    reading the table by hand is the one who would pay for it.
    """
    generated = frames(SCENARIOS["consent_yes"], seed=11, start=1000.0)
    assert [f.at for f in generated] == sorted(f.at for f in generated)


def test_a_camera_is_chosen_by_the_key_s_namespace():
    """`outside.` is the forward camera; everything else is the cabin.

    Provenance that is derived rather than defaulted. Every gold row is driven through
    `Session.observe`, whose default source is the cabin camera — so the hand-authored
    `outside.front_object` rows all claim the cabin saw the road, which is the exact defect
    `intake/sources.py` was declared to end. Generated rows do not inherit it.
    """
    cabin = frames(SCENARIOS["child_clear"], seed=1)
    assert [f.source for f in cabin] == ["cabin_cam"]
    front = [f for f in frames(SCENARIOS["animal_moving"], seed=1) if f.source != "can0"]
    assert [f.source for f in front] == ["front_cam"]
    # Overridable, so a scenario can state a camera reporting what it should not be able to see.
    odd = frames(Scenario("odd", span=1.0,
                          sees=(Sighting(0.5, "outside.front_object", "animal", 0.9,
                                         source="cabin_cam"),)), seed=1)
    assert [f.source for f in odd] == ["cabin_cam"]


def test_the_bus_publishes_a_signal_the_car_declares_sensed():
    """An actuated signal never goes stale, so a bus frame for one would model a decay that does
    not exist — and `animal_stale_bus` would be silent for a reason the vehicle does not have."""
    assert SPEED in {(row[0], row[1]) for row in sensed_signals()}


def test_a_scenario_refuses_an_event_outside_its_span():
    """A typo with a silent consequence, caught at construction.

    The bus stops at `span`, so a sighting stated after it reads against a signal that has gone
    quiet: the scenario would be testing staleness while its name claimed it was testing a rule.
    """
    with pytest.raises(ValueError, match="outside the"):
        Scenario("late", span=1.0, sees=(Sighting(2.0, "inside.rear_occupant", "child", 0.9),))
    with pytest.raises(ValueError, match="outside the"):
        Scenario("early", span=1.0, says=(Speech(-1.0, "好"),))
