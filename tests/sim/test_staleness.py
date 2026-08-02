"""A sensed signal can go stale. An actuated one cannot, and that asymmetry is the point.

`updated_at` has been written on every signal write since Spec 7 and read by NOTHING, so
`SignalAbove("vehicle.all", "speed_kph", above=5.0)` fires the animal warning off a speed
frozen ten minutes ago exactly as readily as off a live one -- a dead bus and a stationary car
are indistinguishable to every rule. Perception received precisely this discipline in the scene
engine (`Observation.is_live`, read-time expiry, no sweeper) and the car never did. These tests
are the car catching up.

Read-time, not swept: nothing deletes an old row. A signal is stale because of when it was last
written and when you asked, not because a background job got round to it -- the same choice
`SceneContext` made, and for the same reason. A sweeper introduces a window in which a value is
expired but still readable, and that window is exactly where a wrong action comes from.
"""
import time

import pytest

from t2f.cards import load_catalog
from sim.seed import seed_from_catalog, sensed_max_age, sensed_signals
from sim.vehicle import SqliteVehicle

CARDS = load_catalog("data/catalog")


@pytest.fixture
def car():
    v = SqliteVehicle(":memory:")
    v.init_schema()
    seed_from_catalog(v, CARDS)
    return v


# --- the wiring ---------------------------------------------------------------------------

def test_the_hub_actually_reads_the_declared_max_age():
    """Wired through a guarded lazy import, so a rename here is swallowed silently and
    staleness becomes a permanent no-op with nothing to notice. This is the test that
    notices."""
    from intake.hub import _declared_max_age
    from sim.seed import sensed_max_age
    assert sensed_max_age("vehicle.all", "speed_kph") == 2.0
    assert _declared_max_age("vehicle.all", "speed_kph") == 2.0
    assert _declared_max_age("window.all", "window_child_lock") is None


# --- age ------------------------------------------------------------------------------------

def test_a_freshly_written_signal_has_no_age_to_speak_of(car):
    car.set_signal("vehicle.all", "speed_kph", 45.0)
    assert car.signal_age("vehicle.all", "speed_kph", time.time()) == pytest.approx(0.0, abs=0.5)


def test_age_grows_as_the_clock_advances(car):
    """Both calls read the same stamp, so the difference is exactly the clock's -- no sleep,
    and nothing that can flake on a slow machine."""
    car.set_signal("vehicle.all", "speed_kph", 45.0)
    now = time.time()
    young = car.signal_age("vehicle.all", "speed_kph", now)
    old = car.signal_age("vehicle.all", "speed_kph", now + 10.0)
    assert old - young == pytest.approx(10.0)
    assert old > young


def test_writing_again_makes_a_signal_young_again(car):
    """What keeps a pumped bus fresh: republishing the same value is a write, and a write is
    what resets the age. Without this, `/bus on` could not undo `/bus off`."""
    now = time.time()
    car.set_signal("vehicle.all", "speed_kph", 45.0)
    assert car.signal_age("vehicle.all", "speed_kph", now + 10.0) > 5.0
    car.set_signal("vehicle.all", "speed_kph", 45.0)
    assert car.signal_age("vehicle.all", "speed_kph", time.time()) < 5.0


def test_a_signal_the_car_does_not_hold_has_no_age(car):
    """`None`, never an exception and never a large number standing in for "unknown".

    "I have no such signal" and "I have one, and it is ancient" are different facts and need
    different words: a sentinel like 1e9 makes the first look like the second, so a consumer
    would report a signal this car has never heard of as merely stale.
    """
    assert car.signal_age("no.such", "thing", time.time()) is None


def test_every_declared_sensed_signal_is_seeded_and_has_an_age(car):
    """The declaration and the seeded car agree, and every row the declaration names can
    actually answer the freshness question -- a `max_age` on a signal with no readable age
    would be a check that never fires, which is worse than no check."""
    now = time.time()
    for entity, attribute, *_ in sensed_signals():
        assert car.get_signal(entity, attribute) is not None, f"{entity}/{attribute} not seeded"
        age = car.signal_age(entity, attribute, now)
        assert age is not None, f"{entity}/{attribute} has no age"
        assert age == pytest.approx(0.0, abs=5.0), f"{entity}/{attribute} seeded stale"


# --- max age, and the asymmetry it encodes --------------------------------------------------

def test_a_sensed_signal_declares_how_fast_it_decays():
    assert sensed_max_age("vehicle.all", "speed_kph") == 2.0


def test_an_actuated_signal_has_no_max_age_at_all():
    """A window position holds until something commands it otherwise: nothing measures it, so
    there is nothing to go quiet. A speed is a continuous measurement and its absence means the
    bus stopped.

    That asymmetry is why `max_age` lives only on the sensed rows rather than on every signal
    with a default -- a default would have to be either "never stale" (a column of dead config)
    or a number, and a number invented for a window position would silence `open_window` the
    moment nobody touched the car for a minute.
    """
    assert sensed_max_age("window.all", "window_position") is None
    assert sensed_max_age("window.all", "window_child_lock") is None
    assert sensed_max_age("climate.all", "ac_power") is None
    assert sensed_max_age("climate.driver", "temperature") is None


def test_a_signal_nothing_declares_at_all_has_no_max_age():
    assert sensed_max_age("no.such", "thing") is None


# --- end to end, over a real car ------------------------------------------------------------

def test_a_real_signal_goes_stale_through_the_world(car):
    """The whole chain: the car stamps, `signal_age` reads the stamp, `sensed_max_age` says how
    long that is good for, and the hub turns the two into an absence."""
    from intake.hub import WorldView
    from tests.perception import perception_store

    world = WorldView(perception_store(), car)
    car.set_signal("vehicle.all", "speed_kph", 45.0)
    now = time.time()
    assert world.signal("vehicle.all", "speed_kph", now) == 45.0
    assert world.signal_status("vehicle.all", "speed_kph", now)[0] == "live"

    status, detail = world.signal_status("vehicle.all", "speed_kph", now + 4.2)
    assert status == "stale"
    assert detail == "vehicle.all/speed_kph is stale (4.2s > 2.0s)"
    assert world.signal("vehicle.all", "speed_kph", now + 4.2) is None
    assert "vehicle.all/speed_kph" not in world.live_facts(now + 4.2)


def test_a_real_actuated_signal_never_goes_stale_through_the_world(car):
    """Ten minutes of silence, and the child lock still reads False rather than absent. If it
    did not, `open_window`'s precondition would start passing on an unknown lock."""
    from intake.hub import WorldView
    from tests.perception import perception_store

    world = WorldView(perception_store(), car)
    later = time.time() + 600.0
    assert world.signal_status("window.all", "window_child_lock", later)[0] == "live"
    assert world.signal("window.all", "window_child_lock", later) is False
