"""One view of everything currently true. It owns nothing."""
import pytest

from intake.hub import WorldView
from scene.context import Observation, SceneContext


class SpyCar:
    # Signals arrive as one dict, not as **kwargs: they are keyed by (entity, attribute) and
    # `f(**{("a","b"): 1})` is a TypeError — keyword keys must be strings.
    def __init__(self, signals=None):
        self._s = dict(signals or {})
        self.writes = []

    def get_signal(self, entity, attribute):
        return self._s.get((entity, attribute))

    def signal_age(self, entity, attribute, now):
        return 0.0 if (entity, attribute) in self._s else None

    def set_signal(self, *a, **k):
        self.writes.append(a)


def _ctx(**kw):
    ctx = SceneContext()
    ctx.update(Observation(kw.get("key", "inside.rear_occupant"), kw.get("value", "child"),
                           kw.get("confidence", 0.9), "cabin_cam",
                           kw.get("at", 100.0), kw.get("ttl", 300.0)))
    return ctx


def test_an_observation_reads_through_to_the_perception_store():
    world = WorldView(_ctx(), SpyCar())
    assert world.observation("inside.rear_occupant", now=150.0).value == "child"


def test_a_stale_observation_reads_as_absent():
    world = WorldView(_ctx(ttl=10.0), SpyCar())
    assert world.observation("inside.rear_occupant", now=200.0) is None


def test_a_signal_reads_through_to_the_car():
    world = WorldView(SceneContext(), SpyCar({("vehicle.all", "speed_kph"): 45.0}))
    assert world.signal("vehicle.all", "speed_kph", now=100.0) == 45.0


def test_a_missing_signal_reads_as_absent():
    assert WorldView(SceneContext(), SpyCar()).signal("no.such", "thing", now=100.0) is None


def test_live_facts_covers_both_stores():
    world = WorldView(_ctx(), SpyCar({("vehicle.all", "speed_kph"): 45.0}))
    facts = world.live_facts(now=150.0)
    assert "inside.rear_occupant" in facts
    assert "vehicle.all/speed_kph" in facts


def test_live_facts_omits_what_has_expired():
    world = WorldView(_ctx(ttl=10.0), SpyCar())
    assert world.live_facts(now=200.0) == {}


def test_the_hub_holds_nothing_writable():
    """`read-through` in a docstring is not an enforcement mechanism. VehicleFacts carried this
    property and WorldView absorbs it: a hub that could write would be a second route to the
    car with none of the executor's checks."""
    world = WorldView(SceneContext(), SpyCar())
    reachable = [v for v in vars(world).values() if hasattr(v, "set_signal")]
    assert reachable == []


def test_the_hub_holds_only_readers():
    """The stronger form: not "no set_signal in sight" but "neither store is here at all".

    The test above would still pass while holding the whole SceneContext, and `update`
    reachable from the hub is a second door into perception — a belief arriving through it
    carries no source and no provenance, which is the exact thing the envelope exists to make
    impossible. Every attribute being a bound reader leaves nothing to write through.
    """
    ctx, car = _ctx(), SpyCar({("vehicle.all", "speed_kph"): 45.0})
    world = WorldView(ctx, car)
    held = list(vars(world).values())
    assert all(callable(v) for v in held), held
    assert ctx not in held and car not in held


def test_reading_never_writes():
    car = SpyCar({("vehicle.all", "speed_kph"): 45.0})
    world = WorldView(_ctx(), car)
    world.observation("inside.rear_occupant", now=150.0)
    world.signal("vehicle.all", "speed_kph", now=150.0)
    world.live_facts(now=150.0)
    assert car.writes == []


# --- why a signal read as absent ---------------------------------------------------------
#
# Task 4's rejection has to say `vehicle.all/speed_kph is stale (4.2s > 2.0s)` rather than a
# bare "not above 5.0", which on a stale value reads as "the car is slow" when the truth is
# "the bus is quiet".


class AgedCar(SpyCar):
    """A car whose signals have an age of the test's choosing."""

    def __init__(self, age, signals=None):
        super().__init__(signals)
        self._age = age

    def signal_age(self, entity, attribute, now):
        return self._age if (entity, attribute) in self._s else None


class AgelessCar(SpyCar):
    """A car from before Task 3: it holds signals and cannot say how old they are."""
    signal_age = None


def test_a_fresh_signal_reports_itself_live():
    world = WorldView(SceneContext(), SpyCar({("vehicle.all", "speed_kph"): 45.0}))
    status, _ = world.signal_status("vehicle.all", "speed_kph", now=100.0)
    assert status == "live"


def test_a_signal_the_car_does_not_hold_reports_itself_missing():
    world = WorldView(SceneContext(), SpyCar())
    status, detail = world.signal_status("no.such", "thing", now=100.0)
    assert status == "missing"
    assert "no.such/thing" in detail


def test_a_stale_signal_names_both_ages():
    """The whole point of the status: the detail is the sentence, so the rule that rejects on
    it cannot phrase the same fact a second, different way.

    Driven by the REAL declaration -- `sim/seed.py` says speed is good for 2.0s -- rather than
    a monkeypatched `_declared_max_age`. A patched one proves the arithmetic and nothing about
    the wiring, and the wiring is a guarded lazy import that fails silently, so a test that
    supplies its own max age would keep passing over a hub that reads nothing at all.
    """
    world = WorldView(SceneContext(), AgedCar(4.2, {("vehicle.all", "speed_kph"): 45.0}))
    status, detail = world.signal_status("vehicle.all", "speed_kph", now=500.0)
    assert status == "stale"
    assert detail == "vehicle.all/speed_kph is stale (4.2s > 2.0s)"


def test_a_stale_signal_reads_as_absent():
    """Identical to a signal the car does not hold, so every condition on it rejects and the
    engine falls silent rather than acting on a value from ten minutes ago."""
    world = WorldView(SceneContext(), AgedCar(4.2, {("vehicle.all", "speed_kph"): 45.0}))
    assert world.signal("vehicle.all", "speed_kph", now=500.0) is None
    assert world.live_facts(now=500.0) == {}


def test_a_signal_with_no_declared_decay_never_goes_stale():
    """An actuated signal — a window position — holds until something commands it otherwise,
    so it has no max age and no amount of age makes it stale.

    Ten minutes of silence and it still reads live, from the real declaration: `sim/seed.py`
    puts `max_age` on the sensed rows only, and a window position is not one of them.
    """
    world = WorldView(SceneContext(), AgedCar(600.0, {("window.all", "window_position"): 0}))
    assert world.signal_status("window.all", "window_position", now=1000.0)[0] == "live"


def test_a_car_that_cannot_report_age_reads_as_live():
    """A signal with no known age is not a stale signal. Treating it as one would silence every
    rule the moment staleness landed, and `SqliteVehicle` is not the only thing that can be
    passed here — the sweep's spies and any future car port are cars too."""
    world = WorldView(SceneContext(), AgelessCar({("vehicle.all", "speed_kph"): 45.0}))
    assert world.signal_status("vehicle.all", "speed_kph", now=1e9)[0] == "live"
    assert world.signal("vehicle.all", "speed_kph", now=1e9) == 45.0


def test_live_observations_keeps_what_live_facts_drops():
    """The prompt wants {name: value}; the engine's fallback needs confidence to describe a
    near-miss and the live key set to decide what is unconsumed. Two callers, two shapes."""
    world = WorldView(_ctx(confidence=0.62), SpyCar())
    whole = world.live_observations(now=150.0)
    assert whole["inside.rear_occupant"].confidence == 0.62
    assert whole["inside.rear_occupant"].source == "cabin_cam"


def test_live_observations_omits_what_has_expired():
    assert WorldView(_ctx(ttl=10.0), SpyCar()).live_observations(now=200.0) == {}
