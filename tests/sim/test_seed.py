"""The starting car.

Two defects are cheap to catch here and expensive in the executor:

* a precondition naming a signal no function writes -- dead config that can never fire, and
  that a mere "preconditions_for(fn) is non-empty" assertion happily accepts;
* a limit taken from the wrong card of an aliased pair -- a valid operation refused as
  physically impossible, or (worse) a limit check that silently never fires.

Both are asserted against the real 92-card catalog, because both only exist there.
"""
import json

import pytest

from t2f.cards import load_catalog
from sim.vehicle import SqliteVehicle
from sim.seed import seed_from_catalog
# One definition of "a signal some function writes", shared with the mapping tests: if the
# two drifted apart this file would stop guarding anything.
from tests.sim.test_mapping import _every_signal

CARDS = load_catalog("data/catalog")
BY = {c.name: c for c in CARDS}


@pytest.fixture
def car():
    v = SqliteVehicle(":memory:")
    v.init_schema()
    seed_from_catalog(v, CARDS)
    return v


def _signal_rows(car):
    return car.conn.execute(
        "SELECT entity, attribute, value, unit, min_value, max_value FROM signal").fetchall()


# --- the plan's four ---------------------------------------------------------------------

def test_numeric_signals_exist_with_limits(car):
    assert car.limits_of("climate.driver", "temperature") == (16.0, 32.0)


def test_signals_start_at_a_plausible_value(car):
    assert car.get_signal("climate.driver", "temperature") is not None


def test_preconditions_are_seeded(car):
    names = [p["attribute"] for p in car.preconditions_for("set_temperature")]
    assert "ac_power" in names


def test_child_lock_precondition_on_rear_window(car):
    pres = car.preconditions_for("open_window")
    assert [(p["entity"], p["attribute"], p["equals"]) for p in pres] == [
        ("window.all", "window_child_lock", False)]


# --- every precondition must address a signal the catalog can actually produce -----------

def test_every_precondition_points_at_a_real_writable_signal(car):
    """A precondition on a row nothing writes is dead: never satisfied, never violated,
    and indistinguishable from working config until an executor needs it."""
    writable = _every_signal()
    rows = car.conn.execute(
        "SELECT function, requires_entity, requires_attr FROM precondition").fetchall()
    assert rows, "no preconditions seeded at all"
    dead = [(r["function"], f"{r['requires_entity']}.{r['requires_attr']}")
            for r in rows if (r["requires_entity"], r["requires_attr"]) not in writable]
    assert dead == [], f"preconditions on signals no function writes: {dead}"


def test_every_precondition_names_a_function_in_the_catalog(car):
    rows = car.conn.execute("SELECT DISTINCT function FROM precondition").fetchall()
    assert [r["function"] for r in rows if r["function"] not in BY] == []


def test_the_gating_signals_are_seeded_to_the_useful_default(car):
    """The A/C on and the child lock off: the state in which most commands just work."""
    assert car.get_signal("climate.all", "ac_power") is True
    assert car.get_signal("window.all", "window_child_lock") is False


# --- limits come from the right card -----------------------------------------------------

@pytest.mark.parametrize("entity,attribute,expected", [
    ("climate.driver", "temperature",     (16.0, 32.0)),   # set_temperature, celsius
    ("seat.driver",    "seat_massage",    (0.0, 5.0)),     # set_seat_massage, level
    ("seat.driver",    "lumbar_support",  (0.0, 4.0)),     # set_lumbar_support, level
    ("climate.driver", "fan_speed",       (1.0, 7.0)),     # set_fan_speed, level
])
def test_limits_come_from_that_signals_own_card(car, entity, attribute, expected):
    assert car.limits_of(entity, attribute) == expected


def test_an_aliased_signal_keeps_the_numeric_cards_limits(car):
    """open_window is a boolean and carries no range; set_window_position carries 0-100.
    They are one actuator, so the row must end up with the range, whichever is seeded
    first -- otherwise the executor's physical-limit check is a no-op for windows."""
    assert car.limits_of("window.driver", "window_position") == (0.0, 100.0)
    assert car.limits_of("window.all", "sunroof_position") == (0.0, 100.0)


# --- the car starts in a state it is allowed to be in ------------------------------------

def test_every_seeded_value_is_inside_its_own_limits(car):
    """A car that starts out of range makes the first operation behave bizarrely."""
    bad = []
    for r in _signal_rows(car):
        if r["min_value"] is None and r["max_value"] is None:
            continue
        value = json.loads(r["value"])
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            bad.append((r["entity"], r["attribute"], value, "not numeric but has limits"))
        elif not (r["min_value"] <= value <= r["max_value"]):
            bad.append((r["entity"], r["attribute"], value,
                        (r["min_value"], r["max_value"])))
    assert bad == [], f"seeded outside their own limits: {bad}"


def test_numeric_signals_all_carry_limits(car):
    """The converse: a numeric signal without limits is a check that never fires."""
    naked = [(r["entity"], r["attribute"]) for r in _signal_rows(car)
             if r["min_value"] is None
             and isinstance(json.loads(r["value"]), (int, float))
             and not isinstance(json.loads(r["value"]), bool)]
    assert naked == []


# --- the seeded car and the mapping agree on what exists ---------------------------------

# The old guard, `test_the_car_holds_exactly_the_signals_the_catalog_can_write`, lived here.
# `test_the_car_holds_exactly_what_it_can_write_plus_what_it_senses` below replaces it: same
# two directions, a definition of "legitimate" widened by the declared sensed signals. Keeping
# both would mean one of them must fail — the sensed row is either legitimate or it is not.
#
# Positions still come from the card, which is what the old docstring was guarding: a fixed
# driver/passenger/rear/all list invents rows for cards that accept neither
# (seat.rear/lumbar_support) and misses seat.left/right. The replacement catches that too.

def test_speed_is_seeded_and_stationary(car):
    """A car nobody has driven is not moving. 0.0 is a meaningful value, not a missing one."""
    assert car.get_signal("vehicle.all", "speed_kph") == 0.0


def test_a_sensed_signal_carries_limits(car):
    lo, hi = car.limits_of("vehicle.all", "speed_kph")
    assert lo == 0.0 and hi == 240.0


def test_every_sensed_signal_declares_how_fast_it_decays():
    """A sensed row without a `max_age` is a signal that reads live forever, which is the state
    the whole staleness discipline exists to end. The behaviour lives in
    tests/sim/test_staleness.py; this is the shape guard, next to the declaration it guards, so
    a row added here cannot quietly arrive without one."""
    from sim.seed import sensed_max_age, sensed_signals
    ages = {(e, a): sensed_max_age(e, a) for e, a, *_ in sensed_signals()}
    naked = [k for k, v in ages.items() if type(v) is not float or v <= 0]
    assert naked == [], f"sensed signals that can never go stale: {naked}"


def test_no_function_writes_a_sensed_signal():
    """That is what makes it sensed. If a card ever gains one, this is the wrong category
    for it and the seeder should be deriving it like everything else."""
    from sim.seed import sensed_signals
    writable = set(_every_signal())
    assert {(e, a) for e, a, *_ in sensed_signals()} & writable == set()


def test_the_car_holds_exactly_what_it_can_write_plus_what_it_senses(car):
    """The original guard, widened rather than weakened — both directions still enforced,
    and a sensed signal nobody declared is still a failure."""
    from sim.seed import sensed_signals
    seeded = {(r["entity"], r["attribute"]) for r in _signal_rows(car)}
    legitimate = set(_every_signal()) | {(e, a) for e, a, *_ in sensed_signals()}
    assert seeded - legitimate == set(), "seeded rows nothing can write or sense"
    assert legitimate - seeded == set(), "legitimate signals the car does not have"
