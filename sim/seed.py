"""Build a plausible starting car from the catalog.

Physical limits default to the card's declared min/max but are DELIBERATELY separate: a card
says a window is 0-100, a physical window can be jammed at 60. Validation and actuation are
different questions, and requirement 4b's third branch only exists because they can disagree.

Two rules this module follows, both learned the hard way:

* **Positions come from the card, never from a fixed list.** One card is addressed
  rear/left/right and eight are driver/passenger only, so a hardcoded
  ["driver","passenger","rear","all"] seeds nine rows no function can ever write and misses
  two that it can.

* **Signals are resolved before anything is written.** `open_window` and `set_window_position`
  are ONE actuator: the boolean card carries no limits, the percentage card carries 0-100.
  `SqliteVehicle.set_signal` keeps the first row's limits on conflict (its ON CONFLICT clause
  updates value and updated_at only), so seeding card-by-card leaves the shared window and
  sunroof signals with no physical limits at all -- and a limit check that silently never
  fires is worse than no limit check. Collect per signal, decide once, write once.
"""
from __future__ import annotations

from typing import Any, Optional

from t2f.types import FunctionCard, ParamSpec
from t2f.state import primary_numeric_param
from sim.mapping import signal_for_function, _primary_param
from sim.vehicle import SqliteVehicle

# (function, entity, attribute, required_value, what the driver is told)
#
# Every entity/attribute here MUST be one that some function actually writes -- a
# precondition on a row nothing produces can never be satisfied or violated, and reads as
# working config. tests/sim/test_seed.py enforces this against the real catalog.
_PRECONDITIONS = [
    ("set_temperature", "climate.all", "ac_power", True, "空调尚未开启"),
    ("set_fan_speed",   "climate.all", "ac_power", True, "空调尚未开启"),
    ("open_window",     "window.all",  "window_child_lock", False, "车窗儿童锁已开启"),
]


# (entity, attribute, resting value, unit, min, max, max_age) -- signals the car KNOWS and
# nothing COMMANDS. Everything else in this file is derived from a function card, because until
# now the simulator modelled exactly what the Central Model can change. A real vehicle bus is
# mostly signals like these; ours had none until a scene needed to condition on one.
#
# They are declared here rather than derived precisely because no card can produce them, and
# tests/sim/test_seed.py asserts no function writes one -- if a card ever gains it, this is
# the wrong category and the seeder should derive it like everything else.
#
# `max_age` is how long a reading stays true, and it is the LAST field for a reason: it exists
# only on this list. An actuated signal has none, because a window position holds until
# something commands it otherwise -- nothing measures it, so there is nothing that can go
# quiet. A speed is a continuous measurement, and its absence means the bus stopped, which is a
# different fact from "the car is stationary" and must not read as one.
#
# 2.0s for speed is a GUESS and should be treated as provisional: no measurement supports it,
# it is a plausible number for a signal a real bus publishes at around 10 Hz, and it is one
# constant in one declaration precisely so that changing it is a one-line decision rather than
# an audit of every rule. Freshness is declared on the SIGNAL, never on a rule, because how
# fast a value decays is a property of the source -- speed at 10 Hz is dead after a second, a
# door-ajar flag is fine for a minute. One declaration, every rule inherits it, and no rule can
# forget to ask.
_SENSED = [
    ("vehicle.all", "speed_kph", 0.0, "kph", 0.0, 240.0, 2.0),
]


def sensed_signals() -> list[tuple]:
    """The declared sensed signals. Public so the seed guard can consult one definition."""
    return list(_SENSED)


def sensed_max_age(entity: str, attribute: str) -> Optional[float]:
    """How long a reading of this signal stays true; `None` if it never decays.

    One definition answers "how fast does this decay", for every reader -- the seeder, the
    session's control, the hub that turns an old value into an absence. A second copy anywhere
    is two beliefs about one bus, and the one that is not consulted is the one that is wrong.

    `None` for anything not declared sensed, which is the right answer rather than an error:
    an actuated signal genuinely has no decay, and asking about a signal that does not exist
    should not be a different kind of event from asking about one that never goes stale. Both
    mean "no amount of age makes this read as absent".

    **The name is load-bearing and must not drift.** `intake/hub.py` reaches this through a
    GUARDED lazy import, so renaming it does not break anything loudly: the import is swallowed,
    `_declared_max_age` returns `None` forever, every signal reads live, and nothing fails. That
    is exactly the dead-precondition failure this module's own docstring warns about -- config
    that can never be satisfied or violated and reads as working. One test exists solely to
    notice: `test_the_hub_actually_reads_the_declared_max_age`, in tests/sim/test_staleness.py.
    """
    for e, a, _value, _unit, _lo, _hi, max_age in _SENSED:
        if (e, a) == (entity, attribute):
            return max_age
    return None


def _positions(card: FunctionCard) -> list[Optional[str]]:
    """Every position this card can actually be called with. None addresses '<domain>.all'."""
    spec = card.param("position")
    if spec is None:
        return [None]
    out: list[Optional[str]] = list(spec.enum or [])
    if not spec.required:
        out.append(None)              # an omitted position means the whole domain
    return out


def _resting_value(spec: Optional[ParamSpec]) -> Any:
    """Where a non-numeric signal sits in a car nobody has touched yet."""
    if spec is None:
        return False
    if spec.type == "enum":
        return (spec.enum or [None])[0] if spec.enum else False
    if spec.type == "string":
        return ""
    return False                      # boolean, and anything unexpected


def _contribute(entry: dict, card: FunctionCard, source: ParamSpec | None) -> None:
    """Fold one card's view of a signal into the entry describing it.

    A numeric card always wins over a boolean/enum one on the same signal (that is the
    open_window / set_window_position pair), and two numeric cards widen to the union of
    their ranges so no value the catalog accepts is refused as physically impossible.
    Both rules are order-independent, so the seeded car does not depend on catalog order.
    """
    p = primary_numeric_param(card)
    if p is None:
        if entry["limits"] is None and entry["value"] is None:
            entry["value"] = _resting_value(source)
        return

    lo = p.minimum if p.minimum is not None else 0
    hi = p.maximum if p.maximum is not None else 100
    if entry["limits"] is not None:
        old_lo, old_hi = entry["limits"]
        lo, hi = min(lo, old_lo), max(hi, old_hi)
    entry["limits"] = (lo, hi)
    entry["unit"] = p.unit or entry["unit"]
    entry["integer"] = entry["integer"] and p.type == "integer"
    entry["value"] = lo + (hi - lo) // 2 if entry["integer"] else (lo + hi) / 2


def signal_plan(cards: list[FunctionCard]) -> dict[tuple, dict]:
    """(entity, attribute) -> {value, unit, limits} for every signal the catalog can write."""
    plan: dict[tuple, dict] = {}
    for card in cards:
        source = _primary_param(card)
        if source is None:
            continue                  # nothing addressable: lock_doors, pause_playback, ...
        spec = card.param(source)
        for position in _positions(card):
            sig = signal_for_function(card, position)
            if sig is None:
                continue
            entry = plan.setdefault(sig, {"value": None, "unit": None,
                                          "limits": None, "integer": True})
            _contribute(entry, card, spec)
    return plan


def seed_from_catalog(car: SqliteVehicle, cards: list[FunctionCard]) -> None:
    for (entity, attribute), s in signal_plan(cards).items():
        car.set_signal(entity, attribute, s["value"], unit=s["unit"], limits=s["limits"])
    # a car with the A/C on and the child lock off is the useful default
    car.set_signal("climate.all", "ac_power", True)
    car.set_signal("window.all", "window_child_lock", False)
    # After the card-derived rows, so a sensed signal can never be silently absorbed into the
    # ON CONFLICT limit-keeping above: nothing writes these, so nothing can collide with them.
    # `max_age` is not written to the row: it is a property of the SOURCE, not of the value,
    # so it belongs in the declaration and not in the car. A copy in the signal table would be
    # a second answer to "how fast does this decay" that no write path keeps in step.
    for entity, attribute, value, unit, lo, hi, _max_age in _SENSED:
        car.set_signal(entity, attribute, value, unit=unit, limits=(lo, hi))
    for fn, entity, attr, equals, detail in _PRECONDITIONS:
        car.add_precondition(fn, entity, attr, equals, detail)
