"""A rule matches, nearly matches, is rejected, or does not apply. Nothing else.

Every case is stated as ONE world rather than a context and a car, because that is now all a
rule is given: which store holds an answer is not the rule's business, and a rule that later
conditions on both reads them the same way.
"""
import pytest

from intake.hub import WorldView
from scene.context import Observation, SceneContext
from scene.rules import (Observed, Rule, Signal, SignalAbove, Verdict, evaluate,
                         evaluate_explained, RULES)
from t2f.types import ToolCall

RULE = Rule(
    id="r", description="d",
    when=(Observed("inside.rear_occupant", equals="child"),
          Signal("window.all", "window_child_lock", equals=False)),
    threshold=0.80, floor=0.50, persist_for=0.0, priority=50, cooldown=120.0,
    intent="ask_rear_child_lock",
    proposes=ToolCall("set_window_child_lock", {"enabled": True}))


class SpyCar:
    """Signals the test dictates, all of them freshly written.

    Keyed by (entity, attribute) and passed as a dict rather than **kwargs, because a tuple
    cannot be a keyword. `signal_age` answers 0.0 for what it holds and None for what it does
    not, which is what a car that has just been written looks like.
    """

    def __init__(self, signals=None):
        self._s = dict(signals or {})
        self.writes = []

    def get_signal(self, entity, attribute):
        return self._s.get((entity, attribute))

    def signal_age(self, entity, attribute, now):
        return 0.0 if (entity, attribute) in self._s else None

    def set_signal(self, *a, **k):
        self.writes.append(a)


class AgedCar(SpyCar):
    """A car whose signals are as old as the test says. A dead bus, in one class."""

    def __init__(self, age, signals=None):
        super().__init__(signals)
        self._age = age

    def signal_age(self, entity, attribute, now):
        return self._age if (entity, attribute) in self._s else None


def _ctx(confidence=0.9, value="child", at=100.0, ttl=300.0):
    ctx = SceneContext()
    ctx.update(Observation("inside.rear_occupant", value, confidence, "cabin_cam", at, ttl))
    return ctx


def _world(confidence=0.9, value="child", at=100.0, ttl=300.0, lock=False, car=None):
    return WorldView(_ctx(confidence, value, at, ttl),
                     car or SpyCar({("window.all", "window_child_lock"): lock}))


def test_all_conditions_met_is_a_match():
    assert evaluate(RULE, _world(0.9, lock=False), now=100.0) is Verdict.MATCH


def test_a_false_signal_condition_is_a_rejection():
    """The lock is already on. There is nothing to ask about, so this is silence — and it is
    checked BEFORE confidence, being the cheapest and most definitive answer available."""
    assert evaluate(RULE, _world(0.9, lock=True), now=100.0) is Verdict.REJECT


def test_the_signal_check_precedes_the_confidence_check():
    """A weak observation against an already-locked car is still a rejection, not a
    near-miss: routing it to the model would spend a decode on a settled question."""
    assert evaluate(RULE, _world(0.60, lock=True), now=100.0) is Verdict.REJECT


def test_confidence_between_floor_and_threshold_is_a_near_miss():
    assert evaluate(RULE, _world(0.62), now=100.0) is Verdict.NEAR_MISS


def test_confidence_below_the_floor_does_not_apply():
    """Too weak to act on AND too weak to ask about — the model sees nothing."""
    assert evaluate(RULE, _world(0.40), now=100.0) is Verdict.NOT_APPLICABLE


def test_a_stale_observation_does_not_apply():
    assert evaluate(RULE, _world(0.9, ttl=10.0), now=200.0) is Verdict.NOT_APPLICABLE


def test_a_different_observed_value_does_not_apply():
    """Absence of the condition is not ambiguity about it."""
    assert evaluate(RULE, _world(0.9, value="adult"), now=100.0) is Verdict.NOT_APPLICABLE


def test_an_unsatisfied_persistence_window_is_a_near_miss():
    persistent = Rule(**{**RULE.__dict__, "persist_for": 5.0})
    assert evaluate(persistent, _world(0.9, at=100.0), now=102.0) is Verdict.NEAR_MISS
    assert evaluate(persistent, _world(0.9, at=100.0), now=106.0) is Verdict.MATCH


def test_the_shipped_rule_set_is_not_empty_and_every_rule_is_well_formed():
    assert RULES
    for r in RULES:
        assert r.id and r.description and r.when
        assert 0.0 <= r.floor <= r.threshold <= 1.0
        assert r.cooldown > 0


def test_observed_keys_lists_only_perception_conditions():
    assert RULE.observed_keys == ("inside.rear_occupant",)


# --- the verdict carries its reason -------------------------------------------------------

def test_a_rejection_names_the_signal_that_already_holds():
    """REJECT with no detail renders as 'nothing happened', which is what the display
    exists to stop saying."""
    verdict, why = evaluate_explained(RULE, _world(0.9, lock=True), now=100.0)
    assert verdict is Verdict.REJECT
    assert "window.all/window_child_lock" in why and "True" in why


def test_a_near_miss_names_the_confidence_and_the_band():
    verdict, why = evaluate_explained(RULE, _world(0.62), now=100.0)
    assert verdict is Verdict.NEAR_MISS
    assert "0.62" in why and "0.80" in why


def test_a_below_floor_observation_says_it_was_below_the_floor():
    verdict, why = evaluate_explained(RULE, _world(0.40), now=100.0)
    assert verdict is Verdict.NOT_APPLICABLE
    assert "0.40" in why and "0.50" in why


def test_a_missing_observation_says_which_key_is_missing():
    world = WorldView(SceneContext(), SpyCar({("window.all", "window_child_lock"): False}))
    verdict, why = evaluate_explained(RULE, world, now=100.0)
    assert verdict is Verdict.NOT_APPLICABLE
    assert "inside.rear_occupant" in why


def test_a_match_explains_itself_too():
    verdict, why = evaluate_explained(RULE, _world(0.9), now=100.0)
    assert verdict is Verdict.MATCH and why


def test_evaluate_still_returns_a_bare_verdict():
    """The engine's hot path does not need the string, and every existing caller passes
    through it."""
    assert evaluate(RULE, _world(0.9), now=100.0) is Verdict.MATCH


def test_a_signal_the_car_does_not_hold_rejects_in_the_world_s_words():
    """The old shape compared the absence to the expected value and reported `is already
    None`, which reads as a car that holds the signal and answers None. The world knows the
    difference between an absence and a value, and says which one this is."""
    world = WorldView(_ctx(0.9), SpyCar())
    verdict, why = evaluate_explained(RULE, world, now=100.0)
    assert verdict is Verdict.REJECT
    assert why == "window.all/window_child_lock is not a signal this car holds"


# --- the third condition form, and the rule built on it -----------------------------------

def _moving(kph, at=100.0, confidence=0.9, car=None):
    return WorldView(_animal_ctx(confidence, at),
                     car or SpyCar({("vehicle.all", "speed_kph"): kph}))


ANIMAL = Rule(
    id="a", description="d",
    when=(Observed("outside.front_object", equals="animal"),
          SignalAbove("vehicle.all", "speed_kph", above=5.0)),
    threshold=0.70, floor=0.40, persist_for=0.0, priority=90, cooldown=30.0,
    intent="notify_animal_ahead", proposes=None)


def _animal_ctx(confidence=0.9, at=100.0):
    ctx = SceneContext()
    ctx.update(Observation("outside.front_object", "animal", confidence, "front_cam", at, 300.0))
    return ctx


def test_a_moving_car_satisfies_signal_above():
    assert evaluate(ANIMAL, _moving(45.0), now=100.0) is Verdict.MATCH


def test_a_stationary_car_rejects_it():
    """Standing still, an animal ahead is not a warning worth making."""
    assert evaluate(ANIMAL, _moving(0.0), now=100.0) is Verdict.REJECT


def test_the_boundary_is_strictly_above():
    assert evaluate(ANIMAL, _moving(5.0), now=100.0) is Verdict.REJECT
    assert evaluate(ANIMAL, _moving(5.1), now=100.0) is Verdict.MATCH


def test_an_absent_sensed_signal_rejects_rather_than_raises():
    """A rule naming a signal the car does not hold must fall silent, not explode — and the
    contract sweep separately guarantees no shipped rule can be in that state."""
    assert evaluate(ANIMAL, _moving(0.0, car=SpyCar()), now=100.0) is Verdict.REJECT


def test_signal_above_explains_itself():
    verdict, why = evaluate_explained(ANIMAL, _moving(0.0), now=100.0)
    assert verdict is Verdict.REJECT
    assert "speed_kph" in why and "0.0" in why and "5.0" in why


# --- a signal that went quiet ---------------------------------------------------------------

def test_a_stale_signal_rejects_and_names_both_ages():
    """The reason a stale read is not allowed to borrow the words of a slow one.

    A car doing 45 whose speed stopped arriving four seconds ago is not a stationary car, and
    "vehicle.all/speed_kph is 45.0, not above 5.0" would be a false statement about the world:
    it reads as "the car is slow" when the truth is "the bus is quiet". Both ages are named
    because the number alone does not say what it was measured against.

    Driven by the REAL declaration -- sim/seed.py says speed is good for 2.0s -- because the
    hub reaches that through a guarded lazy import that fails silently, and a monkeypatched max
    age would prove the arithmetic while the wiring read nothing at all.
    """
    world = _moving(45.0, car=AgedCar(4.2, {("vehicle.all", "speed_kph"): 45.0}))
    verdict, why = evaluate_explained(ANIMAL, world, now=100.0)
    assert verdict is Verdict.REJECT
    assert why == "vehicle.all/speed_kph is stale (4.2s > 2.0s)"
    assert "not above" not in why


def test_a_stale_signal_silences_the_rule_exactly_as_an_absent_one_does():
    """One verdict for both absences, so no rule has to learn a third case."""
    stale = _moving(45.0, car=AgedCar(4.2, {("vehicle.all", "speed_kph"): 45.0}))
    absent = _moving(45.0, car=SpyCar())
    assert evaluate(ANIMAL, stale, now=100.0) is evaluate(ANIMAL, absent, now=100.0)


def test_an_actuated_signal_never_goes_stale_and_the_rule_still_fires():
    """The asymmetry, seen from a rule. A window position holds until something commands it
    otherwise, so ten minutes of silence leaves the child-lock question askable — whereas the
    same ten minutes of silence on a speed would have silenced the animal warning."""
    world = WorldView(_ctx(0.9), AgedCar(600.0, {("window.all", "window_child_lock"): False}))
    assert evaluate(RULE, world, now=100.0) is Verdict.MATCH


def test_the_animal_rule_outranks_the_child_lock_question():
    """A warning beats a convenience question, and this is the first time two shipped rules
    can contend at all."""
    from scene.rules import ANIMAL_AHEAD, REAR_CHILD_WINDOW_LOCK
    assert ANIMAL_AHEAD.priority > REAR_CHILD_WINDOW_LOCK.priority


def test_the_animal_rule_only_warns():
    """A warning proposes nothing: there is no vehicle action that makes an animal safe, and
    the driver is the one who has to act."""
    from scene.rules import ANIMAL_AHEAD
    assert ANIMAL_AHEAD.proposes is None


def test_the_animal_rule_is_readier_to_fire_than_the_question():
    """A deliberate asymmetry. A missed animal is worse than a spurious warning; a spurious
    child-lock question is merely annoying, so it demands more confidence."""
    from scene.rules import ANIMAL_AHEAD, REAR_CHILD_WINDOW_LOCK
    assert ANIMAL_AHEAD.threshold < REAR_CHILD_WINDOW_LOCK.threshold
