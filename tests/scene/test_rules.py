"""A rule matches, nearly matches, is rejected, or does not apply. Nothing else."""
import pytest

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


class FakeFacts:
    def __init__(self, **signals):
        self._s = signals

    def signal(self, entity, attribute):
        return self._s.get(f"{entity}/{attribute}")


def _ctx(confidence=0.9, value="child", at=100.0, ttl=300.0):
    ctx = SceneContext()
    ctx.update(Observation("inside.rear_occupant", value, confidence, "cabin_cam", at, ttl))
    return ctx


def _facts(lock=False):
    return FakeFacts(**{"window.all/window_child_lock": lock})


def test_all_conditions_met_is_a_match():
    assert evaluate(RULE, _ctx(0.9), _facts(False), now=100.0) is Verdict.MATCH


def test_a_false_signal_condition_is_a_rejection():
    """The lock is already on. There is nothing to ask about, so this is silence — and it is
    checked BEFORE confidence, being the cheapest and most definitive answer available."""
    assert evaluate(RULE, _ctx(0.9), _facts(True), now=100.0) is Verdict.REJECT


def test_the_signal_check_precedes_the_confidence_check():
    """A weak observation against an already-locked car is still a rejection, not a
    near-miss: routing it to the model would spend a decode on a settled question."""
    assert evaluate(RULE, _ctx(0.60), _facts(True), now=100.0) is Verdict.REJECT


def test_confidence_between_floor_and_threshold_is_a_near_miss():
    assert evaluate(RULE, _ctx(0.62), _facts(False), now=100.0) is Verdict.NEAR_MISS


def test_confidence_below_the_floor_does_not_apply():
    """Too weak to act on AND too weak to ask about — the model sees nothing."""
    assert evaluate(RULE, _ctx(0.40), _facts(False), now=100.0) is Verdict.NOT_APPLICABLE


def test_a_stale_observation_does_not_apply():
    assert evaluate(RULE, _ctx(0.9, ttl=10.0), _facts(False), now=200.0) is Verdict.NOT_APPLICABLE


def test_a_different_observed_value_does_not_apply():
    """Absence of the condition is not ambiguity about it."""
    assert evaluate(RULE, _ctx(0.9, value="adult"), _facts(False), now=100.0) is Verdict.NOT_APPLICABLE


def test_an_unsatisfied_persistence_window_is_a_near_miss():
    persistent = Rule(**{**RULE.__dict__, "persist_for": 5.0})
    assert evaluate(persistent, _ctx(0.9, at=100.0), _facts(False), now=102.0) is Verdict.NEAR_MISS
    assert evaluate(persistent, _ctx(0.9, at=100.0), _facts(False), now=106.0) is Verdict.MATCH


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
    verdict, why = evaluate_explained(RULE, _ctx(0.9), _facts(True), now=100.0)
    assert verdict is Verdict.REJECT
    assert "window.all/window_child_lock" in why and "True" in why


def test_a_near_miss_names_the_confidence_and_the_band():
    verdict, why = evaluate_explained(RULE, _ctx(0.62), _facts(False), now=100.0)
    assert verdict is Verdict.NEAR_MISS
    assert "0.62" in why and "0.80" in why


def test_a_below_floor_observation_says_it_was_below_the_floor():
    verdict, why = evaluate_explained(RULE, _ctx(0.40), _facts(False), now=100.0)
    assert verdict is Verdict.NOT_APPLICABLE
    assert "0.40" in why and "0.50" in why


def test_a_missing_observation_says_which_key_is_missing():
    verdict, why = evaluate_explained(RULE, SceneContext(), _facts(False), now=100.0)
    assert verdict is Verdict.NOT_APPLICABLE
    assert "inside.rear_occupant" in why


def test_a_match_explains_itself_too():
    verdict, why = evaluate_explained(RULE, _ctx(0.9), _facts(False), now=100.0)
    assert verdict is Verdict.MATCH and why


def test_evaluate_still_returns_a_bare_verdict():
    """The engine's hot path does not need the string, and every existing caller passes
    through it."""
    assert evaluate(RULE, _ctx(0.9), _facts(False), now=100.0) is Verdict.MATCH


# --- the third condition form, and the rule built on it -----------------------------------

def _moving(kph):
    return FakeFacts(**{"vehicle.all/speed_kph": kph})


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
    assert evaluate(ANIMAL, _animal_ctx(), _moving(45.0), now=100.0) is Verdict.MATCH


def test_a_stationary_car_rejects_it():
    """Standing still, an animal ahead is not a warning worth making."""
    assert evaluate(ANIMAL, _animal_ctx(), _moving(0.0), now=100.0) is Verdict.REJECT


def test_the_boundary_is_strictly_above():
    assert evaluate(ANIMAL, _animal_ctx(), _moving(5.0), now=100.0) is Verdict.REJECT
    assert evaluate(ANIMAL, _animal_ctx(), _moving(5.1), now=100.0) is Verdict.MATCH


def test_an_absent_sensed_signal_rejects_rather_than_raises():
    """A rule naming a signal the car does not hold must fall silent, not explode — and the
    contract sweep separately guarantees no shipped rule can be in that state."""
    assert evaluate(ANIMAL, _animal_ctx(), FakeFacts(), now=100.0) is Verdict.REJECT


def test_signal_above_explains_itself():
    verdict, why = evaluate_explained(ANIMAL, _animal_ctx(), _moving(0.0), now=100.0)
    assert verdict is Verdict.REJECT
    assert "speed_kph" in why and "0.0" in why and "5.0" in why


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
