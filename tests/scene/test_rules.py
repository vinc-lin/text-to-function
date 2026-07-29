"""A rule matches, nearly matches, is rejected, or does not apply. Nothing else."""
import pytest

from scene.context import Observation, SceneContext
from scene.rules import Observed, Rule, Signal, Verdict, evaluate, RULES
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
