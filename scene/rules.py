"""Declarative scene rules and their evaluation.

Conditions come in exactly two forms and there are no others. A closed vocabulary keeps every
rule inspectable and lets a contract test walk the whole set and assert properties over all of
it — which is what tests/scene/test_contract_sweep.py does.

Rules are dataclasses rather than YAML on purpose: one rule does not justify a loader, and the
shape is data-only, so a YAML front end is a later addition rather than a rewrite.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

from t2f.types import ToolCall


@dataclass(frozen=True)
class Observed:
    """A perception belief: what the cameras say."""
    key: str
    equals: Any


@dataclass(frozen=True)
class Signal:
    """A vehicle fact, read live from the car — never copied into Scene Context."""
    entity: str
    attribute: str
    equals: Any


Condition = Union[Observed, Signal]


class Verdict(str, Enum):
    MATCH = "match"
    NEAR_MISS = "near_miss"
    REJECT = "reject"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class Rule:
    id: str
    description: str                    # one line, shown to the fallback so it knows what exists
    when: tuple                         # ALL conditions must hold
    threshold: float                    # fire at or above this observation confidence
    floor: float                        # below this, not even a near-miss
    persist_for: float                  # seconds the observation must have held
    priority: int                       # higher wins contention
    cooldown: float                     # seconds before this rule may speak again
    intent: str                         # selects the speech template
    proposes: Optional[ToolCall] = None # what consent would execute; None for a pure notify

    @property
    def observed_keys(self) -> tuple:
        return tuple(c.key for c in self.when if isinstance(c, Observed))


def evaluate(rule: Rule, context, facts, now: float) -> Verdict:
    """The verdict alone. The engine's hot path has no use for the reason."""
    return evaluate_explained(rule, context, facts, now)[0]


def evaluate_explained(rule: Rule, context, facts, now: float) -> tuple:
    """Signal conditions first: cheapest and most definitive.

    An already-satisfied signal means there is nothing to ask about, and that answer beats any
    amount of perception uncertainty — so it is checked before confidence, or a weak detection
    against a settled car would spend a model call on a question already answered.

    The reason travels with the verdict because a bare verdict cannot be rendered usefully:
    NEAR_MISS does not say which observation was weak, and REJECT does not say which signal
    already held. These strings are diagnostics for a developer at a terminal and are NEVER
    spoken to a driver, so they name keys, entities and thresholds deliberately.
    """
    for cond in rule.when:
        if isinstance(cond, Signal):
            actual = facts.signal(cond.entity, cond.attribute)
            if actual != cond.equals:
                return Verdict.REJECT, f"{cond.entity}/{cond.attribute} is already {actual!r}"

    near = []
    for cond in rule.when:
        if not isinstance(cond, Observed):
            continue
        obs = context.get(cond.key, now)
        # No observation, a different value, or one too weak to consider: the rule simply does
        # not apply. Absence of evidence is not ambiguity about it, and treating it as a
        # near-miss would have the fallback fire on an empty context.
        if obs is None:
            return Verdict.NOT_APPLICABLE, f"no live observation for {cond.key}"
        if obs.value != cond.equals:
            return (Verdict.NOT_APPLICABLE,
                    f"{cond.key} is {obs.value!r}, not {cond.equals!r}")
        if obs.confidence < rule.floor:
            return (Verdict.NOT_APPLICABLE,
                    f"{cond.key} conf {obs.confidence:.2f} below floor {rule.floor:.2f}")
        # Both bands are reported when both are short, and every near-miss condition is kept:
        # naming only the first would hide the second reason the rule did not fire.
        if obs.confidence < rule.threshold:
            near.append(f"{cond.key} conf {obs.confidence:.2f} "
                        f"in [{rule.floor:.2f}, {rule.threshold:.2f})")
        if (now - obs.at) < rule.persist_for:
            near.append(f"{cond.key} held {now - obs.at:.0f}s of {rule.persist_for:.0f}s")
    if near:
        return Verdict.NEAR_MISS, "; ".join(near)
    return Verdict.MATCH, "all conditions met"


REAR_CHILD_WINDOW_LOCK = Rule(
    id="rear_child_window_lock",
    description="后排检测到儿童且车窗儿童锁未开启",
    when=(Observed("inside.rear_occupant", equals="child"),
          Signal("window.all", "window_child_lock", equals=False)),
    threshold=0.80,
    floor=0.50,
    # 0.0 deliberately: a child in the rear does not become more real by being observed for
    # longer, and a non-zero value would make this rule unfireable from the single /scene
    # event that is the only way a person can drive it by hand. The mechanism is still real
    # code, unit-tested with explicit `now` values.
    persist_for=0.0,
    priority=50,
    cooldown=120.0,
    intent="ask_rear_child_lock",
    proposes=ToolCall("set_window_child_lock", {"enabled": True}),
)

RULES: tuple = (REAR_CHILD_WINDOW_LOCK,)
