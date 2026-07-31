"""Properties that must hold for EVERY rule, not just the one we shipped.

Modelled on tests/e2e/test_s8_contract_sweep.py, which is the strongest thing in the suite:
a property asserted over the whole set cannot be satisfied by a lucky special case.

Every helper below walks the FULL condition vocabulary, `Signal` and `SignalAbove` alike, and
that is not incidental. This file was written when `Signal` was the only vehicle condition, so
every gate here spelled it `isinstance(c, Signal)`. The moment `animal_ahead` shipped with a
`SignalAbove`, three properties skipped themselves ("no signal condition", "rule did not
fire") and two more passed without exercising anything — the sweep shrank to fit the new rule
instead of measuring it, and reported green while doing so. A sweep that quietly narrows is
worse than no sweep, because the green is read as coverage. A fourth condition form must
break this file loudly.
"""
import re

import pytest

from scene.context import Observation
from scene.engine import SceneEngine, NO_ACTION
from scene.rules import RULES, Observed, Signal, SignalAbove
from scene.speech import speech_for
from sim.seed import seed_from_catalog
from sim.vehicle import SqliteVehicle
from t2f.cards import load_catalog
from t2f.types import ExecResult
from t2f.validate import validate_tool_call


class _Facts:
    def __init__(self, answers):
        self.answers = answers

    def signal(self, e, a):
        return self.answers.get((e, a))


class _Executor:
    def __init__(self):
        self.calls = []

    def execute(self, tc):
        self.calls.append(tc)
        return ExecResult(ok=True)


@pytest.fixture(scope="module")
def cards():
    return {c.name: c for c in load_catalog("data/catalog")}


@pytest.fixture(scope="module")
def seeded_car(cards):
    car = SqliteVehicle(":memory:")
    car.init_schema()
    seed_from_catalog(car, list(cards.values()))
    return car


def _vehicle_conditions(rule):
    """Every condition read from the car, whatever its shape. One definition, so a new form is
    added here once rather than missed in four separate isinstance gates."""
    return [c for c in rule.when if isinstance(c, (Signal, SignalAbove))]


def _satisfying_facts(rule):
    """A car in the state the rule is about. `SignalAbove` is strictly above, so clear it."""
    return _Facts({(c.entity, c.attribute):
                   (c.above + 1.0) if isinstance(c, SignalAbove) else c.equals
                   for c in _vehicle_conditions(rule)})


def _unsatisfying_facts(rule):
    """A car in which every vehicle condition FAILS.

    A boolean `Signal` inverts; any other value gets something it cannot equal; a `SignalAbove`
    gets exactly its own threshold — the strongest failing case there is, because the
    comparison is strictly above and the boundary is where an off-by-one would live.
    """
    def failing(c):
        if isinstance(c, SignalAbove):
            return c.above
        return (not c.equals) if isinstance(c.equals, bool) else object()
    return _Facts({(c.entity, c.attribute): failing(c) for c in _vehicle_conditions(rule)})


def _satisfy(engine, rule, now):
    for cond in rule.when:
        if isinstance(cond, Observed):
            engine.context.update(Observation(cond.key, cond.equals, 1.0, "test", now, 300.0))


def _tick(now=100.0):
    return Observation("_tick", 1, 1.0, "test", now, 300.0)


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_a_rule_match_never_reaches_the_car_on_its_own(cards, rule):
    """Consent is the ONLY path to the vehicle. This is the invariant the whole design exists
    to make true, so it is asserted over every rule rather than over one."""
    ex = _Executor()
    eng = SceneEngine(cards, _satisfying_facts(rule), ex, rules=(rule,))
    _satisfy(eng, rule, now=100.0)
    eng.observe(_tick(), now=100.0)
    assert ex.calls == []


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_every_proposal_validates(cards, rule):
    """A question whose answer could never be honoured must not be asked."""
    if rule.proposes is None:
        pytest.skip("notify-only rule")
    tc, errs = validate_tool_call(rule.proposes.name, dict(rule.proposes.parameters),
                                  cards, [rule.proposes.name])
    assert tc is not None, f"{rule.id}: {[e.code for e in errs]}"


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_every_signal_condition_names_a_row_the_car_actually_holds(rule, seeded_car):
    """A typo'd entity reads as None, which is != the expected value, which is a REJECT — so
    a misspelled rule is permanently and silently unfireable and no other test would notice.
    The failure mode is indistinguishable from working correctly.

    `SignalAbove` is covered too, and needs it more: `vehicle.all/speed_kph` is a SENSED
    signal, so no function card produces it and nothing else in the suite would notice it
    missing from the seeded car. This assertion is the reason sim/seed.py seeds it."""
    for cond in _vehicle_conditions(rule):
        assert seeded_car.get_signal(cond.entity, cond.attribute) is not None, \
            f"{rule.id}: no such signal {cond.entity}/{cond.attribute}"


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_every_rule_speaks_chinese_and_no_identifiers(rule):
    text = speech_for(rule.intent)
    assert text, f"{rule.id} has no speech template"
    assert not re.search(r"[A-Za-z_{}\[\]<>]", text), text


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_a_rule_stays_silent_when_its_vehicle_condition_does_not_hold(cards, rule):
    """Never ask for what is already true, and never warn about a car that is not in the state
    the warning is about — a stationary car settles the animal question exactly as an
    already-closed child lock settles the window one.

    The gate used to read `isinstance(c, Signal)`, so `animal_ahead` skipped itself with "no
    signal condition" while holding one of a different shape. The skip below survives for a
    rule with genuinely no vehicle condition — a pure perception notify is legitimate — but it
    can no longer fire for a rule that simply spells its condition differently.
    """
    if not _vehicle_conditions(rule):
        pytest.skip("no vehicle condition")
    eng = SceneEngine(cards, _unsatisfying_facts(rule), _Executor(), rules=(rule,))
    _satisfy(eng, rule, now=100.0)
    assert eng.observe(_tick(), now=100.0) == NO_ACTION


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_cooldown_is_never_bypassed(cards, rule):
    eng = SceneEngine(cards, _satisfying_facts(rule), _Executor(), rules=(rule,))
    _satisfy(eng, rule, now=100.0)
    first = eng.observe(_tick(100.0), now=100.0)
    # Asserted, not skipped. `pytest.skip("rule did not fire")` is how this property stopped
    # covering `animal_ahead` without saying so: `_satisfying_facts` knew only `Signal`, left
    # speed unset, and the rule REJECTed every time — so the sweep excused itself from testing
    # the cooldown of the only rule whose cooldown had never been tested. A rule the sweep
    # cannot drive to MATCH is a finding about the sweep, never a licence to test less, and it
    # also makes the three match-dependent properties above vacuous in the same silence.
    assert first != NO_ACTION, f"{rule.id}: the sweep could not make it fire"
    eng._pending = None          # remove the pending-dedup path so cooldown is what is tested
    assert eng.observe(_tick(101.0), now=101.0) == NO_ACTION


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_every_rule_yields_to_an_open_router_question(cards, rule):
    """At most one open question across both systems, whatever the rule."""
    eng = SceneEngine(cards, _satisfying_facts(rule), _Executor(), rules=(rule,))
    _satisfy(eng, rule, now=100.0)
    assert eng.observe(_tick(), now=100.0, question_open=True) == NO_ACTION


def test_every_rule_id_is_unique():
    ids = [r.id for r in RULES]
    assert len(ids) == len(set(ids))
