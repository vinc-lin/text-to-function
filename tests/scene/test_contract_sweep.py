"""Properties that must hold for EVERY rule, not just the one we shipped.

Modelled on tests/e2e/test_s8_contract_sweep.py, which is the strongest thing in the suite:
a property asserted over the whole set cannot be satisfied by a lucky special case.
"""
import re

import pytest

from scene.context import Observation
from scene.engine import SceneEngine, NO_ACTION
from scene.rules import RULES, Observed, Signal
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


def _satisfying_facts(rule):
    return _Facts({(c.entity, c.attribute): c.equals
                   for c in rule.when if isinstance(c, Signal)})


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
    The failure mode is indistinguishable from working correctly."""
    for cond in rule.when:
        if isinstance(cond, Signal):
            assert seeded_car.get_signal(cond.entity, cond.attribute) is not None, \
                f"{rule.id}: no such signal {cond.entity}/{cond.attribute}"


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_every_rule_speaks_chinese_and_no_identifiers(rule):
    text = speech_for(rule.intent)
    assert text, f"{rule.id} has no speech template"
    assert not re.search(r"[A-Za-z_{}\[\]<>]", text), text


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_a_rule_stays_silent_when_its_signal_condition_already_holds(cards, rule):
    """Never ask for what is already true."""
    signals = [c for c in rule.when if isinstance(c, Signal)]
    if not signals:
        pytest.skip("no signal condition")
    inverted = _Facts({(c.entity, c.attribute):
                       (not c.equals) if isinstance(c.equals, bool) else object()
                       for c in signals})
    eng = SceneEngine(cards, inverted, _Executor(), rules=(rule,))
    _satisfy(eng, rule, now=100.0)
    assert eng.observe(_tick(), now=100.0) == NO_ACTION


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_cooldown_is_never_bypassed(cards, rule):
    eng = SceneEngine(cards, _satisfying_facts(rule), _Executor(), rules=(rule,))
    _satisfy(eng, rule, now=100.0)
    first = eng.observe(_tick(100.0), now=100.0)
    if first == NO_ACTION:
        pytest.skip("rule did not fire")
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
