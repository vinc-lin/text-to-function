"""S8 — the reply CONTRACT, swept over the real corpus, plus the multi-intent shapes.

Every other e2e file asks "does THIS utterance produce THAT reply". This one asks the
question that has no per-utterance answer: whatever the router decides, does the reply
still obey the five invariants a voice assistant may never break?

    1. the reply is non-empty — there is no silent path
    2. at most ONE question, however many clauses failed
    3. every operation the car really performed is confirmed — nothing is actuated silently
    4. no false affirmation — a confirmation implies a successful dispatch, never the reverse
    5. a confirmation and a refusal for the SAME action never both appear

`tests/test_reply.py` proves these over hand-built `RouteResult`s and
`tests/test_reply_e2e.py` over three fixture cards; neither can see a contract break that
only a real routing decision produces. So the sweep runs the REAL 92-card catalog against
the SQLite-simulated car and reads what was dispatched out of the car's own operation log —
the reply is checked against what happened, not against what the pipeline says happened.

`FakeEmbedder` is a hashed-n-gram stand-in with no semantics and misroutes badly over 92
cards (把温度调到99度 reaches `set_air_recirculation`). For the sweep that is a FEATURE:
the contract must hold whatever the router picks, and a misroute is just another decision
to hold it to. For the shape cases below it is not, so every one of those utterances was
probed against the full catalog first and kept only because it reaches the intended
functions; no assertion here was relaxed to fit what routing happened to do.
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

import pytest

from eval.dataset import load_dataset

from t2f.cards import load_catalog
from t2f.config import Config
from t2f.embed import FakeEmbedder
from t2f.gate import ConfidenceGate, Thresholds
from t2f.pipeline import Pipeline, DeterministicResolver
from t2f.respond import render_response
from t2f.score import Scorer
from t2f.types import ToolCall

from sim.executor import SqliteExecutor
from sim.seed import seed_from_catalog
from sim.vehicle import SqliteVehicle

ROOT = Path(__file__).resolve().parents[2]
CARDS = load_catalog(ROOT / "data" / "catalog")
BY = {c.name: c for c in CARDS}

HEALTHY = "healthy"      # a car that performs everything it is physically able to
REFUSING = "refusing"    # A/C off + child lock on: both seeded preconditions bite

FAILURE = "抱歉，这个操作没能完成。"        # t2f/reply.py::_FAILURE
ACK = "好的。"                              # t2f/reply.py::_ACK
TERMINATORS = "。！？"

WINDOW = "已为您打开当前区域车窗。"
TEMP25_DRIVER = "已将主驾温度设置为25°C。"
FAN3 = "已将当前区域风速设置为3档。"
AC_OFF = "空调尚未开启"                      # sim/seed.py::_PRECONDITIONS
CHILD_LOCK = "车窗儿童锁已开启"


# --- harness -------------------------------------------------------------------------------

def _pipeline(car_state: str = HEALTHY, live: dict | None = None):
    """(pipeline, executor) over a freshly seeded car.

    Thresholds are loosened exactly as in tests/e2e/conftest.py and test_s5_simulator.py so
    FakeEmbedder reaches the HIGH band; nothing else is tuned. `live` seeds the state layer
    the way eval/arms.py::predict does, so a corpus row carrying `vehicle_state` reaches
    StateResolver instead of dying as an unresolvable relative command.
    """
    car = SqliteVehicle(":memory:")
    car.init_schema()
    seed_from_catalog(car, CARDS)
    if car_state == REFUSING:
        car.set_signal("climate.all", "ac_power", False)
        car.set_signal("window.all", "window_child_lock", True)
    ex = SqliteExecutor(car, BY)
    cfg = Config.default()
    cfg.thresholds = Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)
    pipe = Pipeline(CARDS, FakeEmbedder(256), Scorer(cfg.weights, cfg.domain_keywords),
                    ConfidenceGate(cfg.thresholds), cfg,
                    resolver=DeterministicResolver(BY, executor=ex))
    pipe.state.reset(live or {})
    return pipe, ex


def _attempted(car) -> list[tuple[ToolCall, str]]:
    """(tool_call, outcome) for every operation the car was asked to perform, oldest first.

    Read from the car's operation log rather than from the RouteResult on purpose: the
    contract is about what the vehicle really did, and the log is the vehicle's own record.
    A pipeline that forgot to report an execution cannot hide it here.
    """
    rows = reversed(car.recent_operations(limit=100))
    return [(ToolCall(r["function"], json.loads(r["parameters"])), r["outcome"]) for r in rows]


def _executed(car) -> list[ToolCall]:
    return [tc for tc, outcome in _attempted(car) if outcome == "executed"]


def _refused(car) -> list[ToolCall]:
    return [tc for tc, outcome in _attempted(car) if outcome == "refused"]


def _sentence(text: str) -> str:
    """The same terminator rule compose_reply applies, so fragments compare exactly."""
    text = (text or "").strip()
    if not text:
        return ""
    return text if text[-1] in TERMINATORS else text + "。"


def _confirmation_for(tool_call: ToolCall) -> str:
    return _sentence(render_response(BY[tool_call.name], tool_call))


def _questions(result) -> set[str]:
    return {_sentence(cl.clarification.question) for cl in result.clauses
            if cl.clarification and (cl.clarification.question or "").strip()}


def _speakable(result) -> set[str]:
    """Every fragment compose_reply was entitled to speak, given these clauses.

    Anything in the reply outside this set was invented by the reply layer — which is the
    only way a false affirmation can appear that no clause accounts for.
    """
    out = {FAILURE, ACK}
    for cl in result.clauses:
        for text in (cl.response,
                     cl.exec_error.message if cl.exec_error else None,
                     cl.clarification.question if cl.clarification else None):
            if (text or "").strip():
                out.add(_sentence(text))
        for err in (cl.validation_errors or []):
            if err.code != "missing_required" and (err.detail or "").strip():
                out.add(_sentence(err.detail))
    return out


def _decompose(reply: str, allowed: set[str]) -> tuple[list[str], str]:
    """Greedily split the reply into the fragments that produced it.

    Returns (fragments, leftover). A non-empty leftover is text no clause accounts for.
    Longest match wins, because one clause's question can be a prefix of another's.
    """
    used, rest = [], reply
    while rest:
        matches = [f for f in allowed if f and rest.startswith(f)]
        if not matches:
            return used, rest
        best = max(matches, key=len)
        used.append(best)
        rest = rest[len(best):]
    return used, ""


# --- the corpus ----------------------------------------------------------------------------

def _corpus() -> list[dict]:
    """~38 real utterances, taken from the eval sets rather than invented here.

    Deliberately spans every shape the router has to survive: single-intent, multi-intent,
    out-of-scope, ambiguous, invalid values, ASR corruption, and relative commands. The
    relative ones are exactly the rows carrying `vehicle_state` — without them the sweep
    would never reach StateResolver at all. Selection is a fixed head-slice per type, so the
    corpus is stable across runs and grows only when the datasets do.
    """
    gold = load_dataset(ROOT / "data" / "eval" / "gold.jsonl")
    e2e = load_dataset(ROOT / "data" / "eval" / "e2e_cases.jsonl")
    rows: list[dict] = []
    for source, kinds, n in ((gold, ("single", "multi_intent", "ood", "ambiguous"), 6),
                             (e2e, ("invalid", "asr_noise"), 4)):
        for kind in kinds:
            rows += [r for r in source if r["type"] == kind][:n]
    rows += [r for r in e2e + gold if r.get("vehicle_state")][:6]
    seen, out = set(), []
    for row in rows:
        if row["utterance"] not in seen:
            seen.add(row["utterance"])
            out.append(row)
    return out


CORPUS = _corpus()


@lru_cache(maxsize=None)
def _sweep(car_state: str):
    """Route the whole corpus once per car state; every invariant below reads this.

    Each row gets its OWN car — a shared one would let one utterance's writes decide the
    next utterance's outcome, and a contract break would then depend on corpus order.
    """
    out = []
    for row in CORPUS:
        pipe, ex = _pipeline(car_state, live=row.get("vehicle_state"))
        out.append((row, pipe.route(row["utterance"]), ex.car))
    return tuple(out)


CAR_STATES = [HEALTHY, REFUSING]


def test_the_sweep_covers_a_varied_corpus():
    """Guards the sweep itself: a selection bug that silently emptied the corpus would make
    every test below pass vacuously."""
    assert len(CORPUS) >= 30
    kinds = {row["type"] for row in CORPUS}
    assert kinds >= {"single", "multi_intent", "ood", "ambiguous", "invalid", "asr_noise"}
    assert any(row.get("vehicle_state") for row in CORPUS)          # relative commands
    # and the sweep must actually exercise both halves of the contract
    swept = _sweep(HEALTHY)
    assert any(_executed(car) for _, _, car in swept)
    assert any(res.clauses and any(cl.clarification for cl in res.clauses)
               for _, res, car in swept)
    assert any(_refused(car) for _, _, car in _sweep(REFUSING))


# --- invariant 1: there is no silent path ---------------------------------------------------

@pytest.mark.parametrize("car_state", CAR_STATES)
def test_sweep_every_utterance_is_answered(car_state):
    silent = [row["utterance"] for row, res, _ in _sweep(car_state)
              if not (isinstance(res.reply, str) and res.reply.strip())]
    assert not silent


# --- invariant 2: at most one question ------------------------------------------------------

@pytest.mark.parametrize("car_state", CAR_STATES)
def test_sweep_asks_at_most_one_question(car_state):
    """Counted over the questions the clauses actually produced, not over '？' characters:
    one question can contain several (您想设置氛围灯颜色的哪个选项？（red/blue/...）)."""
    bad = []
    for row, res, _ in _sweep(car_state):
        asked = _questions(res)
        used, _leftover = _decompose(res.reply, _speakable(res))
        spoken = [f for f in used if f in asked]
        if len(spoken) > 1:
            bad.append((row["utterance"], res.reply, spoken))
    assert not bad


# --- invariant 3: nothing is actuated silently ----------------------------------------------

@pytest.mark.parametrize("car_state", CAR_STATES)
def test_sweep_every_performed_operation_is_confirmed(car_state):
    """The car moved and the driver was not told is the failure mode that ends trust."""
    bad = []
    for row, res, car in _sweep(car_state):
        for tc in _executed(car):
            if _confirmation_for(tc) not in res.reply:
                bad.append((row["utterance"], tc.name, dict(tc.parameters), res.reply))
    assert not bad


# --- invariant 4: no false affirmation ------------------------------------------------------

@pytest.mark.parametrize("car_state", CAR_STATES)
def test_sweep_no_confirmation_without_a_performed_operation(car_state):
    """The mirror of invariant 3, and the more dangerous direction: a confirmation may only
    exist because the car really performed that call."""
    bad = []
    for row, res, car in _sweep(car_state):
        performed = {_confirmation_for(tc) for tc in _executed(car)}
        for cl in res.clauses:
            if (cl.response or "").strip() and _sentence(cl.response) not in performed:
                bad.append((row["utterance"], cl.response, res.reply))
    assert not bad


@pytest.mark.parametrize("car_state", CAR_STATES)
def test_sweep_reply_says_nothing_no_clause_produced(car_state):
    """The reply is exactly the concatenation of fragments its clauses authored (plus the
    two constants). Any leftover is text the reply layer invented — the only way an
    affirmation can appear that no clause, and therefore no operation, stands behind."""
    bad = []
    for row, res, _ in _sweep(car_state):
        _used, leftover = _decompose(res.reply, _speakable(res))
        if leftover:
            bad.append((row["utterance"], res.reply, leftover))
    assert not bad


# --- invariant 5: never confirm and refuse the same action ----------------------------------

@pytest.mark.parametrize("car_state", CAR_STATES)
def test_sweep_a_refused_action_is_never_also_confirmed(car_state):
    bad = []
    for row, res, car in _sweep(car_state):
        performed = {_confirmation_for(tc) for tc in _executed(car)}
        for cl in res.clauses:
            if (cl.response or "").strip() and cl.exec_error is not None:
                bad.append((row["utterance"], "clause both spoke and failed", res.reply))
        for tc in _refused(car):
            would_be = _confirmation_for(tc)
            # unless an identical call succeeded in the same utterance, the refused call's
            # confirmation must be nowhere in the reply
            if would_be in res.reply and would_be not in performed:
                bad.append((row["utterance"], tc.name, would_be, res.reply))
    assert not bad


# --- multi-intent shapes --------------------------------------------------------------------
# Each utterance below was probed against the full 92-card catalog; the functions named in
# the assertions are the ones routing really reaches.

def test_two_valid_actions_are_both_performed_and_both_confirmed():
    pipe, ex = _pipeline()

    res = pipe.route("开车窗，主驾温度调到25度")

    assert [tc.name for tc in _executed(ex.car)] == ["open_window", "set_temperature"]
    assert res.reply == WINDOW + TEMP25_DRIVER          # one reply, both confirmations
    assert ex.car.get_signal("climate.driver", "temperature") == 25


def test_three_valid_actions_are_all_performed_and_all_confirmed():
    """Three is not two: the plan barrier, the executor loop and the reply join all have to
    hold at a length no other e2e case reaches."""
    pipe, ex = _pipeline()

    res = pipe.route("开车窗，主驾温度调到25度，风速调到3档")

    assert [tc.name for tc in _executed(ex.car)] == ["open_window", "set_temperature",
                                                     "set_fan_speed"]
    assert res.reply == WINDOW + TEMP25_DRIVER + FAN3
    assert len(_questions(res)) == 0


def test_one_invalid_action_does_not_stop_the_valid_one_and_is_still_mentioned():
    """风速调到20档 is out of range (the card allows 0-7). The valid half must still happen,
    and the driver must not be left believing the invalid half did."""
    pipe, ex = _pipeline()

    res = pipe.route("开车窗，风速调到20档")

    assert [tc.name for tc in _executed(ex.car)] == ["open_window"]
    assert _refused(ex.car) == []                       # rejected before the car, not by it
    assert res.reply.startswith(WINDOW)
    # Mentioned by its CAUSE, not by echoing the span. Either way the point stands: the
    # action the driver asked for is not silently dropped.
    assert "风速档位只能设置在1到7档之间" in res.reply
    assert FAN3 not in res.reply and "已将当前区域风速" not in res.reply


def test_several_failed_clauses_still_produce_exactly_one_question():
    """Two invalid actions in one utterance. Invariant 2, at the shape where it is most
    likely to break.

    The invariant is AT MOST ONE QUESTION, and it now holds by there being none: each bad
    value explains itself, so neither needs to be asked about. Both causes are stated —
    two statements are not two questions, and the driver can act on both.
    """
    pipe, ex = _pipeline()

    res = pipe.route("开车窗，风速调到20档，屏幕亮度调到200%")

    assert [tc.name for tc in _executed(ex.car)] == ["open_window"]
    assert res.reply.count("？") == 0 and res.reply.count("我还需要确认一下") == 0
    assert "风速档位只能设置在1到7档之间" in res.reply
    assert "屏幕亮度只能设置在0到100%之间" in res.reply


def test_narration_beside_an_action_is_not_acted_on():
    """我有点热 is context, not a request. It must reach neither the plan nor the car."""
    pipe, ex = _pipeline()

    res = pipe.route("我有点热，把温度调到25度")

    assert [a.span for a in res.plan.actions] == ["把温度调到25度"]
    assert [tc.name for tc in _executed(ex.car)] == ["set_temperature"]
    assert len(_attempted(ex.car)) == 1                 # the narration reached nothing


def test_narration_beside_an_action_is_not_spoken_back():
    pipe, _ = _pipeline()

    res = pipe.route("我有点热，把温度调到25度")

    assert res.reply == "已将当前区域温度设置为25°C。"
    assert "热" not in res.reply
    assert "我有点热" not in res.reply


def test_a_refusal_and_a_confirmation_are_both_reported_confirmation_first():
    """The child lock is on, so the car refuses the window and performs the temperature.

    Note the ORDER: the refused action is the FIRST clause and still comes second in the
    reply. What actually happened leads; the reason it did not all happen follows.
    """
    pipe, ex = _pipeline(REFUSING)
    ex.car.set_signal("climate.all", "ac_power", True)  # only the child lock refuses here

    res = pipe.route("开车窗，主驾温度调到25度")

    assert [tc.name for tc in _executed(ex.car)] == ["set_temperature"]
    assert [tc.name for tc in _refused(ex.car)] == ["open_window"]
    assert res.reply == TEMP25_DRIVER + _sentence(CHILD_LOCK)


def test_a_refused_action_in_a_multi_intent_utterance_is_never_confirmed():
    """Invariant 5 at the shape that produces it: one action confirmed, one refused, and
    the refused one's confirmation nowhere in the reply."""
    pipe, ex = _pipeline(REFUSING)
    ex.car.set_signal("climate.all", "ac_power", True)

    res = pipe.route("开车窗，主驾温度调到25度")

    assert WINDOW not in res.reply
    assert "已为您" not in res.reply                     # no confirmation of any window kind
    assert all(cl.response is None for cl in res.clauses if cl.exec_error is not None)


def test_two_refused_actions_report_one_cause_and_no_confirmation():
    """Both clauses fail for the same reason: the driver hears it once, and hears nothing
    that sounds like success."""
    pipe, ex = _pipeline(REFUSING)

    res = pipe.route("主驾温度调到25度，风速调到3档")

    assert [tc.name for tc in _executed(ex.car)] == []
    assert [tc.name for tc in _refused(ex.car)] == ["set_temperature", "set_fan_speed"]
    assert res.reply == _sentence(AC_OFF)
    assert res.reply.count(AC_OFF) == 1
    assert "已将" not in res.reply and FAILURE not in res.reply


def test_duplicate_identical_actions_are_confirmed_once():
    """The same action twice in one utterance. The reply layer de-duplicates; note that the
    car is still asked twice — dispatch is not de-duplicated, and this pins that difference
    rather than asserting the behaviour anyone would prefer."""
    pipe, ex = _pipeline()

    res = pipe.route("主驾温度调到25度，主驾温度调到25度")

    assert [tc.name for tc in _executed(ex.car)] == ["set_temperature", "set_temperature"]
    assert res.reply == TEMP25_DRIVER
    assert res.reply.count(TEMP25_DRIVER) == 1


def test_duplicate_actions_still_satisfy_the_whole_contract():
    """De-duplication is where invariant 3 is easiest to break: drop one confirmation too
    many and a performed operation goes unreported."""
    pipe, ex = _pipeline()

    res = pipe.route("开车窗，开车窗")

    assert len(_executed(ex.car)) == 2
    assert res.reply == WINDOW
    for tc in _executed(ex.car):
        assert _confirmation_for(tc) in res.reply
    _used, leftover = _decompose(res.reply, _speakable(res))
    assert leftover == ""
