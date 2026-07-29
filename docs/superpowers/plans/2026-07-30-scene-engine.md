# Scene Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a proactive `scene/` subsystem that turns structured perception into at most a spoken question, where only the driver's explicit consent ever moves the car.

**Architecture:** a sibling package beside `t2f/` and `sim/`, meeting the router at exactly one seam — `executor.execute(ToolCall) -> ExecResult`. Deterministic rules decide the clear cases; an xgrammar-constrained LLM sees only near-misses and observations no rule anticipated; everything else is silence. `t2f/` routing is untouched. Full design: [`docs/superpowers/specs/2026-07-30-scene-engine-design.md`](../specs/2026-07-30-scene-engine-design.md).

**Tech Stack:** Python 3.10, pytest, SQLite (`sim/`), transformers + xgrammar for the optional fallback. No new dependencies.

---

## File structure

| File | Responsibility |
|---|---|
| `scene/context.py` | `Observation`, `SceneContext` — perception only, read-time staleness |
| `scene/facts.py` | `VehicleFacts` — read-only port over `SqliteVehicle` |
| `scene/speech.py` | intent → template; every sentence the subsystem can say |
| `scene/rules.py` | `Observed`, `Signal`, `Rule`, `evaluate()`, `RULES` |
| `scene/consent.py` | `PendingConsent`, the closed lexicon, `classify()` |
| `scene/engine.py` | `SceneEngine.observe()`, arbitration, consent resolution |
| `scene/llm.py` | schema, `SceneLLM` protocol, `FakeSceneLLM`, `TransformersSceneLLM` |
| `t2f/respond.py` | boolean humanisation — closes the last red case |
| `cli/session.py`, `cli/__main__.py`, `cli/render.py` | `/scene`, turn arbitration |
| `eval/scene_metrics.py`, `eval/run_scene_eval.py` | the four scene metrics |
| `data/eval/scenes.jsonl` | hand-authored gold |

Run all tests with `python3 -m pytest -q` from the repo root. The catalog path is relative, so **never `cd` elsewhere**.

---

## Task 1: Scene Context

**Files:**
- Create: `scene/__init__.py` (empty), `scene/context.py`
- Create: `tests/scene/__init__.py` (empty), `tests/scene/test_context.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/scene/test_context.py
"""What perception believes, and for how long."""
from scene.context import Observation, SceneContext


def _obs(**kw):
    base = dict(key="inside.rear_occupant", value="child", confidence=0.9,
                source="cabin_cam", at=100.0, ttl=300.0)
    base.update(kw)
    return Observation(**base)


def test_an_observation_is_readable_before_its_ttl():
    ctx = SceneContext()
    ctx.update(_obs())
    got = ctx.get("inside.rear_occupant", now=200.0)
    assert got is not None and got.value == "child" and got.confidence == 0.9


def test_an_observation_is_gone_after_its_ttl():
    """Staleness is read-time, not swept — nothing runs a clock on the SoC."""
    ctx = SceneContext()
    ctx.update(_obs(ttl=30.0))
    assert ctx.get("inside.rear_occupant", now=131.0) is None


def test_the_expiry_boundary_is_inclusive():
    """at + ttl is the last live instant; an off-by-one here silently shortens every ttl."""
    ctx = SceneContext()
    ctx.update(_obs(ttl=30.0))
    assert ctx.get("inside.rear_occupant", now=130.0) is not None


def test_a_newer_observation_replaces_an_older_one():
    ctx = SceneContext()
    ctx.update(_obs(at=100.0, value="child"))
    ctx.update(_obs(at=150.0, value="adult"))
    assert ctx.get("inside.rear_occupant", now=160.0).value == "adult"


def test_a_late_arriving_older_observation_does_not_win():
    """Frames can arrive out of order; the newest belief is the one with the newest
    timestamp, not the one that happened to be delivered last."""
    ctx = SceneContext()
    ctx.update(_obs(at=150.0, value="adult"))
    ctx.update(_obs(at=100.0, value="child"))
    assert ctx.get("inside.rear_occupant", now=160.0).value == "adult"


def test_live_omits_stale_keys():
    ctx = SceneContext()
    ctx.update(_obs(key="a", ttl=30.0))
    ctx.update(_obs(key="b", ttl=300.0))
    assert set(ctx.live(now=200.0)) == {"b"}


def test_an_unknown_key_reads_as_absent_not_an_error():
    assert SceneContext().get("nope", now=0.0) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/scene/test_context.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scene'`

- [ ] **Step 3: Implement**

```python
# scene/context.py
"""What perception currently believes, and for how long.

Holds perception ONLY. Vehicle state is read live from the car (scene/facts.py), because
copying it here would recreate the two-beliefs-about-one-actuator problem that signal-keyed
state was built to prevent — see sim/mapping.py's module docstring.

Staleness is evaluated at read time rather than by a sweeper: there is no clock to run on the
target SoC, and every test can state `now` instead of sleeping.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Observation:
    key: str            # "inside.rear_occupant"
    value: Any          # "child"
    confidence: float
    source: str         # "cabin_cam"
    at: float
    ttl: float

    def is_live(self, now: float) -> bool:
        """Inclusive at the boundary: at + ttl is the last live instant."""
        return now <= self.at + self.ttl


class SceneContext:
    def __init__(self):
        self._by_key: dict[str, Observation] = {}

    def update(self, obs: Observation) -> None:
        prev = self._by_key.get(obs.key)
        # Keep the newest by its OWN timestamp, not by arrival order: a delayed frame must
        # not overwrite a fresher belief about the same key.
        if prev is None or obs.at >= prev.at:
            self._by_key[obs.key] = obs

    def get(self, key: str, now: float) -> Optional[Observation]:
        obs = self._by_key.get(key)
        return obs if obs is not None and obs.is_live(now) else None

    def live(self, now: float) -> dict[str, Observation]:
        return {k: o for k, o in self._by_key.items() if o.is_live(now)}
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/scene/test_context.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add scene/__init__.py scene/context.py tests/scene/__init__.py tests/scene/test_context.py
git commit -m "feat(scene): Scene Context — perception only, read-time staleness"
```

---

## Task 2: Vehicle facts port and the speech table

**Files:**
- Create: `scene/facts.py`, `scene/speech.py`
- Create: `tests/scene/test_speech.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/scene/test_speech.py
"""Every sentence this subsystem can utter is in one table, and the table is checkable."""
import re

from scene.speech import SPEECH, speech_for


def test_every_intent_resolves_to_a_sentence():
    assert SPEECH and all(v.strip() for v in SPEECH.values())


def test_no_template_speaks_ascii_to_the_driver():
    """This repo has twice shipped developer text into the cabin (e433e32, 70bfeb5).
    A table makes the check trivial, so there is no excuse for a third time."""
    for intent, text in SPEECH.items():
        assert not re.search(r"[A-Za-z_]", text), f"{intent}: {text}"


def test_every_template_ends_in_a_terminator():
    assert all(t[-1] in "。！？" for t in SPEECH.values())


def test_an_unknown_intent_is_silence_not_a_traceback():
    """An unsayable intent must degrade to no speech; raising here would kill a session
    after the work of the turn is already done."""
    assert speech_for("no_such_intent") == ""
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/scene/test_speech.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scene.speech'`

- [ ] **Step 3: Implement both modules**

```python
# scene/speech.py
"""Every sentence this subsystem can utter.

A table, not generation. The model selects an intent from these keys; it never authors what
the car says. Two prior fixes in this repo (e433e32, 70bfeb5) exist because generated or
internal text reached the driver, and a 0.6B model writing unreviewed sentences into a cabin
is a larger claim than this subsystem makes.

The confirmation AFTER consent is deliberately not here: it comes from render_response on the
executed card, so a scene-initiated action confirms exactly as a driver-initiated one does.
"""
from __future__ import annotations

SPEECH: dict[str, str] = {
    "ask_rear_child_lock":   "后排有小孩，要打开儿童锁吗？",
    "notify_driver_fatigue": "您看起来有些疲劳，请注意休息。",
    "ack_declined":          "好的。",
}


def speech_for(intent: str) -> str:
    """'' for an unknown intent. An unsayable intent is silence, never an exception."""
    return SPEECH.get(intent, "")
```

```python
# scene/facts.py
"""Read-only access to the car, for rules that condition on vehicle state.

Deliberately read-only. The engine may ask the car what is true and may not write: every
write goes through executor.execute so it gets validation, preconditions, physical limits and
an operation-log entry. A second write path would be a second set of rules about what the car
allows, and the car is the authority on that.
"""
from __future__ import annotations
from typing import Any, Optional


class VehicleFacts:
    def __init__(self, car):
        self.car = car

    def signal(self, entity: str, attribute: str) -> Optional[Any]:
        return self.car.get_signal(entity, attribute)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/scene/test_speech.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add scene/facts.py scene/speech.py tests/scene/test_speech.py
git commit -m "feat(scene): read-only vehicle port and the speech table"
```

---

## Task 3: Rules and evaluation

**Files:**
- Create: `scene/rules.py`
- Create: `tests/scene/test_rules.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/scene/test_rules.py
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/scene/test_rules.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scene.rules'`

- [ ] **Step 3: Implement**

```python
# scene/rules.py
"""Declarative scene rules and their evaluation.

Conditions come in exactly two forms and there are no others. A closed vocabulary keeps every
rule inspectable and lets a contract test walk the whole set and assert properties over all of
it — which is what tests/scene/test_contract_sweep.py does.

Rules are dataclasses rather than YAML on purpose: one rule does not justify a loader, and the
shape is data-only, so a YAML front end is a later addition rather than a rewrite.
"""
from __future__ import annotations
from dataclasses import dataclass, field
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
    """Signal conditions first: cheapest and most definitive.

    An already-satisfied signal means there is nothing to ask about, and that answer beats any
    amount of perception uncertainty — so it is checked before confidence, or a weak detection
    against a settled car would spend a model call on a question already answered.
    """
    for cond in rule.when:
        if isinstance(cond, Signal) and facts.signal(cond.entity, cond.attribute) != cond.equals:
            return Verdict.REJECT

    near = False
    for cond in rule.when:
        if not isinstance(cond, Observed):
            continue
        obs = context.get(cond.key, now)
        # No observation, a different value, or one too weak to consider: the rule simply does
        # not apply. Absence of evidence is not ambiguity about it, and treating it as a
        # near-miss would have the fallback fire on an empty context.
        if obs is None or obs.value != cond.equals or obs.confidence < rule.floor:
            return Verdict.NOT_APPLICABLE
        if obs.confidence < rule.threshold or (now - obs.at) < rule.persist_for:
            near = True
    return Verdict.NEAR_MISS if near else Verdict.MATCH


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
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/scene/test_rules.py -q`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add scene/rules.py tests/scene/test_rules.py
git commit -m "feat(scene): declarative rules with a closed condition vocabulary"
```

---

## Task 4: Consent

**Files:**
- Create: `scene/consent.py`
- Create: `tests/scene/test_consent.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/scene/test_consent.py
"""Consent is exact membership, never substring. This file is the safety case for that."""
import pytest

from scene.consent import Answer, PendingConsent, classify
from t2f.types import ToolCall


@pytest.mark.parametrize("text", ["好", "好的", "好吧", "可以", "行", "嗯", "是的", "对",
                                  "没问题", "麻烦你了", "好的。", "好！"])
def test_affirmative_forms_are_consent(text):
    assert classify(text) is Answer.YES


@pytest.mark.parametrize("text", ["不用", "不要", "不必", "算了", "不了", "没事", "不需要", "不用。"])
def test_negative_forms_decline(text):
    assert classify(text) is Answer.NO


def test_a_sentence_containing_a_yes_is_not_a_yes():
    """好像有点热 contains 好. A substring test would read it as consent and lock the
    windows because the driver mentioned the temperature. This single assertion is the
    difference between consent and a guess."""
    assert classify("好像有点热") is Answer.NOT_AN_ANSWER


@pytest.mark.parametrize("text", ["把窗户关上", "后排太热了", "导航去公司", "开车窗"])
def test_a_command_is_never_consent(text):
    assert classify(text) is Answer.NOT_AN_ANSWER


def test_an_empty_utterance_is_not_an_answer():
    assert classify("") is Answer.NOT_AN_ANSWER


def test_pending_consent_expires_at_its_own_boundary():
    p = PendingConsent("s", ToolCall("f", {}), asked_at=100.0, expires_after=30.0)
    assert p.is_live(now=130.0)
    assert not p.is_live(now=130.1)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/scene/test_consent.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scene.consent'`

- [ ] **Step 3: Implement**

```python
# scene/consent.py
"""Was that a yes?

The engine asks a question the driver never invited, so whatever they say next may or may not
be an answer. Getting it wrong actuates the car on consent that was never given — the
proactive form of the OOD false-execution this project reports as 0.000.

So consent has exactly one shape: EXACT membership in a closed set, on the normalised
utterance. 好 is a yes; 好像有点热 is not, and a substring test would make it one. Anything
outside both sets drops the pending question and is routed as an ordinary command, which
loses an oblique yes like 开吧 — a cost accepted deliberately, and measured by
`scene_recall`.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from t2f.normalize import normalize
from t2f.types import ToolCall

_AFFIRM = frozenset({"好", "好的", "好吧", "可以", "行", "嗯", "嗯嗯",
                     "是", "是的", "对", "没问题", "麻烦你了"})
_DECLINE = frozenset({"不用", "不要", "不必", "算了", "不了", "没事", "不需要"})

# normalize() folds 。 to . and ！ to !, so terminators are stripped in their ASCII form.
_STRIP = " .,!?;:、"


class Answer(str, Enum):
    YES = "yes"
    NO = "no"
    NOT_AN_ANSWER = "not_an_answer"


@dataclass
class PendingConsent:
    scene: str
    proposal: ToolCall
    asked_at: float
    expires_after: float

    def is_live(self, now: float) -> bool:
        return now <= self.asked_at + self.expires_after


def classify(utterance: str) -> Answer:
    text = normalize(utterance or "").strip(_STRIP)
    if text in _AFFIRM:
        return Answer.YES
    if text in _DECLINE:
        return Answer.NO
    return Answer.NOT_AN_ANSWER
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3 -m pytest tests/scene/test_consent.py -q`
Expected: `27 passed`

- [ ] **Step 5: Commit**

```bash
git add scene/consent.py tests/scene/test_consent.py
git commit -m "feat(scene): consent is exact membership, never substring"
```

---

## Task 5: The engine — arbitration and consent resolution (rules only)

**Files:**
- Create: `scene/engine.py`
- Create: `tests/scene/test_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/scene/test_engine.py
"""One observation in, at most one sentence out — and the car moves only after a yes."""
import pytest

from scene.context import Observation
from scene.engine import SceneEngine, NO_ACTION
from scene.rules import RULES
from t2f.cards import load_catalog
from t2f.types import ExecResult, ToolCall


class FakeFacts:
    def __init__(self, lock=False):
        self.lock = lock

    def signal(self, entity, attribute):
        return self.lock if (entity, attribute) == ("window.all", "window_child_lock") else None


class RecordingExecutor:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or ExecResult(ok=True)

    def execute(self, tool_call):
        self.calls.append(tool_call)
        return self.result


@pytest.fixture(scope="module")
def cards():
    return {c.name: c for c in load_catalog("data/catalog")}


def _engine(cards, facts=None, executor=None):
    return SceneEngine(cards_by_name=cards, facts=facts or FakeFacts(),
                       executor=executor or RecordingExecutor(), rules=RULES)


def _child(confidence=0.9, at=100.0):
    return Observation("inside.rear_occupant", "child", confidence, "cabin_cam", at, ttl=300.0)


def test_a_matching_rule_asks_and_touches_nothing(cards):
    ex = RecordingExecutor()
    out = _engine(cards, executor=ex).observe(_child(), now=100.0)
    assert out.kind == "ask" and out.source == "rule"
    assert out.speech == "后排有小孩，要打开儿童锁吗？"
    assert out.proposal == ToolCall("set_window_child_lock", {"enabled": True})
    assert ex.calls == [], "a rule match must never reach the car on its own"


def test_an_already_locked_car_says_nothing(cards):
    out = _engine(cards, facts=FakeFacts(lock=True)).observe(_child(), now=100.0)
    assert out == NO_ACTION


def test_consent_executes_the_proposal(cards):
    ex = RecordingExecutor()
    eng = _engine(cards, executor=ex)
    eng.observe(_child(), now=100.0)
    res = eng.resolve("好", now=105.0)
    assert res.answered and res.executed
    assert ex.calls == [ToolCall("set_window_child_lock", {"enabled": True})]
    assert res.speech == "已为您打开车窗儿童锁。"


def test_declining_executes_nothing_and_acknowledges(cards):
    ex = RecordingExecutor()
    eng = _engine(cards, executor=ex)
    eng.observe(_child(), now=100.0)
    res = eng.resolve("不用", now=105.0)
    assert res.answered and not res.executed and ex.calls == []
    assert res.speech == "好的。"


def test_a_command_drops_the_question_and_is_not_an_answer(cards):
    """The driver ignored us and said something else. The pending question is abandoned and
    the caller routes the utterance normally — it must never be read as consent."""
    ex = RecordingExecutor()
    eng = _engine(cards, executor=ex)
    eng.observe(_child(), now=100.0)
    res = eng.resolve("把窗户关上", now=105.0)
    assert not res.answered and ex.calls == []
    assert eng.pending(now=106.0) is None


def test_an_expired_question_cannot_be_answered(cards):
    ex = RecordingExecutor()
    eng = _engine(cards, executor=ex)
    eng.observe(_child(), now=100.0)
    res = eng.resolve("好", now=100.0 + eng.consent_ttl + 1.0)
    assert not res.answered and ex.calls == []


def test_the_vehicle_refusal_is_spoken_with_its_cause(cards):
    ex = RecordingExecutor(ExecResult(ok=False, error="device_unavailable", detail="车窗控制模块离线"))
    eng = _engine(cards, executor=ex)
    eng.observe(_child(), now=100.0)
    res = eng.resolve("好", now=105.0)
    assert res.answered and not res.executed
    assert res.speech == "车窗控制模块离线。"


def test_a_rule_in_cooldown_does_not_speak_twice(cards):
    eng = _engine(cards)
    assert eng.observe(_child(at=100.0), now=100.0).kind == "ask"
    eng.resolve("不用", now=101.0)
    assert eng.observe(_child(at=102.0), now=102.0) == NO_ACTION


def test_the_same_question_is_not_asked_while_one_is_pending(cards):
    eng = _engine(cards)
    assert eng.observe(_child(at=100.0), now=100.0).kind == "ask"
    assert eng.observe(_child(at=101.0), now=101.0) == NO_ACTION


def test_a_scene_stays_silent_while_the_router_holds_a_question(cards):
    """At most one open question across both systems, or 好 becomes ambiguous about which
    one it answers — and the whole consent design rests on 好 being unambiguous."""
    out = _engine(cards).observe(_child(), now=100.0, question_open=True)
    assert out == NO_ACTION


def test_an_engine_exception_degrades_to_silence(cards):
    class Exploding:
        def signal(self, *a):
            raise RuntimeError("camera bus fell over")
    out = _engine(cards, facts=Exploding()).observe(_child(), now=100.0)
    assert out == NO_ACTION
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/scene/test_engine.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scene.engine'`

- [ ] **Step 3: Implement**

```python
# scene/engine.py
"""Perception in, at most one sentence out.

The engine may speak. It may not act. The car moves in exactly one place — `resolve()`, after
an explicit yes — and a contract test asserts that no rule match ever produces a ToolCall on
its own.

Everything degrades to silence: an exception, a missing model, a proposal that fails
validation. A system nobody asked to speak has silence as its safe default.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from t2f.respond import render_response
from t2f.types import ToolCall
from t2f.validate import validate_tool_call

from .consent import Answer, PendingConsent, classify
from .context import SceneContext
from .rules import RULES, Verdict, evaluate
from .speech import speech_for

CONSENT_TTL = 30.0
_FAILURE = "抱歉，这个操作没能完成。"


@dataclass
class SceneOutcome:
    kind: str                        # "notify" | "ask" | "no_action"
    scene: str
    speech: str
    proposal: Optional[ToolCall]
    source: str                      # "rule" | "llm"
    reason: str                      # diagnostic, never spoken


@dataclass
class ConsentResult:
    answered: bool                   # False -> caller routes the utterance normally
    speech: str = ""
    executed: bool = False
    tool_call: Optional[ToolCall] = None


NO_ACTION = SceneOutcome("no_action", "", "", None, "rule", "")


def _sentence(text: str) -> str:
    return text if (not text or text[-1] in "。！？") else text + "。"


class SceneEngine:
    def __init__(self, cards_by_name, facts, executor, rules=RULES, llm=None,
                 consent_ttl: float = CONSENT_TTL):
        self.cards = cards_by_name
        self.facts = facts
        self.executor = executor
        self.rules = tuple(rules)
        self.llm = llm
        self.consent_ttl = consent_ttl
        self.context = SceneContext()
        self._pending: Optional[PendingConsent] = None
        self._last_spoken: dict[str, float] = {}

    # --- perception -------------------------------------------------------------------
    def observe(self, obs, now: float, *, question_open: bool = False) -> SceneOutcome:
        """Never raises. A traceback here would kill a session after the work is done."""
        try:
            self.context.update(obs)
            return self._evaluate(now, question_open=question_open)
        except Exception:
            return NO_ACTION

    def _evaluate(self, now: float, *, question_open: bool) -> SceneOutcome:
        verdicts = [(r, evaluate(r, self.context, self.facts, now)) for r in self.rules]
        matched = [r for r, v in verdicts if v is Verdict.MATCH and self._speakable(r, now)]
        if matched:
            # Highest priority wins; ties break by declaration order, so the outcome does not
            # depend on dict ordering or on a clock.
            best = sorted(matched, key=lambda r: (-r.priority, self.rules.index(r)))[0]
            return self._fire(best, now, question_open=question_open)
        return NO_ACTION

    def _speakable(self, rule, now: float) -> bool:
        last = self._last_spoken.get(rule.id)
        if last is not None and now - last < rule.cooldown:
            return False
        pending = self.pending(now)
        # Do not re-ask a question we are already waiting on an answer to.
        return not (pending is not None and pending.scene == rule.id)

    def _fire(self, rule, now: float, *, question_open: bool) -> SceneOutcome:
        if rule.proposes is None:
            self._last_spoken[rule.id] = now
            return SceneOutcome("notify", rule.id, _sentence(speech_for(rule.intent)),
                                None, "rule", "rule matched")
        # Validate BEFORE asking. Discovering after the driver says 好 that the call was never
        # usable is the proactive form of a falsely-affirmative reply.
        tc, _ = validate_tool_call(rule.proposes.name, dict(rule.proposes.parameters),
                                   self.cards, [rule.proposes.name])
        if tc is None:
            return NO_ACTION
        if question_open:
            # At most one open question across both systems, or 好 becomes ambiguous.
            return NO_ACTION
        self._last_spoken[rule.id] = now
        self._pending = PendingConsent(rule.id, tc, asked_at=now, expires_after=self.consent_ttl)
        return SceneOutcome("ask", rule.id, _sentence(speech_for(rule.intent)),
                            tc, "rule", "rule matched")

    # --- consent ----------------------------------------------------------------------
    def pending(self, now: float) -> Optional[PendingConsent]:
        if self._pending is not None and not self._pending.is_live(now):
            self._pending = None
        return self._pending

    def resolve(self, utterance: str, now: float) -> ConsentResult:
        pending = self.pending(now)
        if pending is None:
            return ConsentResult(answered=False)
        answer = classify(utterance)
        if answer is Answer.NOT_AN_ANSWER:
            # Abandon the question rather than hold it open: a driver who said something else
            # has moved on, and a stale question would make the NEXT 好 ambiguous.
            self._pending = None
            return ConsentResult(answered=False)
        self._pending = None
        if answer is Answer.NO:
            return ConsentResult(answered=True, speech=speech_for("ack_declined"))
        return self._execute(pending.proposal)

    def _execute(self, proposal: ToolCall) -> ConsentResult:
        """Consent authorises an action, not an outcome.

        The car may have changed between the question and the answer, so the call is
        re-validated and re-dispatched here rather than trusted from ask time.
        """
        tc, _ = validate_tool_call(proposal.name, dict(proposal.parameters),
                                   self.cards, [proposal.name])
        if tc is None:
            return ConsentResult(answered=True, speech=_FAILURE)
        res = self.executor.execute(tc)
        if not res.ok:
            return ConsentResult(answered=True, speech=_sentence(res.detail or "") or _FAILURE,
                                 tool_call=tc)
        return ConsentResult(answered=True, executed=True, tool_call=tc,
                             speech=_sentence(render_response(self.cards[tc.name], tc)))
```

- [ ] **Step 4: Run**

Run: `python3 -m pytest tests/scene/test_engine.py -q`
Expected: FAIL on `test_consent_executes_the_proposal` — it asserts `已为您打开车窗儿童锁。` and the card still says `已为您调整车窗儿童锁状态。`. Every other test passes. **This is expected**; Task 6 fixes the template. Confirm the failure is exactly that one assertion and no other.

- [ ] **Step 5: Commit**

```bash
git add scene/engine.py tests/scene/test_engine.py
git commit -m "feat(scene): engine, arbitration, and consent-before-action"
```

---

## Task 6: Boolean confirmations state their direction

Closes `test_s4a_07_boolean_action_states_on_or_off`, the repo's last red case.

**Files:**
- Modify: `t2f/respond.py:23-31`
- Modify: `data/catalog/*.yaml` — the 38 boolean templates matching `已为您调整…状态。`
- Modify: `tests/e2e/test_s4a_confirmation.py:43-52` — remove the xfail marker
- Create: `tests/test_respond_boolean.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_respond_boolean.py
"""A confirmation that does not say which way it went is not a confirmation."""
import pytest

from t2f.cards import load_catalog
from t2f.respond import render_response
from t2f.types import ToolCall

# Templates where 打开/关闭 does not read naturally. Listed explicitly so the set cannot grow
# by accident: a new ambiguous card fails test_every_other_boolean_card_states_direction.
KNOWN_AMBIGUOUS = {"spray_washer"}


@pytest.fixture(scope="module")
def cards():
    return {c.name: c for c in load_catalog("data/catalog")}


def _bool_param(card):
    return next((p for p in card.params if p.type == "boolean"), None)


def test_the_child_lock_says_which_way_it_went(cards):
    card = cards["set_window_child_lock"]
    on = render_response(card, ToolCall(card.name, {"enabled": True}))
    off = render_response(card, ToolCall(card.name, {"enabled": False}))
    assert on == "已为您打开车窗儿童锁。"
    assert off == "已为您关闭车窗儿童锁。"


def test_a_fold_card_folds_rather_than_opens(cards):
    """折叠/展开, not 打开/关闭 — 已为您打开后视镜折叠 is not Chinese anyone speaks. The verb
    comes from the function name, the same way sim/mapping.py derives a signal attribute."""
    card = cards["fold_mirror"]
    on = render_response(card, ToolCall(card.name, {"enabled": True}))
    assert "折叠" in on and "打开" not in on


def test_an_is_off_parameter_inverts(cards):
    """turn_off_screen{is_off: true} turned the screen OFF. Reading the raw boolean would
    announce the opposite of what happened."""
    card = cards["turn_off_screen"]
    assert "关闭" in render_response(card, ToolCall(card.name, {"is_off": True}))
    assert "打开" in render_response(card, ToolCall(card.name, {"is_off": False}))


def test_every_other_boolean_card_states_direction(cards):
    for card in cards.values():
        p = _bool_param(card)
        if p is None or card.name in KNOWN_AMBIGUOUS:
            continue
        on = render_response(card, ToolCall(card.name, {p.name: True}))
        off = render_response(card, ToolCall(card.name, {p.name: False}))
        assert on != off, f"{card.name} says the same thing both ways: {on}"


def test_a_non_boolean_card_is_unchanged(cards):
    card = cards["set_temperature"]
    out = render_response(card, ToolCall(card.name, {"temperature": 25.0, "position": "driver"}))
    assert out == "已将主驾温度设置为25°C。"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_respond_boolean.py -q`
Expected: FAIL — `test_the_child_lock_says_which_way_it_went` gets `已为您调整车窗儿童锁状态。`

- [ ] **Step 3: Implement the mechanism**

Replace `render_response` in `t2f/respond.py`:

```python
# state words, chosen by the function's own verb. fold_mirror is not "opened".
_STATE_WORDS = {"fold": ("折叠", "展开")}
_STATE_DEFAULT = ("打开", "关闭")
# is_off=True means the thing is OFF: reading the raw boolean would announce the opposite.
_INVERTED = {"is_off"}


def _state_word(card: FunctionCard, tool_call: ToolCall) -> str:
    """打开/关闭 (or 折叠/展开) for a card whose primary parameter is boolean, else ''."""
    spec = next((p for p in card.params if p.type == "boolean"), None)
    if spec is None or spec.name not in tool_call.parameters:
        return ""
    value = bool(tool_call.parameters[spec.name])
    if spec.name in _INVERTED:
        value = not value
    verb = card.name.split("_")[0]
    on, off = _STATE_WORDS.get(verb, _STATE_DEFAULT)
    return on if value else off


def render_response(card: FunctionCard, tool_call: ToolCall) -> str:
    if not card.response_template:
        return f"已执行{card.name}。"
    params = {k: _fmt_num(v) for k, v in tool_call.parameters.items()}
    if "position" in params:
        params["position"] = _POSITION_CN.get(params["position"], params["position"])
    elif card.param("position"):
        params.setdefault("position", "当前区域")
    # `state` is injected for every card; _SafeDict means a template that does not use it is
    # unaffected, so only the boolean templates below had to change.
    params["state"] = _state_word(card, tool_call)
    return card.response_template.format_map(_SafeDict(params))
```

- [ ] **Step 4: Rewrite the 38 boolean templates**

Every boolean card's template matches `已为您调整<X>状态。`. Rewrite each to `已为您{state}<X>。`, dropping the 调整/状态 scaffolding. `spray_washer` (`已为您喷洒玻璃水。`) is left alone — it is momentary, not a state.

Run this to see the exact list and current values:

```bash
python3 -c "
from t2f.cards import load_catalog
for c in load_catalog('data/catalog'):
    if any(p.type=='boolean' for p in c.params) and '{state}' not in c.response_template:
        print(f'{c.name:26s} {c.response_template}')
"
```

Worked examples — apply the same transformation to all 38:

```yaml
# data/catalog/window.yaml
    response_template: "已为您{state}车窗儿童锁。"          # was 已为您调整车窗儿童锁状态。
    response_template: "已为您{state}{position}车窗。"      # was 已为您调整{position}车窗状态。
    response_template: "已为您{state}天窗。"                # was 已为您调整天窗状态。
# data/catalog/misc.yaml
    response_template: "已为您{state}{position}后视镜折叠。" # was 已为您调整{position}后视镜折叠状态。
# data/catalog/display.yaml
    response_template: "已为您{state}屏幕显示。"            # was 已为您调整屏幕显示状态。
```

For `fold_mirror` and `fold_rear_seat` the state word is 折叠/展开, so `已为您{state}{position}后视镜折叠。` would read "已为您折叠…后视镜折叠". Use `已为您{state}{position}后视镜。` and `已为您{state}{position}后排座椅。` instead.

- [ ] **Step 5: Un-red the xfail**

In `tests/e2e/test_s4a_confirmation.py`, delete the `@pytest.mark.xfail(...)` decorator above `test_s4a_07_boolean_action_states_on_or_off` (lines 43-46), leaving the test body intact.

- [ ] **Step 6: Run the full suite**

Run: `python3 -m pytest -q`
Expected: all pass, **0 xfailed** (was 1). Some e2e cases assert exact replies for boolean cards and will need their expected strings updated to the new wording — update the *expectation*, never the implementation, and only where the new sentence is genuinely correct.

- [ ] **Step 7: Commit**

```bash
git add t2f/respond.py data/catalog tests/test_respond_boolean.py tests/e2e/test_s4a_confirmation.py
git commit -m "fix: a boolean confirmation states which way it went

Closes the last red case: opening and closing produced byte-identical replies."
```

---

## Task 7: The LLM fallback

**Files:**
- Create: `scene/llm.py`
- Modify: `scene/engine.py` — the no-match branch
- Create: `tests/scene/test_llm.py`, extend `tests/scene/test_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/scene/test_llm.py
"""The fallback picks from what exists. It cannot invent a scene, a sentence, or an action."""
import pytest

from scene.llm import FakeSceneLLM, scene_decision_schema
from scene.rules import RULES
from scene.speech import SPEECH


def test_the_schema_offers_no_execute_decision():
    """The model cannot ask for the car to move. The most it can do is propose a question,
    and that still needs consent."""
    schema = scene_decision_schema(RULES, SPEECH)
    assert set(schema["properties"]["decision"]["enum"]) == {"notify", "ask", "no_action"}


def test_the_scene_enum_is_the_rule_ids_plus_unmatched():
    schema = scene_decision_schema(RULES, SPEECH)
    assert set(schema["properties"]["scene"]["enum"]) == {r.id for r in RULES} | {"unmatched"}


def test_the_intent_enum_is_exactly_the_speech_table():
    """If these drift apart the model can pick an intent that resolves to silence."""
    schema = scene_decision_schema(RULES, SPEECH)
    assert set(schema["properties"]["reply_intent"]["enum"]) == set(SPEECH)


def test_every_field_is_required_and_nothing_else_is_allowed():
    schema = scene_decision_schema(RULES, SPEECH)
    assert set(schema["required"]) == {"decision", "scene", "reason", "reply_intent"}
    assert schema["additionalProperties"] is False


def test_the_fake_returns_its_script():
    llm = FakeSceneLLM([{"decision": "notify", "scene": "unmatched",
                         "reason": "r", "reply_intent": "notify_driver_fatigue"}])
    assert llm.decide({}, RULES, SPEECH)["decision"] == "notify"


def test_the_fake_returns_none_when_the_script_runs_out():
    """Exhausted script means no decision, which the engine must read as silence."""
    assert FakeSceneLLM([]).decide({}, RULES, SPEECH) is None
```

Append to `tests/scene/test_engine.py`:

```python
from scene.llm import FakeSceneLLM


def _obs(key, value, confidence=0.9, at=100.0):
    return Observation(key, value, confidence, "cabin_cam", at, ttl=300.0)


def test_a_near_miss_reaches_the_fallback(cards):
    llm = FakeSceneLLM([{"decision": "ask", "scene": "rear_child_window_lock",
                         "reason": "低置信但语境明确", "reply_intent": "ask_rear_child_lock"}])
    eng = SceneEngine(cards_by_name=cards, facts=FakeFacts(), executor=RecordingExecutor(),
                      rules=RULES, llm=llm)
    out = eng.observe(_child(confidence=0.62), now=100.0)
    assert out.kind == "ask" and out.source == "llm"
    assert out.proposal == ToolCall("set_window_child_lock", {"enabled": True})


def test_an_unconsumed_observation_reaches_the_fallback(cards):
    """Perception reported something no rule anticipated. Silence would be the alternative."""
    llm = FakeSceneLLM([{"decision": "notify", "scene": "unmatched",
                         "reason": "驾驶员疲劳", "reply_intent": "notify_driver_fatigue"}])
    eng = SceneEngine(cards_by_name=cards, facts=FakeFacts(), executor=RecordingExecutor(),
                      rules=RULES, llm=llm)
    out = eng.observe(_obs("inside.driver_attention", "drowsy"), now=100.0)
    assert out.kind == "notify" and out.speech == "您看起来有些疲劳，请注意休息。"


def test_an_unmatched_scene_may_not_ask(cards):
    """An ask needs a proposal, and an unmatched scene has none — so there would be nothing
    for consent to authorise. It degrades to silence rather than asking an empty question."""
    llm = FakeSceneLLM([{"decision": "ask", "scene": "unmatched",
                         "reason": "x", "reply_intent": "ask_rear_child_lock"}])
    eng = SceneEngine(cards_by_name=cards, facts=FakeFacts(), executor=RecordingExecutor(),
                      rules=RULES, llm=llm)
    assert eng.observe(_obs("inside.driver_attention", "drowsy"), now=100.0) == NO_ACTION


def test_a_below_floor_observation_never_reaches_the_fallback(cards):
    llm = FakeSceneLLM([{"decision": "ask", "scene": "rear_child_window_lock",
                         "reason": "x", "reply_intent": "ask_rear_child_lock"}])
    eng = SceneEngine(cards_by_name=cards, facts=FakeFacts(), executor=RecordingExecutor(),
                      rules=RULES, llm=llm)
    assert eng.observe(_child(confidence=0.30), now=100.0) == NO_ACTION
    assert llm.calls == 0


def test_a_clear_rule_match_never_consults_the_model(cards):
    """Arbitration order is what enforces 'the LLM never overrides the rules'."""
    llm = FakeSceneLLM([{"decision": "no_action", "scene": "unmatched",
                         "reason": "x", "reply_intent": "ack_declined"}])
    eng = SceneEngine(cards_by_name=cards, facts=FakeFacts(), executor=RecordingExecutor(),
                      rules=RULES, llm=llm)
    assert eng.observe(_child(confidence=0.95), now=100.0).source == "rule"
    assert llm.calls == 0


def test_the_fallback_budget_holds(cards):
    llm = FakeSceneLLM([{"decision": "notify", "scene": "unmatched", "reason": "x",
                         "reply_intent": "notify_driver_fatigue"}] * 5)
    eng = SceneEngine(cards_by_name=cards, facts=FakeFacts(), executor=RecordingExecutor(),
                      rules=RULES, llm=llm)
    eng.observe(_obs("inside.driver_attention", "drowsy", at=100.0), now=100.0)
    eng.observe(_obs("inside.driver_attention", "drowsy", at=101.0), now=101.0)
    assert llm.calls == 1, "one call per FALLBACK_COOLDOWN window"


def test_a_model_that_raises_degrades_to_silence(cards):
    class Exploding:
        calls = 0
        def decide(self, *a, **k):
            raise RuntimeError("decode failed")
    eng = SceneEngine(cards_by_name=cards, facts=FakeFacts(), executor=RecordingExecutor(),
                      rules=RULES, llm=Exploding())
    assert eng.observe(_child(confidence=0.62), now=100.0) == NO_ACTION
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/scene/test_llm.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scene.llm'`

- [ ] **Step 3: Implement `scene/llm.py`**

```python
# scene/llm.py
"""The constrained fallback: it decides, it does not act, and it does not write sentences.

Three properties come from the schema rather than from a check that could be forgotten. The
decision vocabulary contains no `execute`. `scene` and `reply_intent` are enums, so the model
selects from what exists. And `no_action` is a legal answer — a constrained decoder always
emits something, which is exactly why REJECT_NAME exists on the tool-call path
(t2f/llm/schema.py:34-41); without a legal way to decline, a model declines by picking
something, which is the mechanism behind the 99°→16° substitution.
"""
from __future__ import annotations
import json
from typing import Optional

UNMATCHED = "unmatched"


def scene_decision_schema(rules, speech: dict) -> dict:
    return {
        "type": "object",
        "properties": {
            "decision": {"enum": ["notify", "ask", "no_action"]},
            "scene": {"enum": [r.id for r in rules] + [UNMATCHED]},
            "reason": {"type": "string"},
            "reply_intent": {"enum": sorted(speech)},
        },
        "required": ["decision", "scene", "reason", "reply_intent"],
        "additionalProperties": False,
    }


def build_scene_prompt(snapshot: dict, rules, speech: dict) -> list[dict]:
    """Only what is live right now, never accumulated history.

    'Do not continuously append raw vision text to the LLM prompt' is enforced structurally:
    this function receives a snapshot, so there is nowhere for history to accumulate.
    """
    known = "\n".join(f"- {r.id}: {r.description}" for r in rules)
    return [
        {"role": "system",
         "content": "你是车内场景助手。只能返回 JSON 决策，不能执行任何车辆功能。"
                    f"已知场景：\n{known}\n无法归类时 scene 填 \"{UNMATCHED}\"。"
                    "不确定时选择 no_action。"},
        {"role": "user", "content": json.dumps(snapshot, ensure_ascii=False)},
    ]


class FakeSceneLLM:
    """Scripted decisions, and a call counter so budget tests can assert on it."""

    def __init__(self, script: list):
        self._script = list(script)
        self.calls = 0

    def decide(self, snapshot, rules, speech) -> Optional[dict]:
        self.calls += 1
        return self._script.pop(0) if self._script else None


class TransformersSceneLLM:
    """Qwen3-0.6B under an xgrammar constraint, mirroring TransformersXGrammarClient."""

    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B", max_new_tokens: int = 96,
                 device: str | None = None):
        import sys as _sys
        _sys.modules.setdefault("torchvision", None)
        import torch
        import xgrammar as xgr
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._torch, self._xgr = torch, xgr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        info = xgr.TokenizerInfo.from_huggingface(self.tok, vocab_size=self.model.config.vocab_size)
        self.compiler = xgr.GrammarCompiler(info)

    def decide(self, snapshot, rules, speech) -> Optional[dict]:
        torch = self._torch
        schema = scene_decision_schema(rules, speech)
        messages = build_scene_prompt(snapshot, rules, speech)
        prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        compiled = self.compiler.compile_json_schema(json.dumps(schema))
        processor = self._xgr.contrib.hf.LogitsProcessor(compiled)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, logits_processor=[processor],
                                      pad_token_id=self.tok.eos_token_id)
        raw = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            return json.loads(raw.strip())
        except Exception:
            return None      # unparseable output is silence, not a crash
```

- [ ] **Step 4: Wire the fallback into `scene/engine.py`**

Add near the top:

```python
from .llm import UNMATCHED, scene_decision_schema
from .speech import SPEECH

FALLBACK_COOLDOWN = 30.0
```

Add `self._last_fallback: float | None = None` to `__init__`, then replace the final
`return NO_ACTION` of `_evaluate` with:

```python
        return self._fallback(verdicts, now, question_open=question_open)
```

and add these methods:

```python
    def _fallback(self, verdicts, now: float, *, question_open: bool) -> SceneOutcome:
        """Reached only when no rule matched. Near-misses and observations no rule mentions."""
        if self.llm is None:
            return NO_ACTION
        near = [r for r, v in verdicts if v is Verdict.NEAR_MISS]
        mentioned = {k for r in self.rules for k in r.observed_keys}
        live = self.context.live(now)
        unconsumed = [k for k in live if k not in mentioned]
        if not near and not unconsumed:
            return NO_ACTION
        if self._last_fallback is not None and now - self._last_fallback < FALLBACK_COOLDOWN:
            return NO_ACTION
        self._last_fallback = now
        snapshot = {k: {"value": o.value, "confidence": o.confidence, "source": o.source}
                    for k, o in live.items()}
        try:
            decision = self.llm.decide(snapshot, self.rules, SPEECH)
        except Exception:
            return NO_ACTION
        return self._from_decision(decision, now, question_open=question_open)

    def _from_decision(self, decision, now: float, *, question_open: bool) -> SceneOutcome:
        if not isinstance(decision, dict):
            return NO_ACTION
        kind = decision.get("decision")
        scene = decision.get("scene", UNMATCHED)
        speech = _sentence(speech_for(decision.get("reply_intent", "")))
        reason = str(decision.get("reason", ""))[:200]
        if kind == "notify" and speech:
            return SceneOutcome("notify", scene, speech, None, "llm", reason)
        if kind == "ask":
            # An ask is legal ONLY when it names a real rule carrying a proposal: an unmatched
            # scene has nothing for consent to authorise, so asking would open a question no
            # answer could act on.
            rule = next((r for r in self.rules if r.id == scene and r.proposes), None)
            if rule is None or question_open or not speech:
                return NO_ACTION
            return self._fire(rule, now, question_open=question_open)
        return NO_ACTION
```

`_fire` re-derives the speech from the rule's own intent, which is correct: the model chose
*whether* to ask, and the rule owns *what is said*.

- [ ] **Step 5: Run**

Run: `python3 -m pytest tests/scene -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scene/llm.py scene/engine.py tests/scene/test_llm.py tests/scene/test_engine.py
git commit -m "feat(scene): constrained LLM fallback for near-misses and unknown observations"
```

---

## Task 8: Session and CLI integration

**Files:**
- Modify: `cli/session.py` — build a `SceneEngine`, add `observe()`, consult consent in `handle()`
- Modify: `cli/__main__.py` — the `/scene` command
- Modify: `cli/render.py` — render a scene turn
- Create: `tests/cli/test_scene_session.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/cli/test_scene_session.py
"""The session arbitrates two speakers: the driver's turn always finishes first."""
import pytest

from cli.session import Session


@pytest.fixture
def session():
    return Session.build(fake=True, llm=False, gate="permissive")


def test_a_scene_event_produces_a_question_and_moves_nothing(session):
    turn = session.observe("rear_occupant", "child", confidence=0.9)
    assert turn.reply == "后排有小孩，要打开儿童锁吗？"
    assert session.changed_signals() == []


def test_consent_moves_the_signal(session):
    session.observe("rear_occupant", "child", confidence=0.9)
    turn = session.handle("好")
    assert turn.reply == "已为您打开车窗儿童锁。"
    assert ("window.all", "window_child_lock", True) in session.changed_signals()


def test_a_command_after_the_question_is_routed_not_consented(session):
    """The driver ignored the question. It is abandoned and the words are routed."""
    session.observe("rear_occupant", "child", confidence=0.9)
    turn = session.handle("把主驾温度调到25度")
    assert "儿童锁" not in turn.reply
    assert not any(a == "window_child_lock" for _, a, _ in session.changed_signals())


def test_an_unknown_scene_key_is_silent_without_a_model(session):
    turn = session.observe("driver_attention", "drowsy", confidence=0.9)
    assert turn.reply == ""
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/cli/test_scene_session.py -q`
Expected: FAIL — `AttributeError: 'Session' object has no attribute 'observe'`

- [ ] **Step 3: Implement in `cli/session.py`**

Import at the top:

```python
import time as _time

from scene.engine import SceneEngine
from scene.facts import VehicleFacts
from scene.llm import FakeSceneLLM
```

In `Session.build`, after `executor = SqliteExecutor(...)` and before `pipe = build_pipeline(...)`, add:

```python
        scene_llm = FakeSceneLLM([]) if fake else None    # a real client is opt-in, see /scene
```

and after `session = cls(...)`:

```python
        session.scene = SceneEngine(cards_by_name={c.name: c for c in cards},
                                    facts=VehicleFacts(car), executor=executor,
                                    llm=scene_llm)
```

Add to `Session`:

```python
    OBSERVATION_TTL = 300.0

    def observe(self, key: str, value, confidence: float = 0.9, source: str = "cabin_cam") -> Turn:
        """One perception event in, one Turn out — the scene analogue of handle()."""
        from scene.context import Observation
        now = _time.monotonic()
        obs = Observation(f"inside.{key}", value, confidence, source, now, self.OBSERVATION_TTL)
        before = self._snapshot()
        outcome = self.scene.observe(obs, now, question_open=False)
        after = self._snapshot()
        deltas = [Delta(e, a, before.get((e, a)), v) for (e, a), v in after.items()
                  if before.get((e, a)) != v]
        return Turn(utterance=f"[scene] {key}={value}", reply=outcome.speech,
                    spans=[], scene=outcome.scene, deltas=deltas)
```

and change `handle()` so consent is consulted **before** routing:

```python
    def handle(self, utterance: str) -> Turn:
        before = self._snapshot()
        now = _time.monotonic()
        try:
            consent = self.scene.resolve(utterance, now)
            if consent.answered:
                after = self._snapshot()
                deltas = [Delta(e, a, before.get((e, a)), v) for (e, a), v in after.items()
                          if before.get((e, a)) != v]
                return Turn(utterance=utterance, reply=consent.speech, spans=[],
                            scene="consent", deltas=deltas)
            result = self.pipeline.route(utterance)
        except Exception as exc:                      # a crash costs a 60s reload; survive it
            return Turn(utterance=utterance, error=f"{type(exc).__name__}: {exc}")
        after = self._snapshot()
        deltas = [Delta(e, a, before.get((e, a)), v) for (e, a), v in after.items()
                  if before.get((e, a)) != v]
        return Turn(utterance=utterance, reply=result.reply,
                    spans=[self._span(cl, deltas) for cl in result.clauses])
```

Extend the `Turn` dataclass with two optional fields (last, so existing keyword construction is
unaffected):

```python
    scene: str = ""
    deltas: list = field(default_factory=list)
```

- [ ] **Step 4: Render a scene turn in `cli/render.py`**

In `render()`, before the span loop:

```python
    if turn.scene:
        lines = [f"  scene        {turn.scene}"]
        lines += [f"  executed     {d.entity}/{d.attribute}   {d.before} → {d.after}"
                  for d in turn.deltas]
        return "\n".join(lines + [f"  reply        {turn.reply}"]) + "\n"
```

- [ ] **Step 5: Add `/scene` to `cli/__main__.py`**

In the command dispatch, alongside `/car` and `/reset`:

```python
        if line.startswith("/scene"):
            parts = line.split()[1:]
            if not parts or "=" not in parts[0]:
                print("  usage: /scene <key>=<value> [conf=0.9]")
                continue
            key, _, value = parts[0].partition("=")
            conf = 0.9
            for extra in parts[1:]:
                if extra.startswith("conf="):
                    conf = float(extra.split("=", 1)[1])
            print(render(session.observe(key, value, confidence=conf)))
            continue
```

Add a `/scene <key>=<value> [conf=]` line to the `/help` text.

- [ ] **Step 6: Run**

Run: `python3 -m pytest tests/cli -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add cli/ tests/cli/test_scene_session.py
git commit -m "feat(cli): /scene, and consent is consulted before routing"
```

---

## Task 9: End-to-end cases and the contract sweep

**Files:**
- Create: `tests/scene/test_scene_e2e.py`, `tests/scene/test_contract_sweep.py`

- [ ] **Step 1: Write the tests**

```python
# tests/scene/test_scene_e2e.py
"""The whole chain, against a real simulated car."""
import pytest

from cli.session import Session


@pytest.fixture
def session():
    return Session.build(fake=True, llm=False, gate="permissive")


def test_a_scene_locks_the_windows_and_the_car_then_refuses_to_open_them(session):
    """The interaction this slice exists to demonstrate. A proactive action changes what a
    later driver-initiated command is allowed to do, and the refusal is explained — both
    entry points, all four workflow steps, one car.

    The precondition is already in the seeded car: open_window requires
    window_child_lock == False and refuses with 车窗儿童锁已开启 (sim/seed.py:39).
    """
    assert session.observe("rear_occupant", "child", 0.9).reply == "后排有小孩，要打开儿童锁吗？"
    assert session.handle("好").reply == "已为您打开车窗儿童锁。"
    assert ("window.all", "window_child_lock", True) in session.changed_signals()
    assert "车窗儿童锁已开启" in session.handle("开车窗").reply


def test_declining_leaves_the_car_untouched(session):
    session.observe("rear_occupant", "child", 0.9)
    assert session.handle("不用").reply == "好的。"
    assert session.changed_signals() == []


def test_a_second_event_after_consent_says_nothing(session):
    """The lock is on, so the rule's Signal condition rejects — silence, not a repeat."""
    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    assert session.observe("rear_occupant", "child", 0.9).reply == ""


def test_a_weak_detection_says_nothing_without_a_model(session):
    assert session.observe("rear_occupant", "child", 0.55).reply == ""
```

```python
# tests/scene/test_contract_sweep.py
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
from t2f.cards import load_catalog
from t2f.types import ExecResult
from t2f.validate import validate_tool_call


class _Facts:
    def __init__(self, answers): self.answers = answers
    def signal(self, e, a): return self.answers.get((e, a))


class _Executor:
    def __init__(self): self.calls = []
    def execute(self, tc):
        self.calls.append(tc)
        return ExecResult(ok=True)


@pytest.fixture(scope="module")
def cards():
    return {c.name: c for c in load_catalog("data/catalog")}


def _satisfying_facts(rule):
    return _Facts({(c.entity, c.attribute): c.equals for c in rule.when if isinstance(c, Signal)})


def _satisfying_context(engine, rule, now):
    for cond in rule.when:
        if isinstance(cond, Observed):
            engine.context.update(Observation(cond.key, cond.equals, 1.0, "test", now, 300.0))


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_a_rule_match_never_reaches_the_car_on_its_own(cards, rule):
    """Consent is the ONLY path to the vehicle. This is the invariant the whole design
    exists to make true, so it is asserted over every rule rather than one."""
    ex = _Executor()
    eng = SceneEngine(cards, _satisfying_facts(rule), ex, rules=(rule,))
    _satisfying_context(eng, rule, now=100.0)
    eng.observe(Observation("_tick", 1, 1.0, "test", 100.0, 300.0), now=100.0)
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
def test_every_rule_speaks_chinese_and_no_identifiers(rule):
    text = speech_for(rule.intent)
    assert text, f"{rule.id} has no speech template"
    assert not re.search(r"[A-Za-z_]", text), text


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_a_rule_stays_silent_when_its_signal_condition_already_holds(cards, rule):
    """Never ask for what is already true."""
    inverted = _Facts({(c.entity, c.attribute): (not c.equals) if isinstance(c.equals, bool)
                       else object()
                       for c in rule.when if isinstance(c, Signal)})
    if not any(isinstance(c, Signal) for c in rule.when):
        pytest.skip("no signal condition")
    eng = SceneEngine(cards, inverted, _Executor(), rules=(rule,))
    _satisfying_context(eng, rule, now=100.0)
    assert eng.observe(Observation("_tick", 1, 1.0, "test", 100.0, 300.0), now=100.0) == NO_ACTION


@pytest.mark.parametrize("rule", RULES, ids=[r.id for r in RULES])
def test_cooldown_is_never_bypassed(cards, rule):
    eng = SceneEngine(cards, _satisfying_facts(rule), _Executor(), rules=(rule,))
    _satisfying_context(eng, rule, now=100.0)
    first = eng.observe(Observation("_tick", 1, 1.0, "test", 100.0, 300.0), now=100.0)
    if first == NO_ACTION:
        pytest.skip("rule did not fire")
    eng._pending = None          # remove the pending-dedup path so cooldown is what is tested
    second = eng.observe(Observation("_tick", 2, 1.0, "test", 101.0, 300.0), now=101.0)
    assert second == NO_ACTION
```

- [ ] **Step 2: Run**

Run: `python3 -m pytest tests/scene -q`
Expected: all pass.

- [ ] **Step 3: Run the whole suite**

Run: `python3 -m pytest -q`
Expected: everything passes, 0 xfailed.

- [ ] **Step 4: Commit**

```bash
git add tests/scene/test_scene_e2e.py tests/scene/test_contract_sweep.py
git commit -m "test(scene): end-to-end chain and a contract sweep over every rule"
```

---

## Task 10: Scene evaluation

**Files:**
- Create: `data/eval/scenes.jsonl`, `eval/scene_metrics.py`, `eval/run_scene_eval.py`
- Create: `tests/scene/test_scene_metrics.py`

- [ ] **Step 1: Author the gold file**

`data/eval/scenes.jsonl` — one JSON object per line. `events` are applied in order, then
`utterance` (if present) is handled. `expect` is the reply the driver should hear, `""` for
silence.

```jsonl
{"id": "child_clear",       "events": [{"key": "rear_occupant", "value": "child", "conf": 0.90}], "expect": "后排有小孩，要打开儿童锁吗？"}
{"id": "child_weak",        "events": [{"key": "rear_occupant", "value": "child", "conf": 0.55}], "expect": ""}
{"id": "child_too_weak",    "events": [{"key": "rear_occupant", "value": "child", "conf": 0.30}], "expect": ""}
{"id": "adult_rear",        "events": [{"key": "rear_occupant", "value": "adult", "conf": 0.95}], "expect": ""}
{"id": "unknown_key",       "events": [{"key": "driver_attention", "value": "drowsy", "conf": 0.90}], "expect": ""}
{"id": "consent_yes",       "events": [{"key": "rear_occupant", "value": "child", "conf": 0.90}], "utterance": "好",           "expect": "已为您打开车窗儿童锁。", "expect_executed": true}
{"id": "consent_no",        "events": [{"key": "rear_occupant", "value": "child", "conf": 0.90}], "utterance": "不用",         "expect": "好的。"}
{"id": "consent_lookalike", "events": [{"key": "rear_occupant", "value": "child", "conf": 0.90}], "utterance": "好像有点热",   "expect_consent": false}
{"id": "consent_command",   "events": [{"key": "rear_occupant", "value": "child", "conf": 0.90}], "utterance": "把窗户关上",   "expect_consent": false}
{"id": "consent_unrelated", "events": [{"key": "rear_occupant", "value": "child", "conf": 0.90}], "utterance": "后排太热了",   "expect_consent": false}
```

- [ ] **Step 2: Write the failing metric tests**

```python
# tests/scene/test_scene_metrics.py
from eval.scene_metrics import (avg_llm_calls_per_event, scene_false_consent_rate,
                                scene_false_speech_rate, scene_recall)

SPOKE_WHEN_SILENT = [{"expect": "", "actual": "话"}, {"expect": "", "actual": ""}]
SHOULD_HAVE_SPOKEN = [{"expect": "问", "actual": "问"}, {"expect": "问", "actual": ""}]


def test_false_speech_counts_only_rows_gold_says_are_silent():
    assert scene_false_speech_rate(SPOKE_WHEN_SILENT) == 0.5


def test_false_speech_is_zero_when_nothing_spoke_out_of_turn():
    assert scene_false_speech_rate(SHOULD_HAVE_SPOKEN) == 0.0


def test_recall_counts_only_rows_gold_says_should_speak():
    assert scene_recall(SHOULD_HAVE_SPOKEN) == 0.5


def test_false_consent_counts_rows_that_must_not_consent():
    rows = [{"expect_consent": False, "consented": True},
            {"expect_consent": False, "consented": False}]
    assert scene_false_consent_rate(rows) == 0.5


def test_an_empty_denominator_is_zero_not_a_crash():
    """A vacuous 1.0 reads as success. Every metric here reports 0.0 on no rows, and the
    runner prints the denominator beside it."""
    for fn in (scene_false_speech_rate, scene_recall, scene_false_consent_rate,
               avg_llm_calls_per_event):
        assert fn([]) == 0.0
```

- [ ] **Step 3: Implement `eval/scene_metrics.py`**

```python
"""Scene metrics. Every one reports its denominator so a vacuous score is visible.

`scene_false_speech_rate` is the number this design optimises for: it is the proactive
analogue of `ood_false_execution_rate`, and it counts the times the system spoke when the
gold says it should have kept quiet.
"""
from __future__ import annotations


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def scene_false_speech_rate(rows) -> float:
    silent = [r for r in rows if not r.get("expect")]
    return _rate(sum(1 for r in silent if r.get("actual")), len(silent))


def scene_recall(rows) -> float:
    should = [r for r in rows if r.get("expect")]
    return _rate(sum(1 for r in should if r.get("actual") == r["expect"]), len(should))


def scene_false_consent_rate(rows) -> float:
    must_not = [r for r in rows if r.get("expect_consent") is False]
    return _rate(sum(1 for r in must_not if r.get("consented")), len(must_not))


def avg_llm_calls_per_event(rows) -> float:
    return _rate(sum(r.get("llm_calls", 0) for r in rows), len(rows))
```

- [ ] **Step 4: Implement `eval/run_scene_eval.py`**

```python
"""Two arms: S (rules only) and S_llm (rules + fallback), differing only in the client.

Usage:
    python3 -m eval.run_scene_eval --arm S
    python3 -m eval.run_scene_eval --arm S_llm
"""
from __future__ import annotations
import argparse
import json

from cli.session import Session
from eval import scene_metrics as M


def _run_row(row) -> dict:
    session = Session.build(fake=True, llm=False, gate="permissive")
    spoken = ""
    for event in row["events"]:
        spoken = session.observe(event["key"], event["value"], event.get("conf", 0.9)).reply or spoken
    out = dict(row)
    out["llm_calls"] = getattr(session.scene.llm, "calls", 0)
    if "utterance" not in row:
        out["actual"] = spoken
        return out
    before = session.changed_signals()
    reply = session.handle(row["utterance"]).reply
    out["actual"] = reply
    out["consented"] = session.changed_signals() != before
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["S", "S_llm"], default="S")
    ap.add_argument("--dataset", default="data/eval/scenes.jsonl")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.dataset, encoding="utf-8") if line.strip()]
    results = [_run_row(r) for r in rows]
    report = {
        "arm": args.arm,
        "n_rows": len(results),
        "metrics": {
            "scene_false_speech_rate": M.scene_false_speech_rate(results),
            "scene_recall": M.scene_recall(results),
            "scene_false_consent_rate": M.scene_false_consent_rate(results),
            "avg_llm_calls_per_event": M.avg_llm_calls_per_event(results),
        },
        "denominators": {
            "silent_rows": sum(1 for r in results if not r.get("expect")),
            "speaking_rows": sum(1 for r in results if r.get("expect")),
            "consent_rows": sum(1 for r in results if r.get("expect_consent") is False),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run**

```bash
python3 -m pytest tests/scene/test_scene_metrics.py -q
python3 -m eval.run_scene_eval --arm S
```

Expected: tests pass; the arm-S report shows `scene_false_speech_rate 0.0` and
`scene_false_consent_rate 0.0`, with non-zero denominators printed beside them. If any
denominator is 0 the corresponding metric is vacuous — say so rather than reporting the score.

- [ ] **Step 6: Full suite, then commit**

```bash
python3 -m pytest -q
git add data/eval/scenes.jsonl eval/scene_metrics.py eval/run_scene_eval.py tests/scene/test_scene_metrics.py
git commit -m "eval(scene): four scene metrics and the S / S_llm arms"
```

---

## Done criteria

- `python3 -m pytest -q` passes with **0 xfailed** (the last red case is closed by Task 6)
- `python3 -m eval.run_scene_eval --arm S` reports `scene_false_speech_rate` and
  `scene_false_consent_rate` at **0.000** with non-zero denominators
- `python3 -m cli` then `/scene rear_occupant=child conf=0.9` → 好 → `开车窗` reproduces the
  three-step interaction in Task 9's first test
- No routing metric moved: re-run `python3 -m eval.run_eval --arm C --dataset data/eval/gold.jsonl --fake --permissive` before and after Task 6 and diff the two reports. `reply_exact_match` is expected to move; `recall@1`, `multi_intent_set_recall`, `ood_false_execution_rate`, `context_false_action_rate` and `incorrect_execution_rate` must not.
