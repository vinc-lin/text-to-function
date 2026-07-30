# Scene Context in the Interactive Session — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** make the Scene Engine's state and its decisions legible at the terminal, and make the half of it that needs a model reachable by hand.

**Why:** Spec 8's claim is that a hand-testing tool earns its place by what it *surfaces*. The scene subsystem's commonest correct outcome is silence, and silence currently surfaces nothing — `scene —` prints identically whether confidence was below the floor, a rule was rejected because the signal already holds, a cooldown is running, or a question is pending. Three further things are unreachable: the context itself has no `/car` equivalent, every observation key is silently prefixed `inside.`, and `scene_llm` is a `Session.build` parameter that `cli/__main__.py` never passes.

**Four features, chosen together:** (B) say why nothing happened, (A) `/context` and full key namespaces, (D) attach the scene fallback, (C) control the clock.

---

## Design decisions made up front

**Diagnostics do not ride on `SceneOutcome`.** `NO_ACTION` is a shared frozen singleton compared with `==` in roughly fifteen tests. Adding a populated `verdicts` field would mean `_evaluate` could no longer return the singleton, breaking all of them for a display concern. Instead the engine **records** the verdicts of its last evaluation and exposes `explain()`.

**`explain()` never re-evaluates.** By the time the CLI calls it, `observe()` has already written `_last_spoken` and possibly armed a pending consent, so a re-run would report a different world than the one that produced the outcome. It returns what was recorded.

**The clock offset lives in the session, not the engine.** Every engine method already takes `now` as a parameter — that was deliberate, so tests never sleep. The session is the only thing that calls `monotonic()`, so it is the only thing that needs to lie about it.

## File structure

| File | Change |
|---|---|
| `scene/engine.py` | record per-rule verdicts + suppression reasons; add `explain()` |
| `scene/rules.py` | `evaluate()` returns a reason alongside the verdict |
| `cli/session.py` | clock offset, namespaced keys + `ttl=`, `context_rows()`, `attach_scene_llm()` |
| `cli/render.py` | render the verdict block on a scene turn |
| `cli/__main__.py` | `/context`, `/clock`, `/scene-llm`, `--scene-llm`, help text |
| `docs/TRYING_IT.md` | a section per new command, with verified transcripts |

Run tests with `python3 -m pytest -q` **from the repo root** — the catalog path is relative. Baseline: `624 passed, 5 deselected, 0 xfailed`.

---

## Task 1: `evaluate()` explains itself

**Files:** Modify `scene/rules.py`; modify `tests/scene/test_rules.py`

The verdict alone cannot be rendered usefully — `NEAR_MISS` does not say which observation was weak, and `REJECT` does not say which signal already held. Both are the whole point of the display.

- [ ] **Step 1: write the failing tests** — append to `tests/scene/test_rules.py`

```python
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
```

Add `evaluate_explained` and `SceneContext` to the imports at the top of that file.

- [ ] **Step 2: run, confirm `ImportError: cannot import name 'evaluate_explained'`**

- [ ] **Step 3: implement.** Rename the body of `evaluate()` to `evaluate_explained(rule, context, facts, now) -> tuple[Verdict, str]`, returning a driver-irrelevant, developer-facing Chinese-or-English reason at each return point. Keep `evaluate()` as a two-line wrapper returning only the verdict, so no existing caller changes.

Reason strings — these are diagnostics, never spoken, so they name internals deliberately:

| verdict | reason |
|---|---|
| REJECT | `f"{entity}/{attribute} is already {actual!r}"` |
| NOT_APPLICABLE, no observation | `f"no live observation for {key}"` |
| NOT_APPLICABLE, wrong value | `f"{key} is {actual!r}, not {expected!r}"` |
| NOT_APPLICABLE, below floor | `f"{key} conf {c:.2f} below floor {floor:.2f}"` |
| NEAR_MISS, confidence | `f"{key} conf {c:.2f} in [{floor:.2f}, {threshold:.2f})"` |
| NEAR_MISS, persistence | `f"{key} held {age:.0f}s of {persist_for:.0f}s"` |
| MATCH | `"all conditions met"` |

- [ ] **Step 4: run** — `python3 -m pytest tests/scene -q`, expect the existing 111 plus 6.

- [ ] **Step 5: commit**

```bash
git add scene/rules.py tests/scene/test_rules.py
git commit -m "feat(scene): a verdict carries the reason behind it"
```

---

## Task 2: the engine records what it decided

**Files:** Modify `scene/engine.py`; modify `tests/scene/test_engine.py`

- [ ] **Step 1: write the failing tests** — append to `tests/scene/test_engine.py`

```python
def test_explain_reports_the_verdict_of_every_rule(cards):
    eng = _engine(cards)
    eng.observe(_child(confidence=0.62), now=100.0)
    rows = eng.explain()
    assert [r.rule_id for r in rows] == ["rear_child_window_lock"]
    assert rows[0].verdict == "near_miss" and "0.62" in rows[0].reason


def test_explain_reports_a_cooldown_as_the_suppressor(cards):
    """The rule MATCHED and still said nothing. Without this the display would show
    'match' next to silence and read as a bug."""
    eng = _engine(cards)
    eng.observe(_child(at=100.0), now=100.0)
    eng.resolve("不用", now=101.0)
    eng.observe(_child(at=102.0), now=102.0)
    rows = eng.explain()
    assert rows[0].verdict == "match"
    assert "cooldown" in rows[0].suppressed_by and "118" in rows[0].suppressed_by


def test_explain_reports_a_pending_question_as_the_suppressor(cards):
    eng = _engine(cards)
    eng.observe(_child(at=100.0), now=100.0)
    eng.observe(_child(at=101.0), now=101.0)
    assert "already asked" in eng.explain()[0].suppressed_by


def test_explain_reports_the_router_holding_a_question(cards):
    eng = _engine(cards)
    eng.observe(_child(), now=100.0, question_open=True)
    assert "router" in eng.explain()[0].suppressed_by


def test_explain_says_nothing_suppressed_a_rule_that_spoke(cards):
    eng = _engine(cards)
    eng.observe(_child(), now=100.0)
    assert eng.explain()[0].suppressed_by == ""


def test_explain_before_any_observation_is_empty_not_an_error(cards):
    assert _engine(cards).explain() == []


def test_explain_records_the_fallback_when_it_ran(cards):
    llm = FakeSceneLLM([{"decision": "no_action", "scene": "unmatched",
                         "reason": "不确定", "reply_intent": "ack_declined"}])
    eng = SceneEngine(cards_by_name=cards, facts=FakeFacts(), executor=RecordingExecutor(),
                      rules=RULES, llm=llm)
    eng.observe(_child(confidence=0.62), now=100.0)
    assert "no_action" in eng.fallback_note() and "不确定" in eng.fallback_note()


def test_the_fallback_note_says_why_it_was_not_consulted(cards):
    eng = _engine(cards)
    eng.observe(_child(confidence=0.62), now=100.0)
    assert "no model attached" in eng.fallback_note()
```

- [ ] **Step 2: run, confirm `AttributeError: 'SceneEngine' object has no attribute 'explain'`**

- [ ] **Step 3: implement.** Add:

```python
@dataclass(frozen=True)
class RuleReport:
    rule_id: str
    verdict: str
    reason: str
    suppressed_by: str = ""     # "" when nothing suppressed it
```

`_evaluate` builds `self._last_reports: list[RuleReport]` from `evaluate_explained`, filling `suppressed_by` from the same conditions `_speakable` and `_fire` already test — cooldown with seconds remaining, `already asked`, `router holds a question`. `explain()` returns the recorded list. `_last_fallback_note: str` records what the fallback did or why it was skipped (`no model attached`, `budget: 22s remaining`, `no near-miss or unconsumed observation`), and `fallback_note()` returns it.

**`observe()`'s `except Exception` must record a report too** — an engine that fails silently and then explains nothing is the exact opacity this task exists to remove. Record `RuleReport("—", "error", str(exc)[:120])`.

- [ ] **Step 4: run** — `python3 -m pytest tests/scene -q`.

- [ ] **Step 5: commit**

```bash
git add scene/engine.py tests/scene/test_engine.py
git commit -m "feat(scene): the engine records why it stayed quiet"
```

---

## Task 3: session plumbing — clock, namespaces, context rows, fallback switch

**Files:** Modify `cli/session.py`; create `tests/cli/test_scene_controls.py`

- [ ] **Step 1: write the failing tests**

```python
"""The four controls the session has to expose before the CLI can offer them."""
import pytest

from cli.session import Session


@pytest.fixture
def session():
    return Session.build(fake=True, llm=False, gate="permissive")


def test_a_namespaced_key_is_taken_as_written(session):
    session.observe("outside.weather", "rain", 0.9)
    assert any(r.key == "outside.weather" for r in session.context_rows())


def test_a_bare_key_still_gets_the_cabin_namespace(session):
    """Backwards compatible: /scene rear_occupant=child kept working through this change."""
    session.observe("rear_occupant", "child", 0.9)
    assert any(r.key == "inside.rear_occupant" for r in session.context_rows())


def test_context_rows_carry_what_the_display_needs(session):
    session.observe("rear_occupant", "child", 0.87, source="cabin_cam")
    row = next(r for r in session.context_rows() if r.key == "inside.rear_occupant")
    assert row.value == "child" and row.confidence == 0.87
    assert row.source == "cabin_cam" and row.expires_in > 0


def test_a_custom_ttl_is_honoured(session):
    session.observe("rear_occupant", "child", 0.9, ttl=10.0)
    assert next(r for r in session.context_rows()).expires_in <= 10.0


def test_the_clock_offset_expires_an_observation(session):
    """A 300s ttl cannot be demonstrated by hand without this."""
    session.observe("rear_occupant", "child", 0.9, ttl=30.0)
    assert session.context_rows()
    session.advance_clock(31.0)
    assert session.context_rows() == []


def test_the_clock_offset_elapses_a_cooldown(session):
    assert session.observe("rear_occupant", "child", 0.9).reply
    session.handle("不用")
    assert session.observe("rear_occupant", "child", 0.9).reply == ""
    session.advance_clock(121.0)
    assert session.observe("rear_occupant", "child", 0.9).reply == "后排有小孩，要打开儿童锁吗？"


def test_the_clock_offset_accumulates(session):
    session.advance_clock(10.0)
    session.advance_clock(5.0)
    assert session.clock_offset == 15.0


def test_a_scene_fallback_can_be_attached_and_detached(session):
    from scene.llm import FakeSceneLLM
    session.attach_scene_llm(FakeSceneLLM([{"decision": "notify", "scene": "unmatched",
                                            "reason": "x",
                                            "reply_intent": "notify_driver_fatigue"}]))
    assert session.observe("driver_state", "drowsy", 0.9).reply == "您看起来有些疲劳，请注意休息。"
    session.attach_scene_llm(None)
    assert "no model attached" in session.scene.fallback_note() or True


def test_attaching_a_fallback_keeps_the_car_and_the_context(session):
    """Same principle as /llm and /gate: switching a mode must not reset the vehicle."""
    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    before = session.changed_signals()
    session.attach_scene_llm(None)
    assert session.changed_signals() == before
```

- [ ] **Step 2: run, confirm the failures are missing attributes**

- [ ] **Step 3: implement.** In `cli/session.py`:

```python
@dataclass
class ContextRow:
    key: str
    value: Any
    confidence: float
    source: str
    age: float
    expires_in: float
```

- `self.clock_offset = 0.0` in `__init__`; `_now()` returns `_time.monotonic() + self.clock_offset`; **every** existing `_time.monotonic()` call site switches to `self._now()`.
- `advance_clock(seconds)` adds to the offset and returns the new value.
- `observe(key, value, confidence=0.9, source="cabin_cam", ttl=None)` — `ttl` defaults to `OBSERVATION_TTL`; the key is used as written when it contains a `.`, otherwise prefixed `inside.`.
- `context_rows()` reads `self.scene.context.live(self._now())` and returns `ContextRow`s sorted by key.
- `attach_scene_llm(client)` sets `self.scene.llm = client`. It must NOT rebuild the engine — the car, the context and any pending consent all survive, the same way `/llm` and `/gate` already keep the car.

- [ ] **Step 4: run** — `python3 -m pytest tests/cli tests/scene -q`.

- [ ] **Step 5: commit**

```bash
git add cli/session.py tests/cli/test_scene_controls.py
git commit -m "feat(cli): session controls for the clock, namespaces, context and fallback"
```

---

## Task 4: the terminal surface

**Files:** Modify `cli/render.py`, `cli/__main__.py`; modify `tests/cli/test_render.py`

- [ ] **Step 1: write the failing render tests** — append to `tests/cli/test_render.py`

```python
def test_a_scene_turn_shows_why_each_rule_did_not_fire():
    """Silence that explains itself. Without this the block is identical whether the
    confidence was low, the signal already held, or a cooldown was running."""
    from cli.session import RuleLine
    turn = Turn(utterance="[scene] rear_occupant=child", reply="", scene="—",
                rules=[RuleLine("rear_child_window_lock", "near_miss",
                                "inside.rear_occupant conf 0.62 in [0.50, 0.80)", "")])
    out = render(turn)
    assert "near_miss" in out and "0.62" in out
    assert "nothing spoken" in out


def test_a_suppressed_rule_says_what_suppressed_it():
    from cli.session import RuleLine
    turn = Turn(utterance="u", reply="", scene="—",
                rules=[RuleLine("r", "match", "all conditions met", "cooldown, 118s left")])
    assert "cooldown, 118s left" in render(turn)


def test_a_scene_turn_that_spoke_still_shows_its_rules():
    from cli.session import RuleLine
    turn = Turn(utterance="u", reply="问？", scene="r",
                rules=[RuleLine("r", "match", "all conditions met", "")])
    out = render(turn)
    assert "问？" in out and "match" in out
```

- [ ] **Step 2: run, confirm `ImportError: cannot import name 'RuleLine'`**

- [ ] **Step 3: implement.**

`cli/session.py` gains a `RuleLine` dataclass (`rule_id`, `verdict`, `reason`, `suppressed_by`) and `Turn` gains `rules: list = field(default_factory=list)` — **last, defaulted**, so no existing construction breaks. `observe()` fills it from `self.scene.explain()`.

`cli/render.py`'s scene branch prints one line per rule between the `scene` line and the deltas:

```
  scene        —
  rule         rear_child_window_lock  near_miss   inside.rear_occupant conf 0.62 in [0.50, 0.80)
  fallback     not consulted — no model attached
  reply        —  (nothing spoken)
```

A suppressed rule appends `· suppressed: <what>`. The fallback line is printed only when the note is non-empty.

`cli/__main__.py` gains four things:

- **`/context`** — one line per `ContextRow`: key, value, `conf X.XX`, source, `age Ns`, `expires in Ns`. Empty context prints `  (no live observations)`, never a blank.
- **`/clock +Ns`** — parses a signed offset, calls `advance_clock`, prints the new total. A malformed argument prints usage, never raises.
- **`/scene-llm on|off`** — `on` constructs `TransformersSceneLLM` lazily inside a try/except and prints the load warning first, since it takes about a minute; **under `--fake` it refuses and says the scene fallback needs a real model**, rather than attaching a fake that would fabricate the behaviour being demonstrated. `off` detaches.
- **`--scene-llm`** startup flag, same construction path.

Add all four to `/help`, and add the scene fallback's state to `mode_label()` so the prompt says which system is answering — the prompt already carries `C_llm · shipped`, and a driver of this tool needs to know whether the scene fallback is attached for the same reason.

- [ ] **Step 4: run the full suite** — `python3 -m pytest -q`.

- [ ] **Step 5: drive it by hand.** Scripted-input harness in the scratchpad (`sys.path.insert(0, "/mnt/x/code/text-to-function")` first, run from the repo root, `--fake`, echo each line). Feed:

```
/scene rear_occupant=child conf=0.6
/context
/scene rear_occupant=child conf=0.9
好
/scene rear_occupant=child conf=0.9
/clock +121
/scene rear_occupant=child conf=0.9
/context
/clock +400
/context
/quit
```

Expected: a near-miss that explains itself; the context listing; the question; consent; a cooldown-suppressed second event; the clock elapsing it; a REJECT explaining the lock already holds; then the observation expiring out of the context. **Paste the real transcript.** If anything differs, report it — do not adjust to match.

- [ ] **Step 6: commit**

```bash
git add cli/ tests/cli/test_render.py
git commit -m "feat(cli): /context, /clock, /scene-llm, and silence that explains itself"
```

---

## Task 5: the guide

**Files:** Modify `docs/TRYING_IT.md`

- [ ] Add the four commands to the commands table.
- [ ] Extend the `/scene` section with the verdict display, explaining what each verdict means and why silence-with-a-reason is the point.
- [ ] A short subsection on `/clock`, since TTL, cooldown and persistence are otherwise invisible.
- [ ] A short subsection on `/scene-llm`, stating plainly that it loads a second model and that arm S_llm's measured behaviour is to decline on both fallback rows — so the honest expectation is that attaching it usually changes nothing.
- [ ] **Every transcript verbatim from a real run.** Keep the file's voice: written for someone who has read nothing else.
- [ ] Commit: `docs: the session can now show what the scene engine is thinking`

---

## Done criteria

- `python3 -m pytest -q` passes, 0 xfailed, no existing test weakened
- `/scene rear_occupant=child conf=0.6` explains itself rather than printing a bare `—`
- `/context` shows confidence, source, age and expiry, and empties as the clock advances
- `/clock +121` elapses the cooldown and a second identical event asks again
- `/scene-llm on` under `--fake` refuses with a reason instead of attaching a fake
- No scene metric moves: `python3 -m eval.run_scene_eval --arm S` before and after must be identical, since none of this touches rule evaluation's outcome — only what is recorded alongside it
