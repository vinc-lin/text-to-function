# End-to-End Test Cases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a body of end-to-end cases covering the Central Model workflow — a deterministic pytest
suite (green = today's behaviour, red = the workflow as specified) plus new metric-graded eval slices.

**Architecture:** Suite A drives the real `Pipeline.route()` against a 3-card fixture catalog with
`FakeEmbedder` (no model, no GPU). Suite B adds labelled rows in a **separate** dataset file so no
existing `RESULTS.md` number moves. Red cases use `xfail(strict=True)` so they self-report when a gap
closes.

**Tech Stack:** Python 3.10+, pytest, existing `t2f` / `eval` packages. No new dependencies.

**Spec:** [`docs/superpowers/specs/2026-07-26-e2e-test-cases-design.md`](../specs/2026-07-26-e2e-test-cases-design.md)

---

## READ THIS FIRST — measured reality overrides the spec

The spec was written before the fixture harness was probed. **Probing changed five things.** Every
expected string below was **measured by running the real pipeline**, not assumed. Do not invent
literals; if a case does not produce the value stated here, stop and report rather than "fixing" the
assertion to match.

**1. The false-confirmation bug is live and reproducible.** With an executor returning `{"ok": False}`,
`route("把空调调到25度")` returns `'已将当前区域温度设置为25°C。'` — a full success confirmation.
This is the highest-value case in the suite.

**2. `别关车窗` ("don't close the window") dispatches `open_window{is_open: False}` — it closes the
window.** The system executes the opposite of the instruction. The spec listed this as a green
characterization case; it is a **red** case.

**3. Three planned cases are NOT expressible in Suite A** and move to Suite B:
   - `帮我打开自动泊车` (unsupported in-car request) routes to `open_window` and **executes**. This is a
     fixture artifact — 3 cards, loosened thresholds, no OOD prototypes — not a product defect (the
     real catalog scores OOD false-execution at 0.000). Testing OOD here would assert a lie.
   - `把温度调一下` routes to **`open_window`**, not `set_temperature`, because `FakeEmbedder` is a
     hashed-n-gram stand-in with no semantics. A "missing *named* required param" case cannot be
     provoked without catalog work (Task 1, Step 4).
   - `温度调高一点` lands in **MEDIUM** band, and the deterministic planner only plans HIGH spans
     (`t2f/pipeline.py:206-207`), so `missing_state` is unreachable in Suite A.

**4. A MEDIUM span excluded from a plan IS surfaced**, as the failure line — Spec 5's fix working. Do
not describe this as a silent drop.

**5. `把温度调一下` and `车窗` both produce `请补充更多信息。`** — both route to `open_window`, whose
required `is_open` is not in the `_CLARIFY` bank.

---

## Overlap with `tests/test_reply_e2e.py` — a deliberate, bounded exception

The spec (§8) says new cases must not restate existing assertions. Checking the 11 existing tests
found **three exact overlaps on the reply half**:

| new case | existing test | same utterance |
|---|---|---|
| S2-01 | `test_e2e_single_intent` (`:28`) | `把空调调到25度` |
| S2-06 / S4A-03 | `test_e2e_two_actions_sentence_joined` (`:34`) | `开车窗,温度调到25度` |
| S2-13 | `test_e2e_low_confidence_reject` (`:48`) | `今天天气怎么样` |

**They are retained anyway**, because the new cases assert something the old ones cannot: **what was
dispatched to the vehicle**, via `RecordingExecutor`. The existing suite has no executor visibility at
all — it can only see what was *said*. An end-to-end case for a control system has to assert both,
and the pairing is itself the assertion (said-X-and-did-X, not said-X and separately did-X).

**Do not delete or modify `tests/test_reply_e2e.py`.** Its 11 tests are the Spec-5 record and its
count appears in `RESULTS.md`. Any *further* overlap beyond the three above should be dropped rather
than restated.

---

## File Structure

| File | Responsibility |
|---|---|
| `tests/e2e/__init__.py` | package marker (empty) |
| `tests/e2e/doubles.py` | `RecordingExecutor`, `FailingExecutor` — the only new non-test code |
| `tests/e2e/conftest.py` | `build_pipeline()` factory: executor / llm_client / state / thresholds |
| `tests/e2e/test_s2_recognition.py` | S2 cases — segmentation + recognition |
| `tests/e2e/test_s3_execution.py` | S3 cases — dispatch, barrier, executor failure |
| `tests/e2e/test_s4a_confirmation.py` | S4a cases — success confirmations |
| `tests/e2e/test_s4b_failure_cause.py` | S4b cases — failure-cause explanation (mostly red) |
| `eval/metrics.py` | +3 metric functions |
| `eval/dataset.py` | validator rules for the new row types |
| `eval/run_eval.py` | load + score `e2e_cases.jsonl`, register the new metrics |
| `data/eval/e2e_cases.jsonl` | new labelled rows (separate file — do NOT touch `gold.jsonl`) |
| `pyproject.toml` | no change — `xfail` needs no marker registration |

---

### Task 1: Fixture harness — doubles and pipeline factory

**Files:**
- Create: `tests/e2e/__init__.py`, `tests/e2e/doubles.py`, `tests/e2e/conftest.py`
- Test: `tests/e2e/test_s2_recognition.py` (first consumer, Task 2)

- [ ] **Step 1: Create the package marker**

```bash
mkdir -p tests/e2e && touch tests/e2e/__init__.py
```

- [ ] **Step 2: Write the executor doubles**

Create `tests/e2e/doubles.py`:

```python
"""Executor doubles for end-to-end cases.

The repo ships exactly one executor (`t2f/execute.py::MockExecutor`) and it always succeeds,
so a vehicle-side failure is currently inexpressible. `FailingExecutor` is what makes
requirement 4b's "the car refused" branch testable at all.
"""
from __future__ import annotations
from t2f.types import ToolCall


class RecordingExecutor:
    """Records every dispatched call so a case can assert WHAT was actuated."""

    def __init__(self, ok: bool = True):
        self.calls: list[ToolCall] = []
        self.ok = ok

    def execute(self, tool_call: ToolCall) -> dict:
        self.calls.append(tool_call)
        return {"ok": self.ok, "name": tool_call.name, "parameters": tool_call.parameters}

    @property
    def dispatched(self) -> list[tuple[str, dict]]:
        """(function_name, parameters) pairs, in dispatch order."""
        return [(c.name, dict(c.parameters)) for c in self.calls]


class FailingExecutor(RecordingExecutor):
    """Reports a vehicle-side failure. Still records, so a case can assert that the call
    WAS attempted and the reply nonetheless must not claim success."""

    def __init__(self, error: str = "device_unavailable"):
        super().__init__(ok=False)
        self.error = error

    def execute(self, tool_call: ToolCall) -> dict:
        self.calls.append(tool_call)
        return {"ok": False, "error": self.error, "name": tool_call.name}
```

- [ ] **Step 3: Write the pipeline factory**

Create `tests/e2e/conftest.py`:

```python
"""Deterministic end-to-end harness: the REAL Pipeline.route() over a 3-card fixture
catalog with FakeEmbedder. No model, no network, no GPU.

Thresholds are loosened so the hashed-n-gram FakeEmbedder reaches the HIGH band on the
fixture utterances; this mirrors tests/test_reply_e2e.py, which established the pattern.
"""
from __future__ import annotations
from pathlib import Path
import pytest

from t2f.cards import load_catalog
from t2f.config import Config
from t2f.embed import FakeEmbedder
from t2f.gate import ConfidenceGate, Thresholds
from t2f.pipeline import Pipeline, DeterministicResolver, LLMResolver
from t2f.score import Scorer

from .doubles import RecordingExecutor

FIXTURE_CATALOG = Path(__file__).parent.parent / "fixtures" / "catalog"


def build_pipeline(executor=None, llm_client=None, state=None, thresholds=None):
    """Return (pipeline, executor). `executor` defaults to a fresh RecordingExecutor."""
    cards = load_catalog(FIXTURE_CATALOG)
    cfg = Config.default()
    cfg.thresholds = thresholds or Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)
    executor = executor if executor is not None else RecordingExecutor()

    medium = LLMResolver(llm_client) if llm_client is not None else None
    resolver = DeterministicResolver({c.name: c for c in cards},
                                     executor=executor, medium_resolver=medium)
    pipe = Pipeline(cards, FakeEmbedder(256), Scorer(cfg.weights, cfg.domain_keywords),
                    ConfidenceGate(cfg.thresholds), cfg, resolver=resolver)
    if llm_client is not None:
        pipe.llm_client = llm_client          # enables the per-span plan path
    for key, value in (state or {}).items():
        pipe.state.set(key, value, layer="confirmed")
    return pipe, executor


@pytest.fixture
def pipeline():
    """(pipeline, executor) with a RecordingExecutor — the common case."""
    return build_pipeline()
```

- [ ] **Step 4: Probe before asserting — the calibration rule**

`FakeEmbedder` has no semantics, so **which card an utterance retrieves is an empirical fact, not a
guess.** Before writing any new case not listed in this plan, run it and read the result:

```bash
python3 - <<'PY'
import sys; sys.path.insert(0, '.')
sys.path.insert(0, 'tests')
from e2e.conftest import build_pipeline
pipe, ex = build_pipeline()
r = pipe.route("YOUR UTTERANCE HERE")
print("reply     :", repr(r.reply))
print("dispatched:", ex.dispatched)
print("bands     :", [c.decision.band.value for c in r.clauses])
print("chosen    :", [c.decision.chosen for c in r.clauses])
PY
```

If the utterance routes to an unintended card, **do not add an assertion for the intended card** — pick
a different utterance, or move the case to Suite B (Task 9). Record the decision in the task report.

- [ ] **Step 5: Verify the harness imports and routes**

Run: `python3 -m pytest tests/e2e -q --collect-only`
Expected: `no tests ran` with **no import errors** (no test files exist yet).

- [ ] **Step 6: Commit**

```bash
git add tests/e2e/__init__.py tests/e2e/doubles.py tests/e2e/conftest.py
git commit -m "test: e2e harness — executor doubles and deterministic pipeline factory"
```

---

### Task 2: S2 — segmented intent recognition (9 green, 1 red)

**Files:**
- Create: `tests/e2e/test_s2_recognition.py`

Every literal below was measured. `WINDOW`, `TEMP25` etc. are module constants so a template
reword is a one-line edit rather than a scatter of literals.

- [ ] **Step 1: Write the green recognition cases**

```python
"""S2 — segmented intent recognition. Utterance -> spans -> function(s) + parameters.

Expected replies and dispatches are MEASURED against the fixture catalog, not assumed.
"""
import pytest
from .conftest import build_pipeline

WINDOW = "已为您调整当前区域车窗状态。"
TEMP25 = "已将当前区域温度设置为25°C。"
TEMP22 = "已将当前区域温度设置为22°C。"
FAN3 = "已将当前区域风速设置为3档。"
REJECT = "抱歉，我不太确定您的意思，可以换个说法吗？"

# (case id, utterance, expected reply, expected dispatches)
GREEN = [
    ("S2-01", "把空调调到25度", TEMP25, [("set_temperature", {"temperature": 25.0})]),
    ("S2-02", "温度设成22度", TEMP22, [("set_temperature", {"temperature": 22.0})]),
    ("S2-03", "风速调到三档", FAN3, [("set_fan_speed", {"level": 3})]),
    ("S2-04", "开车窗", WINDOW, [("open_window", {"is_open": True})]),
    ("S2-05", "关闭车窗", WINDOW, [("open_window", {"is_open": False})]),
    ("S2-06", "开车窗,温度调到25度", WINDOW + TEMP25,
     [("open_window", {"is_open": True}), ("set_temperature", {"temperature": 25.0})]),
    ("S2-08", "开车窗,风速调到三档,温度调到25度", WINDOW + FAN3 + TEMP25,
     [("open_window", {"is_open": True}), ("set_fan_speed", {"level": 3}),
      ("set_temperature", {"temperature": 25.0})]),
    ("S2-13", "今天天气怎么样", REJECT, []),
]


@pytest.mark.parametrize("case_id,utterance,expected_reply,expected_calls", GREEN,
                         ids=[c[0] for c in GREEN])
def test_s2_recognition(case_id, utterance, expected_reply, expected_calls):
    pipe, ex = build_pipeline()
    result = pipe.route(utterance)
    assert result.reply == expected_reply
    assert ex.dispatched == expected_calls


def test_s2_07_context_is_suppressed_and_absent_from_reply():
    """Narration must not act, and must not be spoken back."""
    pipe, ex = build_pipeline()
    result = pipe.route("我有点热,温度调到25度")
    assert result.reply == TEMP25
    assert "我有点热" not in result.reply
    assert ex.dispatched == [("set_temperature", {"temperature": 25.0})]
    assert result.plan is not None          # context present -> plan path


def test_s2_09_relative_without_value_does_not_execute():
    """A relative op with no state reaches MEDIUM with no LLM: it must fail honestly,
    never silently succeed. (Spec 5's falsely-affirmative fix.)"""
    pipe, ex = build_pipeline()
    result = pipe.route("温度调高一点")
    assert ex.dispatched == []
    assert result.reply == "抱歉，这个操作没能完成。"
```

- [ ] **Step 2: Run — expect all green**

Run: `python3 -m pytest tests/e2e/test_s2_recognition.py -q`
Expected: `10 passed`

- [ ] **Step 3: Add the red negation case**

Append to the same file:

```python
@pytest.mark.xfail(strict=True,
                   reason="别关车窗 ('don't close the window') dispatches is_open=False and "
                          "closes it — polarity is keyword-derived with no negation handling "
                          "(t2f/lexical.py:70-73). Executing the opposite of the instruction.")
def test_s2_11_negation_must_not_invert_the_action():
    pipe, ex = build_pipeline()
    result = pipe.route("别关车窗")
    # Correct behaviour is to NOT close the window. Either take no action, or ask.
    assert ("open_window", {"is_open": False}) not in ex.dispatched
```

- [ ] **Step 4: Run — expect 1 xfailed**

Run: `python3 -m pytest tests/e2e/test_s2_recognition.py -q`
Expected: `10 passed, 1 xfailed`

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/test_s2_recognition.py
git commit -m "test: S2 end-to-end recognition cases (9 green, 1 red: negation inverts the action)"
```

---

### Task 3: S3 — execution and the plan barrier (4 green)

**Files:**
- Create: `tests/e2e/test_s3_execution.py`

- [ ] **Step 1: Write the green execution cases**

```python
"""S3 — execution. What actually gets dispatched, and what the barrier refuses to dispatch."""
from .conftest import build_pipeline

WINDOW = "已为您调整当前区域车窗状态。"
TEMP25 = "已将当前区域温度设置为25°C。"


def test_s3_01_single_valid_dispatches_exactly_once():
    pipe, ex = build_pipeline()
    pipe.route("把空调调到25度")
    assert ex.dispatched == [("set_temperature", {"temperature": 25.0})]


def test_s3_02_multi_intent_dispatches_in_order():
    pipe, ex = build_pipeline()
    pipe.route("开车窗,温度调到25度")
    assert ex.dispatched == [("open_window", {"is_open": True}),
                             ("set_temperature", {"temperature": 25.0})]


def test_s3_03_barrier_executes_only_the_valid_subset():
    """One action is out of range. The barrier must dispatch the valid one and ONLY that."""
    pipe, ex = build_pipeline()
    pipe.route("开车窗,把温度调到99度")
    assert ex.dispatched == [("open_window", {"is_open": True})]


def test_s3_04_barrier_names_the_unexecuted_action_in_the_reply():
    """The refused action must not vanish — the driver has to hear that it did not happen."""
    pipe, ex = build_pipeline()
    result = pipe.route("开车窗,把温度调到99度")
    assert result.reply == WINDOW + "关于「把温度调到99度」我还需要确认一下，请补充信息。"
    assert "把温度调到99度" in result.reply


def test_s3_06_out_of_range_alone_dispatches_nothing():
    pipe, ex = build_pipeline()
    pipe.route("把温度调到99度")
    assert ex.dispatched == []
```

- [ ] **Step 2: Run**

Run: `python3 -m pytest tests/e2e/test_s3_execution.py -q`
Expected: `5 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_s3_execution.py
git commit -m "test: S3 execution + plan-barrier cases (valid subset only, refused action is voiced)"
```

---

### Task 4: S3 — executor failure (2 red) — THE MOST IMPORTANT CASES

**Files:**
- Modify: `tests/e2e/test_s3_execution.py`

These reproduce gap 1. Measured: with `{"ok": False}` the reply is
`'已将当前区域温度设置为25°C。'` — a full success confirmation for an action that failed.

- [ ] **Step 1: Write the failing-executor cases**

Append to `tests/e2e/test_s3_execution.py`:

```python
import pytest
from .doubles import FailingExecutor

GAP1 = ("gap 1: execute()'s return value is discarded at all four call sites "
        "(t2f/plan.py:43, t2f/pipeline.py:64,104, t2f/dialog.py:42) and no dataclass "
        "carries a vehicle-reported outcome, so a failed actuation is spoken as success.")


@pytest.mark.xfail(strict=True, reason=GAP1)
def test_s3_08_failed_actuation_is_not_confirmed_as_success():
    """The single most important case in the suite: the day a real vehicle adapter is
    attached, this is the bug that would tell the driver the car did something it did not."""
    pipe, ex = build_pipeline(executor=FailingExecutor())
    result = pipe.route("把空调调到25度")
    assert ex.dispatched == [("set_temperature", {"temperature": 25.0})]   # it WAS attempted
    assert result.reply != "已将当前区域温度设置为25°C。"                      # but must not claim success


@pytest.mark.xfail(strict=True, reason=GAP1)
def test_s3_09_failed_action_does_not_commit_vehicle_state():
    """t2f/plan.py:44-48 writes the confirmed state layer before knowing the call succeeded."""
    pipe, ex = build_pipeline(executor=FailingExecutor())
    pipe.route("开车窗,温度调到25度")
    assert pipe.state.get("set_temperature") is None
```

- [ ] **Step 2: Run — expect 2 xfailed**

Run: `python3 -m pytest tests/e2e/test_s3_execution.py -q`
Expected: `5 passed, 2 xfailed`

> If `test_s3_09` reports **xpass** (strict → failure), the state key is not what this plan assumes.
> Print `pipe.state` after routing and use the real key. Do not delete the assertion.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_s3_execution.py
git commit -m "test: S3 executor-failure cases (red) — failed actuation is spoken as success"
```

---

### Task 5: S4a — success confirmation (4 green, 1 red)

**Files:**
- Create: `tests/e2e/test_s4a_confirmation.py`

- [ ] **Step 1: Write the cases**

```python
"""S4a — on success, inform the user of the completed action."""
import pytest
from .conftest import build_pipeline

WINDOW = "已为您调整当前区域车窗状态。"
TEMP25 = "已将当前区域温度设置为25°C。"
FAN3 = "已将当前区域风速设置为3档。"


def test_s4a_01_confirmation_states_the_value_set():
    pipe, _ = build_pipeline()
    assert pipe.route("把空调调到25度").reply == TEMP25


def test_s4a_02_confirmation_states_a_level():
    pipe, _ = build_pipeline()
    assert pipe.route("风速调到三档").reply == FAN3


def test_s4a_03_two_confirmations_are_sentence_joined():
    pipe, _ = build_pipeline()
    reply = pipe.route("开车窗,温度调到25度").reply
    assert reply == WINDOW + TEMP25
    assert reply.count(WINDOW) == 1


def test_s4a_04_every_dispatched_call_is_mentioned():
    """Coverage invariant: nothing is actuated silently."""
    pipe, ex = build_pipeline()
    result = pipe.route("开车窗,风速调到三档,温度调到25度")
    assert len(ex.dispatched) == 3
    for confirmation in (WINDOW, FAN3, TEMP25):
        assert confirmation in result.reply


@pytest.mark.xfail(strict=True,
                   reason="gap 4: render_response humanizes only `position`, so 43 of 92 catalog "
                          "cards confirm an action without stating the value. Opening and closing "
                          "the window produce byte-identical replies.")
def test_s4a_07_boolean_action_states_on_or_off():
    pipe, _ = build_pipeline()
    opened = pipe.route("开车窗").reply
    pipe2, _ = build_pipeline()
    closed = pipe2.route("关闭车窗").reply
    assert opened != closed
```

- [ ] **Step 2: Run**

Run: `python3 -m pytest tests/e2e/test_s4a_confirmation.py -q`
Expected: `4 passed, 1 xfailed`

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_s4a_confirmation.py
git commit -m "test: S4a confirmation cases (4 green, 1 red: boolean polarity unhearable)"
```

---

### Task 6: S4b — failure cause (2 green, 6 red)

**Files:**
- Create: `tests/e2e/test_s4b_failure_cause.py`

The green cases lock in the *safety* half (nothing executes). The red cases assert the *explanation*
half, which is gap 2.

- [ ] **Step 1: Write the green no-execution cases and the red cause cases**

```python
"""S4b — on failure, explain the SPECIFIC cause.

Today every hard failure collapses into one constant. These cases separate the two halves:
nothing-executed (green, already true) from cause-explained (red, gap 2).
"""
import pytest
from .conftest import build_pipeline

GENERIC = "抱歉，这个操作没能完成。"
GENERIC_QUESTION = "请补充更多信息。"
GAP2 = ("gap 2: t2f/reply.py never reads ClauseResult.validation_errors, so every cause "
        "collapses into one constant. The cause data exists and reaches the reply layer.")
GAP3 = ("gap 3: _CLARIFY (t2f/respond.py:7-9) knows 3 parameter names, covering 10 of the "
        "catalog's 76 required-parameter slots. `is_open` is not one of them.")

# (case id, utterance) — each fails validation with the cause named in the id
NO_EXECUTION = [
    ("S4B-03 out_of_range above max", "把温度调到99度"),
    ("S4B-04 out_of_range below min", "把温度调到5度"),
    ("S4B-05 out_of_range integer", "风速调到20档"),
]


@pytest.mark.parametrize("case_id,utterance", NO_EXECUTION, ids=[c[0] for c in NO_EXECUTION])
def test_s4b_invalid_value_dispatches_nothing(case_id, utterance):
    """GREEN — the safety half. An unusable value must never reach the vehicle."""
    pipe, ex = build_pipeline()
    pipe.route(utterance)
    assert ex.dispatched == []


@pytest.mark.parametrize("case_id,utterance", NO_EXECUTION, ids=[c[0] for c in NO_EXECUTION])
@pytest.mark.xfail(strict=True, reason=GAP2)
def test_s4b_invalid_value_explains_the_cause(case_id, utterance):
    """RED — the explanation half. The driver is told nothing about WHY."""
    pipe, _ = build_pipeline()
    assert pipe.route(utterance).reply != GENERIC


@pytest.mark.xfail(strict=True, reason=GAP2)
def test_s4b_03_out_of_range_names_the_limit():
    """The bounds are in the card (minimum 16, maximum 32) and never spoken."""
    pipe, _ = build_pipeline()
    reply = pipe.route("把温度调到99度").reply
    assert "16" in reply and "32" in reply


def test_s4b_02_missing_required_param_dispatches_nothing():
    """GREEN — a missing required parameter must not be guessed."""
    pipe, ex = build_pipeline()
    pipe.route("车窗")
    assert ex.dispatched == []


@pytest.mark.xfail(strict=True, reason=GAP3)
def test_s4b_02_missing_required_param_names_what_is_missing():
    """RED — the question does not say WHICH parameter it needs."""
    pipe, _ = build_pipeline()
    assert pipe.route("车窗").reply != GENERIC_QUESTION
```

- [ ] **Step 2: Run**

Run: `python3 -m pytest tests/e2e/test_s4b_failure_cause.py -q`
Expected: `4 passed, 5 xfailed`

- [ ] **Step 3: Add the LLM-driven cause cases**

`bad_enum` and `type_mismatch` cannot come from the deterministic extractors — they only arise from
LLM output. Append:

```python
from t2f.llm.client import FakeLLMClient
from t2f.types import LLMResult, ToolCall


@pytest.mark.xfail(strict=True, reason=GAP2)
def test_s4b_06_bad_enum_explains_the_cause():
    llm = FakeLLMClient(default=LLMResult(
        tool_call=ToolCall("set_temperature", {"temperature": 25, "position": "trunk"})))
    pipe, ex = build_pipeline(llm_client=llm)
    result = pipe.route("温度调高一点")
    assert ex.dispatched == []          # invalid enum must not execute
    assert result.reply != GENERIC      # and the cause must be explained


@pytest.mark.xfail(strict=True, reason=GAP2)
def test_s4b_07_type_mismatch_explains_the_cause():
    llm = FakeLLMClient(default=LLMResult(
        tool_call=ToolCall("set_temperature", {"temperature": "warm"})))
    pipe, ex = build_pipeline(llm_client=llm)
    result = pipe.route("温度调高一点")
    assert ex.dispatched == []
    assert result.reply != GENERIC
```

- [ ] **Step 4: Run and verify the no-execution halves hold**

Run: `python3 -m pytest tests/e2e/test_s4b_failure_cause.py -q`
Expected: `4 passed, 7 xfailed`

> If either LLM case reports **xpass**, it means the reply already differs from `GENERIC` — read the
> actual reply and report it. It may be a clarification rather than a cause, in which case tighten the
> assertion to require the cause word (e.g. `"trunk"` or the enum list) rather than mere difference.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/test_s4b_failure_cause.py
git commit -m "test: S4b failure-cause cases (4 green safety, 7 red explanation)"
```

---

### Task 7: Eval — dataset validator for the new row types

**Files:**
- Modify: `eval/dataset.py:17-40`
- Test: `tests/test_dataset_validation.py` (create if absent; check first with `ls tests/ | grep dataset`)

- [ ] **Step 1: Write the failing validator tests**

```python
from eval.dataset import validate_against_catalog

NAMES = {"set_temperature"}


def test_invalid_row_requires_a_cause():
    rows = [{"utterance": "把温度调到99度", "type": "invalid",
             "expected_functions": ["set_temperature"], "expected_execution": False}]
    assert any("expected_cause" in p for p in validate_against_catalog(rows, NAMES))


def test_invalid_row_must_forbid_execution():
    rows = [{"utterance": "x", "type": "invalid", "expected_functions": ["set_temperature"],
             "expected_cause": "out_of_range"}]
    assert any("expected_execution" in p for p in validate_against_catalog(rows, NAMES))


def test_unknown_cause_is_rejected():
    rows = [{"utterance": "x", "type": "invalid", "expected_functions": ["set_temperature"],
             "expected_execution": False, "expected_cause": "banana"}]
    assert any("banana" in p for p in validate_against_catalog(rows, NAMES))


def test_new_types_must_not_carry_expected_params():
    """param_exact_match is NOT type-filtered, so expected_params on a new row would
    contaminate a headline metric even from a separate file."""
    rows = [{"utterance": "x", "type": "asr_noise", "expected_functions": ["set_temperature"],
             "source_utterance": "y", "expected_params": {"set_temperature": {"temperature": 25}}}]
    assert any("expected_params" in p for p in validate_against_catalog(rows, NAMES))


def test_asr_noise_requires_a_source():
    rows = [{"utterance": "x", "type": "asr_noise", "expected_functions": ["set_temperature"]}]
    assert any("source_utterance" in p for p in validate_against_catalog(rows, NAMES))


def test_a_well_formed_invalid_row_passes():
    rows = [{"utterance": "把温度调到99度", "type": "invalid",
             "expected_functions": ["set_temperature"], "expected_execution": False,
             "expected_cause": "out_of_range"}]
    assert validate_against_catalog(rows, NAMES) == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_dataset_validation.py -q`
Expected: FAIL — the validator has no rules for these types yet.

- [ ] **Step 3: Implement the rules**

In `eval/dataset.py`, add above `validate_against_catalog`:

```python
# Cause codes a row may claim. Eight from t2f/validate.py, two from t2f/state.py, plus
# executor_failed, which becomes reachable once gap 1 threads the executor result.
KNOWN_CAUSES = {
    "not_in_candidates", "unknown_function", "unknown_param", "missing_required",
    "type_mismatch", "out_of_range", "bad_enum", "llm_no_toolcall",
    "no_numeric_param", "missing_state", "executor_failed",
}
NEW_TYPES = {"invalid", "asr_noise"}
```

and inside the row loop, after the existing `multi_intent` check:

```python
        if r["type"] in NEW_TYPES and r.get("expected_params"):
            problems.append(f"row {i}: {r['type']} must not carry expected_params "
                            f"(param_exact_match is not type-filtered)")
        if r["type"] == "invalid":
            if r.get("expected_execution") is not False:
                problems.append(f"row {i}: invalid needs expected_execution: false")
            cause = r.get("expected_cause")
            if not cause:
                problems.append(f"row {i}: invalid needs expected_cause")
            elif cause not in KNOWN_CAUSES:
                problems.append(f"row {i}: unknown expected_cause {cause}")
        if r["type"] == "asr_noise" and not r.get("source_utterance"):
            problems.append(f"row {i}: asr_noise needs source_utterance")
```

- [ ] **Step 4: Run**

Run: `python3 -m pytest tests/test_dataset_validation.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add eval/dataset.py tests/test_dataset_validation.py
git commit -m "eval: validator rules for invalid/asr_noise rows + expected_params contamination guard"
```

---

### Task 8: Eval — three new metrics

**Files:**
- Modify: `eval/metrics.py` (append after `reply_question_drop_rate`)
- Test: `tests/test_metrics_e2e.py`

- [ ] **Step 1: Write the failing metric tests**

```python
from eval import metrics as M


def _rec(row, reply="", executed=None):
    return {"row": row, "reply": reply, "executed": executed or [],
            "responses": [], "questions": []}


def test_invalid_no_execution_rate_is_one_when_nothing_executed():
    recs = [_rec({"type": "invalid"}, executed=[False]),
            _rec({"type": "invalid"}, executed=[False])]
    assert M.invalid_no_execution_rate(recs) == 1.0


def test_invalid_no_execution_rate_catches_an_execution():
    recs = [_rec({"type": "invalid"}, executed=[False]),
            _rec({"type": "invalid"}, executed=[True])]
    assert M.invalid_no_execution_rate(recs) == 0.5


def test_invalid_no_execution_rate_ignores_other_types():
    recs = [_rec({"type": "single"}, executed=[True])]
    assert M.invalid_no_execution_rate(recs) == 1.0        # empty denominator -> want-1.0


def test_reply_exact_match_scores_only_annotated_rows():
    recs = [_rec({"expected_reply": "好的。"}, reply="好的。"),
            _rec({"expected_reply": "不对。"}, reply="好的。"),
            _rec({}, reply="anything")]
    assert M.reply_exact_match(recs) == 0.5


def test_reply_exact_match_empty_denominator_is_one():
    assert M.reply_exact_match([_rec({}, reply="x")]) == 1.0


def test_n_reply_annotated_counts_the_denominator():
    recs = [_rec({"expected_reply": "a"}), _rec({}), _rec({"expected_reply": "b"})]
    assert M.n_reply_annotated(recs) == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_metrics_e2e.py -q`
Expected: FAIL — `AttributeError: module 'eval.metrics' has no attribute 'invalid_no_execution_rate'`

- [ ] **Step 3: Implement**

Append to `eval/metrics.py`:

```python
def invalid_no_execution_rate(records) -> float:
    """Of `type:"invalid"` rows, the fraction where NOTHING was dispatched.

    Want 1.0 — an unusable parameter value must never reach the vehicle. Unlike the four
    Spec-5 reply metrics this one can genuinely fail. Empty denominator -> 1.0, matching
    the want-1.0 convention of schema_valid_rate.
    """
    rows = [r for r in records if r["row"].get("type") == "invalid"]
    if not rows:
        return 1.0
    clean = sum(1 for r in rows if not any(r.get("executed") or []))
    return clean / len(rows)


def reply_exact_match(records) -> float:
    """Over rows carrying `expected_reply`, the fraction where the spoken reply matches.

    Report alongside `n_reply_annotated` — a 1.0 over zero rows is vacuous, which is the
    lesson of reply_action_coverage.
    """
    rows = [r for r in records if r["row"].get("expected_reply")]
    if not rows:
        return 1.0
    hit = sum(1 for r in rows if _reply_of(r) == r["row"]["expected_reply"])
    return hit / len(rows)


def n_reply_annotated(records) -> int:
    """The denominator of reply_exact_match. Emitted so a vacuous 1.0 is visible."""
    return sum(1 for r in records if r["row"].get("expected_reply"))
```

- [ ] **Step 4: Run**

Run: `python3 -m pytest tests/test_metrics_e2e.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add eval/metrics.py tests/test_metrics_e2e.py
git commit -m "eval: invalid_no_execution_rate, reply_exact_match, n_reply_annotated"
```

---

### Task 9: Author the eval rows and wire them into the runner

**Files:**
- Create: `data/eval/e2e_cases.jsonl`
- Modify: `eval/run_eval.py:60-116`

**Do NOT touch `data/eval/gold.jsonl`.** Eight metrics plus the latency percentiles have no row-type
filter; appending there would move numbers throughout `RESULTS.md`.

- [ ] **Step 1: Author the rows**

Create `data/eval/e2e_cases.jsonl`. Target ~50 rows. Every `utterance` must use the **real** 92-card
catalog vocabulary, not the fixture catalog. Shape:

```jsonc
{"utterance": "把温度调到99度", "type": "invalid", "expected_functions": ["set_temperature"], "expected_execution": false, "expected_cause": "out_of_range", "expected_reply": "温度只能设置在16到32度之间。"}
{"utterance": "空调调到零下5度", "type": "invalid", "expected_functions": ["set_temperature"], "expected_execution": false, "expected_cause": "out_of_range"}
{"utterance": "风速开到二十档", "type": "invalid", "expected_functions": ["set_fan_speed"], "expected_execution": false, "expected_cause": "out_of_range"}
{"utterance": "座椅加热开到十档", "type": "invalid", "expected_functions": ["set_seat_heating"], "expected_execution": false, "expected_cause": "out_of_range"}
{"utterance": "帮我把空调调到二十五都", "type": "asr_noise", "expected_functions": ["set_temperature"], "source_utterance": "帮我把空调调到二十五度"}
{"utterance": "那个嗯把车窗开一下", "type": "asr_noise", "expected_functions": ["open_window"], "source_utterance": "把车窗开一下"}
```

Composition target:
| slice | rows | notes |
|---|---|---|
| `invalid` | ~20 | spread across `out_of_range`, `missing_required`, `bad_enum`; use each domain's real min/max |
| `asr_noise` | ~20 | homophone (度/都, 挡/档), filler prefix, dropped particle. **Each must name its `source_utterance`.** |
| relative + `vehicle_state` | ~10 | type `single`, carrying a `vehicle_state` dict |
| `expected_reply` | ~30 annotations | spread across the above; the exact string the driver should hear |

For the ~30 `expected_reply` values: write what the driver **should** hear per the requirement, not
what the system says today. Most will not match until gaps 2–4 close — that is the point, and
`reply_exact_match` will report a low number until then. Say so in the task report; do not soften the
strings to raise the score.

- [ ] **Step 2: Validate the file before wiring it**

```bash
python3 - <<'PY'
from eval.dataset import load_dataset, validate_against_catalog
from t2f.cards import load_catalog
names = {c.name for c in load_catalog("data/catalog")}
rows = load_dataset("data/eval/e2e_cases.jsonl")
problems = validate_against_catalog(rows, names)
print(f"{len(rows)} rows, {len(problems)} problems")
for p in problems[:20]:
    print("  ", p)
PY
```

Expected: `N rows, 0 problems`. Fix the data, never the validator.

- [ ] **Step 3: Wire the runner**

In `eval/run_eval.py`, immediately after the line that loads `ctx_rows` (search for
`ctx_rows =`), add:

```python
    e2e_path = Path("data/eval/e2e_cases.jsonl")
    e2e_rows = load_dataset(e2e_path) if e2e_path.exists() else []
```

(`from pathlib import Path` may already be imported; add it if not.)

At `eval/run_eval.py:81`, inside the `if calibrate:` block, filter it the same way as the others:

```python
        ctx_rows = [r for r in ctx_rows if r.get("split") != "dev"]
        e2e_rows = [r for r in e2e_rows if r.get("split") != "dev"]      # <- add
```

At `eval/run_eval.py:84`, next to `ctx_records`:

```python
    ctx_records = [A.predict(pipe, r) for r in ctx_rows]
    e2e_records = [A.predict(pipe, r) for r in e2e_rows]                 # <- add
```

Then add to the `metrics` dict immediately after `reply_question_drop_rate`:

```python
        "invalid_no_execution_rate": M.invalid_no_execution_rate(e2e_records),
        "reply_exact_match": M.reply_exact_match(records + e2e_records),
        "n_reply_annotated": M.n_reply_annotated(records + e2e_records),
        "n_e2e_rows": len(e2e_records),
```

`records + e2e_records` is correct for the reply metrics: an `expected_reply` may be annotated on a
gold row too, and both metrics are scoped by the presence of the field, not by file.

- [ ] **Step 4: Run the fast harness check**

Run: `python3 -m eval.run_eval --arm C --dataset data/eval/gold.jsonl --fake --permissive`
Expected: the table now ends with `invalid_no_execution_rate`, `reply_exact_match`,
`n_reply_annotated`, `n_e2e_rows`, and **every pre-existing metric is unchanged** from before this task.

- [ ] **Step 5: Commit**

```bash
git add data/eval/e2e_cases.jsonl eval/run_eval.py
git commit -m "eval: e2e_cases.jsonl (invalid / asr_noise / relative slices) wired into the runner"
```

---

### Task 10: Prove no contamination, then document

**Files:**
- Modify: `docs/superpowers/RESULTS.md`, `README.md`

- [ ] **Step 1: Run the full suite**

Run: `python3 -m pytest -q`
Expected: **`243 passed, 11 xfailed`** — zero failures, zero xpasses.

Arithmetic, so a mismatch is diagnosable: 208 existing + 35 new green (Task 2: 10, Task 3: 5,
Task 5: 4, Task 6: 4, Task 7: 6, Task 8: 6) = 243. Red: Task 2: 1, Task 4: 2, Task 5: 1, Task 6: 7
= 11. This is fewer than the spec's 13 because Tasks 2–6 dropped three cases as unprovokable in
Suite A (see READ THIS FIRST, item 3) and merged two others.

Record the exact numbers. If any xpass appears, a gap closed or an assertion is wrong; investigate
before proceeding — do not delete the case.

- [ ] **Step 2: Prove the eval numbers did not move**

Run both arms and diff every metric against the values in `RESULTS.md`:

```bash
python3 -m eval.run_eval --arm C     --dataset data/eval/gold.jsonl --calibrate
python3 -m eval.run_eval --arm C_llm --dataset data/eval/gold.jsonl --calibrate
```

Expected, to four decimals — arm C: recall@1 **0.8644**, multi_intent_set_recall **0.8194**,
param_exact_match **0.2733**, e2e_deterministic **0.1067**, incorrect_execution_rate **0.0312**,
ood_false_execution_rate **0.0000**. Arm C_llm: recall@1 **0.8559**, param_exact_match **0.7248**,
e2e_deterministic **0.6200**, incorrect_execution_rate **0.2850**.

**Any movement is a bug in Task 9, not a new result.** The most likely cause is a row leaking into a
non-type-filtered metric.

- [ ] **Step 3: Write the results section**

Append a Spec-6 section to `docs/superpowers/RESULTS.md` recording: the green/red split with the
measured counts, the four defects the suite reproduces (false confirmation on executor failure,
negation inverting the action, boolean polarity unhearable, no cause spoken), the new metric values
with their denominators, and the proof that gold numbers are unchanged.

State plainly that `asr_noise` rows encode **our belief** about ASR errors rather than measured
misrecognitions, and are therefore weaker evidence than the rest.

- [ ] **Step 4: Update the README status line**

The status block currently says "Specs 1–5 complete (208 automated tests + 3 model-backed)". Update
the counts and add the red count, e.g. "… plus 13 red cases encoding the unmet parts of the workflow
(steps 3 and 4b)". Update the step-4b and step-3 rows of the coverage table to cite the suite.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/RESULTS.md README.md
git commit -m "docs: Spec 6 results — e2e suite, 13 red cases, gold metrics proven unchanged"
```

---

## Definition of done

1. `python3 -m pytest -q` — `243 passed, 11 xfailed`; zero failures, zero xpasses.
2. Both eval arms reproduce every pre-existing metric to four decimals.
3. `invalid_no_execution_rate` reports a real value over a non-zero denominator.
4. Every case ID in the spec's §6 maps to a test, a row, or a written note explaining why it moved to
   Suite B or was dropped as unprovokable.
5. No production code under `t2f/` changed. This plan adds cases and eval plumbing only; the gaps the
   red cases describe are separate work.

## Notes for the implementer

- **Never soften a red assertion to make it pass.** If a red case is wrong, it is wrong about the
  *requirement*, and that is a conversation, not an edit.
- **Never widen a green assertion to make it pass.** Every green literal here was measured. A
  mismatch means behaviour changed and you have found a regression.
- `xfail(strict=True)` turns an unexpected pass into a failure. That is deliberate: it is how the
  suite reports its own progress.
- Task 9's row authoring is the only open-ended step. Timebox it; ~50 well-chosen rows beat 200 sloppy
  ones, and the validator will catch structural mistakes but not semantic ones.
