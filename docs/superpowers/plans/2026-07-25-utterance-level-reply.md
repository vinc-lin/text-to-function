# Utterance-Level Reply Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `Pipeline.route()` returns a single, non-empty `reply` string composed from what the router already produced — executed confirmations sentence-joined, at most one clarification question.

**Architecture:** One pure function `compose_reply(RouteResult) -> str` in a new `t2f/reply.py`, wired at exactly one call site in `Pipeline.route`. It reads only `ClauseResult.response` / `.clarification` / `.validation_errors` — no cards, no state, no I/O. Because it runs *after* execution, it must never raise. Three contract metrics make the harness enforce the guarantee on every eval run.

**Tech Stack:** Python 3.10, pytest, numpy. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-25-utterance-level-reply-design.md`

---

## Background you need

Run tests from the repo root with `python3 -m pytest -q` (pytest prepend-mode; there is no editable install). Model-backed tests are opt-in: `python3 -m pytest -q -m model`.

Relevant existing pieces:

- `t2f/respond.py::render_response(card, tool_call)` renders **one sentence per action** from a card's `response_template`. Real examples, verified against `data/catalog`:
  - `set_window_child_lock {enabled: True}` → `已为您调整车窗儿童锁状态。`
  - `set_window_position {percent: 40, position: "driver"}` → `已将主驾车窗开度调整到40%。`
  - `set_sunroof_position {percent: 50}` → `已将天窗开度调整到50%。`
  - `set_temperature {temperature: 25}` → `已将当前区域温度设置为25°C。`
- `t2f/respond.py::build_low_confidence_clarification()` → `抱歉，我不太确定您的意思，可以换个说法吗？`
- `t2f/respond.py::build_plan_clarification(pending)` → `关于「温度调高」我还需要确认一下，请补充信息。` — **note: no `？` character.** Any "one question" check based on counting `？` would pass trivially. This is why the metric in Task 7 is defined over recorded question strings.
- `Pipeline._route_plan` attaches the **same** `ClarificationRequest` object to every unresolved clause. Deduping questions therefore collapses to one by construction.

Types you will use (`t2f/types.py`, all dataclasses):

```python
ToolCall(name: str, parameters: dict)
Decision(band: Band, chosen: Optional[str], candidates: list[Candidate], ...)
ClarificationRequest(question: str, pending: Optional[PendingState] = None)
ValidationError(code: str, message: str)
ClauseResult(clause, decision, tool_call=None, validation_errors=[], clarification=None,
             response=None, needs_llm=False, latency_ms=0.0)
RouteResult(utterance, clauses=[], plan=None)
```

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `t2f/reply.py` | create | `compose_reply(RouteResult) -> str` — the only reply-composition logic |
| `t2f/types.py` | modify | add `reply: str = ""` to `RouteResult` |
| `t2f/pipeline.py` | modify | one call site in `route()` |
| `eval/metrics.py` | modify | three contract metrics |
| `eval/arms.py` | modify | record `reply`, `responses`, `questions` |
| `eval/run_eval.py` | modify | report the three metrics |
| `tests/test_reply.py` | create | unit — composition rules over hand-built `RouteResult`s |
| `tests/test_reply_golden.py` | create | exact strings rendered through real catalog cards |
| `tests/test_reply_e2e.py` | create | full `route()` with `FakeEmbedder` + fixture catalog |
| `tests/test_metrics_reply.py` | create | the three metrics over synthetic records |
| `tests/test_pipeline.py` | modify | contract: `route()` always sets a non-empty reply |
| `tests/test_integration_plan.py` | modify | `@model` canonical reply, exact string |
| `docs/superpowers/RESULTS.md` | modify | Spec 5 section |

`respond.py` is deliberately **not** modified — it stays card-level. The new failure string `抱歉，这个操作没能完成。` is a reply-layer constant and lives in `reply.py`.

---

## Task 1: Composition core — confirmations, sentence-join, ack

Implements spec rules 1, 2 and 5.

**Files:**
- Create: `t2f/reply.py`
- Test: `tests/test_reply.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reply.py`:

```python
# tests/test_reply.py
from t2f.types import (RouteResult, ClauseResult, Decision, Band,
                       ClarificationRequest, ValidationError)
from t2f.reply import compose_reply


def _clause(response=None, question=None, errors=None):
    clar = ClarificationRequest(question=question) if question is not None else None
    return ClauseResult(clause="x", decision=Decision(Band.HIGH, "f", []),
                        response=response, clarification=clar,
                        validation_errors=list(errors or []))


def _result(*clauses):
    return RouteResult(utterance="u", clauses=list(clauses))


def test_single_confirmation_passes_through():
    res = _result(_clause(response="已将天窗开度调整到50%。"))
    assert compose_reply(res) == "已将天窗开度调整到50%。"


def test_multiple_confirmations_are_sentence_joined():
    res = _result(_clause(response="已为您调整车窗儿童锁状态。"),
                  _clause(response="已将主驾车窗开度调整到40%。"),
                  _clause(response="已将天窗开度调整到50%。"))
    assert compose_reply(res) == ("已为您调整车窗儿童锁状态。"
                                  "已将主驾车窗开度调整到40%。"
                                  "已将天窗开度调整到50%。")


def test_duplicate_confirmations_are_deduped():
    res = _result(_clause(response="已为您调整当前区域车窗状态。"),
                  _clause(response="已为您调整当前区域车窗状态。"))
    assert compose_reply(res) == "已为您调整当前区域车窗状态。"


def test_clause_order_is_preserved():
    res = _result(_clause(response="甲。"), _clause(response="乙。"))
    assert compose_reply(res) == "甲。乙。"


def test_confirmation_without_terminator_gets_one():
    res = _result(_clause(response="已执行set_fan_speed"))
    assert compose_reply(res) == "已执行set_fan_speed。"


def test_nothing_acted_returns_ack():
    assert compose_reply(_result(_clause())) == "好的。"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_reply.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 't2f.reply'`

- [ ] **Step 3: Write the minimal implementation**

Create `t2f/reply.py`:

```python
# t2f/reply.py
"""Utterance-level reply composition.

Sits one altitude above `respond.py`: that module renders a sentence for a single
(card, tool_call); this one composes a whole RouteResult into the single string a
voice assistant speaks. Pure — no cards, no state, no I/O.
"""
from __future__ import annotations
from .types import RouteResult

_TERMINATORS = "。！？"
_ACK = "好的。"


def _sentence(text: str) -> str:
    """Ensure a rendered fragment ends with a sentence terminator."""
    if not text:
        return ""
    return text if text[-1] in _TERMINATORS else text + "。"


def _confirmations(clauses) -> list[str]:
    """Non-empty responses in clause order, exact duplicates dropped."""
    out: list[str] = []
    for cl in clauses:
        text = (cl.response or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def compose_reply(result: RouteResult) -> str:
    """Compose the utterance-level reply. Runs AFTER execution, so it must never raise."""
    clauses = result.clauses or []
    parts = [_sentence(t) for t in _confirmations(clauses)]
    return "".join(parts) if parts else _ACK
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_reply.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add t2f/reply.py tests/test_reply.py
git commit -m "feat(reply): compose confirmations into one sentence-joined reply"
```

---

## Task 2: One question, and the hard-failure line

Implements spec rules 3 and 4. A reply carries **at most one** clarification question; a clause that failed validation with nothing to say contributes a failure line only when there is no question.

**Files:**
- Modify: `t2f/reply.py`
- Test: `tests/test_reply.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reply.py`:

```python
def test_confirmation_then_single_question():
    res = _result(_clause(response="已为您调整车窗儿童锁状态。"),
                  _clause(question="关于「温度调高」我还需要确认一下，请补充信息。"))
    assert compose_reply(res) == ("已为您调整车窗儿童锁状态。"
                                  "关于「温度调高」我还需要确认一下，请补充信息。")


def test_repeated_plan_question_collapses_to_one():
    # _route_plan attaches the SAME ClarificationRequest to every unresolved clause
    q = "关于「温度调高」「天窗开到一半」我还需要确认一下，请补充信息。"
    res = _result(_clause(response="已为您调整车窗儿童锁状态。"),
                  _clause(question=q), _clause(question=q))
    assert compose_reply(res) == "已为您调整车窗儿童锁状态。" + q
    assert compose_reply(res).count(q) == 1


def test_distinct_questions_first_wins():
    res = _result(_clause(question="问甲？"), _clause(question="问乙？"))
    assert compose_reply(res) == "问甲？"


def test_question_only():
    res = _result(_clause(question="抱歉，我不太确定您的意思，可以换个说法吗？"))
    assert compose_reply(res) == "抱歉，我不太确定您的意思，可以换个说法吗？"


def test_question_without_terminator_gets_one():
    res = _result(_clause(question="请补充信息"))
    assert compose_reply(res) == "请补充信息。"


def test_hard_failure_line_when_no_question():
    res = _result(_clause(errors=[ValidationError("out_of_range", "temperature 99 > 32")]))
    assert compose_reply(res) == "抱歉，这个操作没能完成。"


def test_hard_failure_appended_after_confirmations():
    res = _result(_clause(response="已为您调整当前区域车窗状态。"),
                  _clause(errors=[ValidationError("out_of_range", "bad")]))
    assert compose_reply(res) == "已为您调整当前区域车窗状态。抱歉，这个操作没能完成。"


def test_question_suppresses_the_failure_line():
    res = _result(_clause(errors=[ValidationError("out_of_range", "bad")]),
                  _clause(question="请补充信息。"))
    reply = compose_reply(res)
    assert reply == "请补充信息。"
    assert "没能完成" not in reply
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_reply.py -q`
Expected: `8 failed, 6 passed` — the new tests return `好的。` instead of the question/failure text.

- [ ] **Step 3: Extend the implementation**

In `t2f/reply.py`, add the `_FAILURE` constant next to `_ACK`:

```python
_TERMINATORS = "。！？"
_ACK = "好的。"
_FAILURE = "抱歉，这个操作没能完成。"
```

Add two helpers after `_confirmations`:

```python
def _questions(clauses) -> list[str]:
    """Non-empty clarification questions in clause order, exact duplicates dropped."""
    out: list[str] = []
    for cl in clauses:
        clar = cl.clarification
        text = ((clar.question if clar else "") or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _has_failure(clauses) -> bool:
    """A clause that failed validation and has nothing else to say."""
    for cl in clauses:
        spoke = bool((cl.response or "").strip())
        asked = bool(cl.clarification and (cl.clarification.question or "").strip())
        if cl.validation_errors and not spoke and not asked:
            return True
    return False
```

Replace the body of `compose_reply` with:

```python
def compose_reply(result: RouteResult) -> str:
    """Compose the utterance-level reply. Runs AFTER execution, so it must never raise."""
    clauses = result.clauses or []
    parts = [_sentence(t) for t in _confirmations(clauses)]
    questions = _questions(clauses)
    if questions:                      # at most ONE question per reply
        parts.append(_sentence(questions[0]))
    elif _has_failure(clauses):
        parts.append(_FAILURE)
    return "".join(parts) if parts else _ACK
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_reply.py -q`
Expected: `14 passed`

- [ ] **Step 5: Commit**

```bash
git add t2f/reply.py tests/test_reply.py
git commit -m "feat(reply): at most one clarification question, plus hard-failure line"
```

---

## Task 3: Totality — the composer must never raise

`compose_reply` runs **after** the tool calls have executed. An exception here would lose the confirmation for work the vehicle actually performed. Degenerate input degrades to `好的。` instead.

Note the composer never touches `ClauseResult.decision`, so a malformed decision is structurally impossible to trip over — no code needed for that case, but it is asserted.

**Files:**
- Modify: `t2f/reply.py` (only if a test fails)
- Test: `tests/test_reply.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reply.py`:

```python
import pytest


def test_empty_clause_list_returns_ack():
    assert compose_reply(RouteResult(utterance="u", clauses=[])) == "好的。"


def test_whitespace_response_treated_as_absent():
    assert compose_reply(_result(_clause(response="   "))) == "好的。"


def test_blank_question_treated_as_absent():
    assert compose_reply(_result(_clause(question=""))) == "好的。"


def test_none_question_treated_as_absent():
    res = _result(ClauseResult(clause="x", decision=Decision(Band.LOW, None, []),
                               clarification=ClarificationRequest(question=None)))
    assert compose_reply(res) == "好的。"


def test_missing_decision_is_never_read():
    res = _result(ClauseResult(clause="x", decision=None, response="已执行。"))
    assert compose_reply(res) == "已执行。"


@pytest.mark.parametrize("res", [
    RouteResult(utterance="", clauses=[]),
    RouteResult(utterance="u", clauses=None),
    _result(_clause()),
    _result(_clause(response=None, question=None, errors=[])),
])
def test_composer_never_raises(res):
    assert isinstance(compose_reply(res), str)
    assert compose_reply(res)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_reply.py -q`
Expected: most of the new tests already pass (the rules were written to degrade); any that fail identify a real gap. Note the exact failures before continuing.

- [ ] **Step 3: Fix only what failed**

If `test_whitespace_response_treated_as_absent` or `test_blank_question_treated_as_absent` fail, the `.strip()` calls in `_confirmations` / `_questions` are missing — add them as written in Tasks 1 and 2. If `RouteResult(clauses=None)` fails, confirm `compose_reply` starts with `clauses = result.clauses or []`. Make no other change: do **not** add a `try/except` around the body, which would hide real defects.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_reply.py -q`
Expected: `23 passed`

- [ ] **Step 5: Commit**

```bash
git add t2f/reply.py tests/test_reply.py
git commit -m "test(reply): composer is total — degenerate input degrades to ack"
```

---

## Task 4: Wire into `RouteResult` and `Pipeline.route`

**Files:**
- Modify: `t2f/types.py` (the `RouteResult` dataclass)
- Modify: `t2f/pipeline.py` (imports; `route`, currently lines 125-136)
- Test: `tests/test_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
def test_route_always_sets_a_nonempty_reply():
    """Uniform contract: every path returns something speakable."""
    p = _pipeline()
    for utterance in ["把空调调到25度", "开车窗,温度调到25度", "今天天气怎么样",
                      "今天天气怎么样，开车窗", "", "   ", "。", "，，，"]:
        res = p.route(utterance)
        assert isinstance(res.reply, str), utterance
        assert res.reply.strip(), utterance


def test_single_intent_reply_matches_clause_response():
    res = _pipeline().route("把空调调到25度")
    assert res.reply == res.clauses[0].response
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_pipeline.py -q`
Expected: FAIL with `AttributeError: 'RouteResult' object has no attribute 'reply'`

- [ ] **Step 3: Add the field and the call site**

In `t2f/types.py`, extend the `RouteResult` dataclass:

```python
@dataclass
class RouteResult:
    utterance: str
    clauses: list[ClauseResult] = field(default_factory=list)
    plan: Optional[ActionPlan] = None
    reply: str = ""
```

In `t2f/pipeline.py`, add the import next to the existing `respond` import:

```python
from .reply import compose_reply
```

Replace the tail of `route` (the two `return` statements) so both branches flow through one composition point:

```python
        if len(action_spans) >= 2 or (action_spans and context_spans):
            res = self._route_plan(utterance, action_spans)
        else:
            res = self._route_legacy(utterance)
        res.reply = compose_reply(res)
        return res
```

Leave the comment block above the `if` untouched, and leave `_route_plan` / `_route_legacy` unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_pipeline.py tests/test_reply.py -q`
Expected: `28 passed`

Then confirm nothing else broke: `python3 -m pytest -q`
Expected: `169 passed`, 3 deselected (144 existing + 23 from `test_reply.py` + 2 here).

- [ ] **Step 5: Commit**

```bash
git add t2f/types.py t2f/pipeline.py tests/test_pipeline.py
git commit -m "feat(reply): RouteResult.reply, composed at one call site in route()"
```

---

## Task 5: Golden replies against the real catalog

Exact-string protection against template drift, with fixed inputs so nothing depends on routing. No embedder, no models — this runs in the default suite.

**Files:**
- Create: `tests/test_reply_golden.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reply_golden.py`:

```python
# tests/test_reply_golden.py
"""Exact replies rendered through REAL catalog cards with fixed tool calls.
Catches drift in either respond.py's templates or reply.py's composition."""
from pathlib import Path
from t2f.cards import load_catalog
from t2f.types import (RouteResult, ClauseResult, Decision, Band, ToolCall,
                       ClarificationRequest, ValidationError)
from t2f.respond import render_response
from t2f.reply import compose_reply

CATALOG = Path(__file__).resolve().parents[1] / "data" / "catalog"
CARDS = {c.name: c for c in load_catalog(CATALOG)}


def _executed(name, params):
    tc = ToolCall(name=name, parameters=params)
    return ClauseResult(clause=name, decision=Decision(Band.HIGH, name, []),
                        tool_call=tc, response=render_response(CARDS[name], tc))


def _asked(question):
    return ClauseResult(clause="q", decision=Decision(Band.MEDIUM, None, []),
                        clarification=ClarificationRequest(question=question))


def _failed():
    return ClauseResult(clause="bad", decision=Decision(Band.HIGH, "set_temperature", []),
                        validation_errors=[ValidationError("out_of_range", "temperature 99 > 32")])


def _reply(*clauses):
    return compose_reply(RouteResult(utterance="u", clauses=list(clauses)))


def test_golden_canonical_three_actions():
    assert _reply(
        _executed("set_window_child_lock", {"enabled": True}),
        _executed("set_window_position", {"percent": 40, "position": "driver"}),
        _executed("set_sunroof_position", {"percent": 50}),
    ) == "已为您调整车窗儿童锁状态。已将主驾车窗开度调整到40%。已将天窗开度调整到50%。"


def test_golden_partial_failure():
    assert _reply(
        _executed("set_window_child_lock", {"enabled": True}),
        _asked("关于「温度调高」我还需要确认一下，请补充信息。"),
    ) == "已为您调整车窗儿童锁状态。关于「温度调高」我还需要确认一下，请补充信息。"


def test_golden_single_temperature_no_position():
    assert _reply(_executed("set_temperature", {"temperature": 25})) == \
        "已将当前区域温度设置为25°C。"


def test_golden_single_temperature_with_position():
    assert _reply(_executed("set_temperature", {"temperature": 26, "position": "passenger"})) == \
        "已将副驾温度设置为26°C。"


def test_golden_low_confidence_only():
    assert _reply(_asked("抱歉，我不太确定您的意思，可以换个说法吗？")) == \
        "抱歉，我不太确定您的意思，可以换个说法吗？"


def test_golden_duplicate_actions_deduped():
    assert _reply(
        _executed("set_sunroof_position", {"percent": 50}),
        _executed("set_sunroof_position", {"percent": 50}),
    ) == "已将天窗开度调整到50%。"


def test_golden_hard_failure_only():
    assert _reply(_failed()) == "抱歉，这个操作没能完成。"


def test_golden_nothing_acted():
    assert _reply(ClauseResult(clause="x", decision=Decision(Band.LOW, None, []))) == "好的。"
```

- [ ] **Step 2: Run the test**

Run: `python3 -m pytest tests/test_reply_golden.py -q`
Expected: `8 passed`. If a golden string mismatches, **do not edit the expectation** — the template or the composer changed, and one of them is the bug. Diff against `data/catalog/*.yaml` `response_template` values before touching anything.

- [ ] **Step 3: Commit**

```bash
git add tests/test_reply_golden.py
git commit -m "test(reply): golden replies rendered through the real catalog"
```

---

## Task 6: End-to-end through `Pipeline.route()`

Full routing with `FakeEmbedder` + `tests/fixtures/catalog` (`set_temperature`, `set_fan_speed`, `open_window`). No models, runs in the default suite. **Every expected string below was measured against the current code**, so they are safe to assert exactly.

**Files:**
- Create: `tests/test_reply_e2e.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reply_e2e.py`:

```python
# tests/test_reply_e2e.py
"""End-to-end: utterance -> route() -> reply, with the deterministic FakeEmbedder
and the reduced fixture catalog. Expected strings are measured, not assumed."""
from pathlib import Path
from t2f.cards import load_catalog
from t2f.embed import FakeEmbedder
from t2f.score import Scorer
from t2f.gate import ConfidenceGate, Thresholds
from t2f.pipeline import Pipeline
from t2f.config import Config

FIX = Path(__file__).parent / "fixtures" / "catalog"

WINDOW = "已为您调整当前区域车窗状态。"
TEMP25 = "已将当前区域温度设置为25°C。"
REJECT = "抱歉，我不太确定您的意思，可以换个说法吗？"


def _pipeline():
    cards = load_catalog(FIX)
    cfg = Config.default()
    cfg.thresholds = Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)
    return Pipeline(cards, FakeEmbedder(256), Scorer(cfg.weights, cfg.domain_keywords),
                    ConfidenceGate(cfg.thresholds), cfg)


def test_e2e_single_intent():                                            # E1
    res = _pipeline().route("把空调调到25度")
    assert res.reply == TEMP25
    assert res.reply == res.clauses[0].response


def test_e2e_two_actions_sentence_joined():                              # E2
    res = _pipeline().route("开车窗,温度调到25度")
    assert res.plan is not None
    assert res.reply == WINDOW + TEMP25
    assert res.reply.count(WINDOW) == 1


def test_e2e_narration_absent_from_reply():                              # E3
    res = _pipeline().route("今天天气怎么样，开车窗")
    assert res.plan is not None
    assert res.reply == WINDOW
    assert "天气" not in res.reply


def test_e2e_low_confidence_reject():                                    # E4
    assert _pipeline().route("今天天气怎么样").reply == REJECT


def test_e2e_partial_failure_one_question():                             # E5
    """开车窗 executes; 温度调高 is LOW-band -> exactly one question, after the confirmation."""
    res = _pipeline().route("开车窗，温度调高")
    assert res.reply == WINDOW + REJECT
    assert res.reply.startswith(WINDOW)
    assert res.reply.count(REJECT) == 1


def test_e2e_reply_is_always_speakable():                                # E6
    p = _pipeline()
    for utterance in ["把空调调到25度", "开车窗,温度调到25度", "今天天气怎么样",
                      "今天天气怎么样，开车窗", "开车窗，温度调高", "外面在下雨，把车窗关上",
                      "", "   ", "。", "，，，"]:
        reply = p.route(utterance).reply
        assert isinstance(reply, str) and reply.strip(), utterance


def test_e2e_every_executed_confirmation_appears():                      # coverage invariant
    p = _pipeline()
    for utterance in ["把空调调到25度", "开车窗,温度调到25度", "外面在下雨，把车窗关上"]:
        res = p.route(utterance)
        for cl in res.clauses:
            if cl.response:
                assert cl.response in res.reply, (utterance, cl.response)
```

- [ ] **Step 2: Run the test**

Run: `python3 -m pytest tests/test_reply_e2e.py -q`
Expected: `7 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/test_reply_e2e.py
git commit -m "test(reply): end-to-end reply through route() on the fake-embedder path"
```

---

## Task 7: Contract metrics in the eval harness

Three metrics that should all read **1.0**. A drop is a bug, not a quality signal — this is what turns the reply contract into something every eval run enforces.

**Files:**
- Modify: `eval/metrics.py` (append)
- Modify: `eval/arms.py` (the `predict` return dict, currently lines 74-78)
- Modify: `eval/run_eval.py` (the `metrics` dict, currently lines 87-105)
- Test: `tests/test_metrics_reply.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics_reply.py`:

```python
# tests/test_metrics_reply.py
from eval.metrics import reply_action_coverage, reply_single_question, reply_nonempty_rate


def _rec(reply, responses=(), questions=()):
    return {"row": {"type": "single"}, "reply": reply,
            "responses": list(responses), "questions": list(questions)}


def test_reply_nonempty_rate():
    recs = [_rec("已将天窗开度调整到50%。"), _rec(""), _rec("   ")]
    assert reply_nonempty_rate(recs) == 1 / 3


def test_reply_action_coverage_all_present():
    assert reply_action_coverage([_rec("甲。乙。", ["甲。", "乙。"])]) == 1.0


def test_reply_action_coverage_missing_one():
    assert reply_action_coverage([_rec("甲。", ["甲。", "乙。"])]) == 0.0


def test_reply_action_coverage_skips_rows_with_no_execution():
    recs = [_rec("好的。", [None]), _rec("甲。", ["甲。"])]
    assert reply_action_coverage(recs) == 1.0


def test_reply_action_coverage_tolerates_dedup():
    """The reply dedups identical confirmations; containment must still hold."""
    assert reply_action_coverage([_rec("甲。", ["甲。", "甲。"])]) == 1.0


def test_reply_single_question_ok_when_repeated_identically():
    # _route_plan attaches the SAME question object to every unresolved clause
    assert reply_single_question([_rec("甲。问A", questions=["问A", "问A"])]) == 1.0


def test_reply_single_question_fails_on_two_distinct():
    assert reply_single_question([_rec("问A问B", questions=["问A", "问B"])]) == 0.0


def test_reply_single_question_is_not_punctuation_based():
    """build_plan_clarification contains no '？' — a '？'-counting metric would pass trivially."""
    q1 = "关于「甲」我还需要确认一下，请补充信息。"
    q2 = "关于「乙」我还需要确认一下，请补充信息。"
    assert reply_single_question([_rec(q1 + q2, questions=[q1, q2])]) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_metrics_reply.py -q`
Expected: collection error — `ImportError: cannot import name 'reply_action_coverage' from 'eval.metrics'`

- [ ] **Step 3: Add the metrics**

Append to `eval/metrics.py`:

```python
def _reply_of(record) -> str:
    return (record.get("reply") or "").strip()


def reply_nonempty_rate(records) -> float:
    """Uniform contract: route() always produces something speakable (want -> 1.0)."""
    if not records:
        return 0.0
    return sum(1 for r in records if _reply_of(r)) / len(records)


def reply_action_coverage(records) -> float:
    """Over rows that executed something: does the reply carry EVERY confirmation? (want -> 1.0)"""
    rows = []
    for r in records:
        confirms = [c.strip() for c in (r.get("responses") or []) if c and c.strip()]
        if confirms:
            rows.append((r, confirms))
    if not rows:
        return 0.0
    ok = sum(1 for r, confirms in rows if all(c in _reply_of(r) for c in confirms))
    return ok / len(rows)


def reply_single_question(records) -> float:
    """At most ONE distinct clarification question may appear in a reply (want -> 1.0).

    Deliberately NOT defined by counting '？': build_plan_clarification returns
    '关于「…」我还需要确认一下，请补充信息。', which has no question mark, so a
    punctuation-counting metric would pass trivially.
    """
    if not records:
        return 0.0
    ok = 0
    for r in records:
        reply = _reply_of(r)
        distinct = {q.strip() for q in (r.get("questions") or []) if q and q.strip()}
        if sum(1 for q in distinct if q in reply) <= 1:
            ok += 1
    return ok / len(records)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_metrics_reply.py -q`
Expected: `8 passed`

- [ ] **Step 5: Record the new fields in `eval/arms.py`**

In `predict`, extend the returned dict — add three keys alongside the existing ones (keep everything already there):

```python
    return {"row": row, "ranked_per_clause": ranked, "predicted_functions": preds,
            "bands": bands, "tool_calls": tcs, "executed": executed, "needs_llm": needs,
            "params_per_clause": params, "exec_correct": exec_ok, "val_errors": verrs,
            "llm_json_ok": llm_json_ok,
            "reply": res.reply,
            "responses": [cl.response for cl in res.clauses],
            "questions": [cl.clarification.question for cl in res.clauses if cl.clarification],
            "latencies": [cl.latency_ms for cl in res.clauses]}
```

- [ ] **Step 6: Report them in `eval/run_eval.py`**

In the `metrics` dict, add three entries immediately after `"json_valid_rate"`:

```python
        "json_valid_rate": M.json_valid_rate(records),
        "reply_action_coverage": M.reply_action_coverage(records),
        "reply_single_question": M.reply_single_question(records),
        "reply_nonempty_rate": M.reply_nonempty_rate(records),
```

- [ ] **Step 7: Verify the harness end-to-end**

Run: `python3 -m eval.run_eval --arm C --dataset data/eval/gold.jsonl --fake --permissive`
Expected: the printed markdown table now contains `reply_action_coverage`, `reply_single_question`, and `reply_nonempty_rate`, all `1.0000`.

Then the full suite: `python3 -m pytest -q`
Expected: `192 passed`, 3 deselected.

- [ ] **Step 8: Commit**

```bash
git add eval/metrics.py eval/arms.py eval/run_eval.py tests/test_metrics_reply.py
git commit -m "eval: reply contract metrics (coverage, single question, non-empty)"
```

---

## Task 8: Model-backed canonical reply, real eval runs, RESULTS

**Files:**
- Modify: `tests/test_integration_plan.py` (extend `test_canonical_multi_intent`)
- Modify: `docs/superpowers/RESULTS.md`

- [ ] **Step 1: Extend the `@model` canonical test**

Append these assertions to the **end of the existing** `test_canonical_multi_intent` in `tests/test_integration_plan.py` (do not add a second test — a fresh one would reload both models):

```python
    # the spoken reply: three confirmations, sentence-joined, narration absent
    assert rr.reply == ("已为您调整车窗儿童锁状态。"
                        "已将主驾车窗开度调整到40%。"
                        "已将天窗开度调整到50%。")
    assert "后排小孩" not in rr.reply
```

- [ ] **Step 2: Run the model test**

Run: `python3 -m pytest -q -m model`
Expected: `3 passed` (~30s; loads the real embedder and the xgrammar-constrained Qwen3-0.6B on GPU).

If the reply mismatches, print `rr.reply` and compare against the plan's action list — a difference means either an action failed to confirm (a routing issue, out of scope) or the composition changed (a real regression).

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration_plan.py
git commit -m "test(reply): assert the canonical multi-intent reply end-to-end"
```

- [ ] **Step 4: Run the real evaluation**

Arm C first — fast, deterministic, no LLM:

```bash
PYTHONPATH=/mnt/x/code/text-to-function python3 -m eval.run_eval \
  --arm C --dataset data/eval/gold.jsonl --calibrate
```

Then arm C_llm, which calls the LLM per span and takes far longer — launch it in the background:

```bash
PYTHONPATH=/mnt/x/code/text-to-function python3 -m eval.run_eval \
  --arm C_llm --dataset data/eval/gold.jsonl --calibrate
```

Record the three reply metrics from each run. **All three must be 1.0.** Anything less is a defect in `compose_reply` — find it before writing RESULTS.

- [ ] **Step 5: Verify no routing regression**

Compare against the Spec 4 numbers in `docs/superpowers/RESULTS.md`. These four must be **unchanged** on both arms:

`multi_intent_set_recall`, `context_false_action_rate`, `ood_false_execution_rate`, `incorrect_execution_rate`

This work is presentational; movement in any of them means something leaked into routing. Stop and investigate rather than documenting the drift.

- [ ] **Step 6: Write the RESULTS section**

Append a `# Spec 5 — Utterance-Level Reply Results` section to `docs/superpowers/RESULTS.md` covering: the three contract metrics per arm (measured, not estimated); the unchanged Spec 4 metrics as evidence of no regression; the canonical reply string; and the two design findings — sentence-join was chosen because the catalog's templates are self-contained `已…` sentences that comma-join into `已…，已…，已…`, and `reply_single_question` is not punctuation-based because `build_plan_clarification` contains no `？`. Note the final test count.

- [ ] **Step 7: Final verification and commit**

```bash
python3 -m pytest -q          # expect: 192 passed, 3 deselected
python3 -m pytest -q -m model # expect: 3 passed
git add docs/superpowers/RESULTS.md eval_report_C.json eval_report_C_llm.json
git commit -m "eval: Spec 5 reply contract results + RESULTS section"
```

---

## Self-review notes

**Spec coverage:** §2 component → Task 1; §3 rules 1/2/5 → Task 1, rules 3/4 → Task 2; §4 wiring → Task 4; §5 totality → Task 3; §6 metrics → Task 7; §7 unit → Tasks 1-3, golden → Task 5, fast e2e → Task 6, model e2e → Task 8, metric tests → Task 7; §8 regression gate → Task 8 steps 5 and 7.

**Deviation from spec §7, E4/E5.** The spec left the fast e2e assertions as contract-only, on the assumption that fake-embedder band assignment was too unstable to pin. It was measured instead: under `Thresholds(0.2, 0.0, 0.05)` with the fixture catalog, `今天天气怎么样` is reliably LOW and `开车窗，温度调高` reliably splits into one HIGH execution plus one LOW clarification. Task 6 therefore asserts exact strings, which is strictly stronger. E5 exercises a LOW-band clarification rather than the `missing_state` path the spec sketched; the `missing_state` shape is covered by golden case 2 and by the unit tests.

**Test counts:** 144 existing + 23 (`test_reply.py`, Tasks 1-3) + 2 (`test_pipeline.py`, Task 4) + 8 (`test_reply_golden.py`, Task 5) + 7 (`test_reply_e2e.py`, Task 6) + 8 (`test_metrics_reply.py`, Task 7) = **192**, plus 3 model tests deselected by default. Intermediate full-suite checkpoints: 169 after Task 4, 192 after Task 7. The `-q` totals quoted inside a task refer to the files named in that command. If the arithmetic disagrees with the observed total, trust the observed total and note the discrepancy — never edit tests to hit a number.
