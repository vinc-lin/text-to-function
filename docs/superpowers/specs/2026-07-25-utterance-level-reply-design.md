# Spec 5 — Utterance-Level Reply

Status: design approved 2026-07-25. Builds on Spec 1 (deterministic router), Spec 2 (constrained-LLM
fallback + multi-turn), Spec 3 (learned confidence gate), Spec 4 (multi-intent, context-aware
routing). On-device SA8797/GGUF port remains deferred.

## 0. Motivation

The pipeline routes, validates, and executes — but it never produces the single string a voice
assistant actually speaks.

What exists today is per-clause only:

- `render_response(card, tool_call)` renders one sentence per executed action from the card's
  `response_template`.
- Clarifications come from `build_clarification`, `build_low_confidence_clarification`, and
  `build_plan_clarification`.
- `RouteResult` has **no aggregate reply field**.

Two concrete defects follow from that:

1. **The plan path leaves the caller to stitch.** Three executed actions produce three separate
   `ClauseResult.response` strings with no defined joining.
2. **The consolidated clarification is duplicated.** `_route_plan` attaches the *same*
   `ClarificationRequest` object to *every* unresolved clause. A caller concatenating naively asks
   the same question two or three times.

Nothing acknowledges a span the actionability filter suppressed as narration, and there is no
contract guaranteeing the caller gets *something* to say.

## 1. Objective

Add a single utterance-level reply, composed deterministically from what the router already
produced.

1. **Uniform contract** — `route()` always returns a non-empty `reply` string, on every path: plan,
   legacy single-intent, low-confidence/OOD, and validation failure.
2. **One question, never three** — at most one clarification question per reply.
3. **Presentational only** — no routing, gating, or execution behavior changes.
4. **Enforced by the harness** — three contract metrics reported for every arm, so a regression
   shows up in the eval run and not only in unit tests.

### Non-goals

- **No LLM-generated phrasing.** The reply is templated. An LLM call on the happy path would cost
  latency, break the 73 ms zero-LLM operating point, and put hallucination risk on a safety-critical
  confirmation.
- **No routing change for pure-context utterances.** An utterance whose spans are all narration
  still falls through to the legacy path exactly as today, including the known
  `context_false_action` gap (Spec 4 RESULTS, lever 1). Closing that is a separate change with a
  measurable coverage cost; bundling it into a presentational feature would confound both.
- **No multi-turn reply.** `FollowUpResolver.resolve` returns a bare `ClauseResult`, not a
  `RouteResult`, and keeps its current `.response` contract.
- **No length cap.** Gold `multi_intent` rows top out at three actions (61 rows × 2, 3 rows × 3), so
  a summarizing cap would be dead code — and it would break `reply_action_coverage` by construction.

## 2. Component

One new module, `t2f/reply.py`, exposing a pure function:

```python
def compose_reply(result: RouteResult) -> str
```

No cards, no state, no I/O. It reads only the `ClauseResult`s already on the `RouteResult`;
per-action strings were rendered at execution time by `render_response`.

**Why a new module rather than growing `respond.py`.** `respond.py` renders card-level strings from a
`FunctionCard` + `ToolCall`. The composer works one altitude up, over a whole `RouteResult`. Keeping
them separate preserves `respond.py`'s single input shape and makes the composer testable from
hand-built `RouteResult`s with no catalog involved.

## 3. Composition rules

Applied in order. Terminology: a *confirmation* is a non-empty `ClauseResult.response`; a *question*
is a non-empty `ClauseResult.clarification.question`.

1. **Collect confirmations** in clause order, dropping exact duplicates.
2. **Sentence-join** — the catalog's templates are self-contained sentences ending in `。`
   (`已为您调整车窗儿童锁状态。`), so confirmations are concatenated with no separator. Defensive: a
   confirmation not already ending in `。`, `！`, or `？` gets a `。` appended.
3. **At most one question.** Collect questions and dedup. The plan path attaches one shared object to
   every unresolved clause, so this collapses to one by construction; if distinct questions remain,
   take the first. Appended as its own sentence after all confirmations.
4. **Hard failure.** A clause with validation errors but neither confirmation nor question
   contributes `抱歉，这个操作没能完成。` — appended only when there is no question. This string is
   new; it is a reply-layer constant and lives in `reply.py`, not in `respond.py`.
5. **Nothing acted.** No confirmations, no question, no failure → `好的。`

### Worked examples

```
3 executed          → 已为您调整车窗儿童锁状态。已将主驾车窗开度调整到40%。已将天窗开度调整到50%。
1 executed + 1 open → 已为您调整车窗儿童锁状态。关于「天窗开到一半」我还需要确认一下，请补充信息。
0 executed + 1 open → 关于「天窗开到一半」我还需要确认一下，请补充信息。
LOW / OOD           → 抱歉，我不太确定您的意思，可以换个说法吗？
legacy HIGH         → 已将当前区域温度设置为25°C。        (== clauses[0].response)
validation failure  → 抱歉，这个操作没能完成。
nothing acted       → 好的。
```

## 4. Wiring

`RouteResult` gains `reply: str = ""`. Exactly one call site — `Pipeline.route` wraps both branches:

```python
def route(self, utterance: str) -> RouteResult:
    ...
    res = self._route_plan(utterance, action_spans) if <multi> else self._route_legacy(utterance)
    res.reply = compose_reply(res)
    return res
```

`_route_plan` and `_route_legacy` are otherwise unchanged.

## 5. Error handling — the composer must be total

By the time `compose_reply` runs, the tool calls have **already executed**. An exception here would
lose the confirmation for work the vehicle actually performed. Therefore an empty `clauses` list, a
`None` or empty `response`, a `clarification` whose question is blank, and a missing `decision` all
degrade through the rules to `好的。` rather than raising.

This is achieved by writing the rules so degenerate cases fall through naturally — not by wrapping
the body in `try/except`, which would hide real defects.

## 6. Evaluation

Three **contract** metrics in `eval/metrics.py`, reported for every arm. All three should read 1.0; a
drop is a bug, not a quality signal.

| metric | definition |
|---|---|
| `reply_action_coverage` | over records with ≥1 executed clause: fraction whose reply contains every executed clause's confirmation |
| `reply_single_question` | fraction of records where at most one **distinct clarification question** appears in the reply |
| `reply_nonempty_rate` | fraction with a non-empty reply — the uniform contract |

`reply_single_question` must **not** be defined by counting `？`: `build_plan_clarification` returns
`关于「…」我还需要确认一下，请补充信息。`, which contains no question mark, so a punctuation-counting
metric would pass trivially. It is defined instead over the recorded question strings — count how many
distinct ones occur as substrings of the reply, require ≤1.

`eval/arms.py::predict` gains three fields to feed these: `"reply": res.reply`,
`"responses": [cl.response for cl in res.clauses]`, and
`"questions": [cl.clarification.question for cl in res.clauses if cl.clarification]`.
`eval/run_eval.py` reports the three metrics and `RESULTS.md` gains a Spec 5 section.

**Rejected: gold `reply` strings.** Annotating a gold reply on the 64 `multi_intent` rows was
considered and rejected. A gold reply bakes in *which actions the arm executed* — arm C
(deterministic) executes a subset of what C_llm executes, so C would score near zero even when its
reply is perfectly composed from what it did execute. That measures routing again, not composition,
and every template tweak would invalidate 64 annotations. Exact-wording protection is bought far more
cheaply by the golden test in §7.

## 7. Test plan

### Unit — `tests/test_reply.py`

Hand-built `RouteResult`s, no catalog: each of the five rules; duplicate-confirmation dedup;
duplicate-question collapse; clause ordering preserved; and the degenerate/total cases from §5.

### Golden — `tests/test_reply_golden.py`

Eight exact-string cases built by rendering real catalog cards through `render_response` with fixed
tool calls, then composing. Exercises `respond.py` + `reply.py` together so template drift is caught,
with no embedder needed.

### End-to-end, fast — `tests/test_reply_e2e.py`

Full `Pipeline.route()` with `FakeEmbedder` + `tests/fixtures/catalog` (`set_temperature`,
`set_fan_speed`, `open_window`). Runs in the default suite. These assert the **contract**, not exact
strings, because band assignment under the fake embedder is an implementation detail not worth
pinning.

| # | utterance | assertion |
|---|---|---|
| E1 | `把空调调到25度` | single-intent: `reply == clauses[0].response` when a response exists; non-empty either way |
| E2 | `开车窗,温度调到25度` | both confirmations present; no duplicated fragment |
| E3 | multi-intent with a narration span | the narration text appears nowhere in `reply` |
| E4 | `今天天气怎么样` | `reply == "抱歉，我不太确定您的意思，可以换个说法吗？"` |
| E5 | `开车窗，温度调高` (relative, no seeded state → `missing_state` → clarify) | reply leads with the confirmation and contains the clarification question exactly once |
| E6 | ~10 varied utterances incl. empty and punctuation-only | `route(u).reply` is always a non-empty `str` |

### End-to-end, model — `tests/test_integration_plan.py`

Extends the existing `test_canonical_multi_intent` (`pytest -m model`, real embedder + real
xgrammar-constrained Qwen3-0.6B). The only exact-string e2e:

| # | assertion |
|---|---|
| E7 | with `state.reset({"set_window_position/driver": 30})`, the canonical utterance yields exactly `已为您调整车窗儿童锁状态。已将主驾车窗开度调整到40%。已将天窗开度调整到50%。` — exact string, span order, narration absent |

E7 is no more brittle than the assertions already in that test, which require all three spans to
confirm.

### Metrics — `tests/test_metrics_reply.py`

The three metrics over synthetic records, including the failure directions (a reply missing a
confirmation, a reply containing two distinct questions, an empty reply).

## 8. Regression gate

- The existing **144 core + 3 model tests** stay green.
- Spec 4 metrics — `multi_intent_set_recall`, `context_false_action_rate`, `ood_false_execution_rate`,
  `incorrect_execution_rate` — are **unchanged**. This work is presentational; any movement there
  means something leaked into routing.
