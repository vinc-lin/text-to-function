# End-to-End Test Cases — Design

**Date:** 2026-07-26
**Goal:** A body of end-to-end cases that exercises the whole Central Model workflow — utterance in,
tool calls dispatched, reply spoken — serving both development (fast regression net) and evaluation
(metric-graded slices).

**Architecture:** Two suites, deliberately separate. **Suite A** is deterministic pytest scenarios
running the real `Pipeline.route()` against a small fixture catalog with `FakeEmbedder` — no model, no
GPU, runs on every commit. **Suite B** is new labelled rows scored by the existing eval harness on
real models. Within Suite A, cases are split by polarity: **A1** captures behaviour as it is today,
**A2** encodes the workflow as specified and therefore starts failing.

---

## 1. Why

The repo has 208 tests and 328 labelled eval rows, and yet **not one case anywhere asserts that the
driver hears the right thing**, and **not one case exercises a failure**.

Two facts establish this:

- `data/eval/gold.jsonl` has fields `utterance`, `expected_functions`, `expected_params`, `type`,
  `split`, `vehicle_state`. **There is no expected-reply field.** Step 4 of the business workflow —
  half the product — is graded only by the four Spec-5 contract metrics, which check non-emptiness and
  question-count and, as `RESULTS.md` already admits for `reply_action_coverage`, cannot fail by
  construction.
- Gold row types are `single` / `multi_intent` / `ood` / `ambiguous`. A scan for out-of-range values
  finds **zero** rows carrying an unusable parameter. Every case ever scored either routes correctly
  or is refused. **The failure taxonomy has no representation at all**, so there is nothing to attach
  a failure-cause assertion to even if we wanted one.

This is what makes "does the system meet the workflow?" a document you have to trust rather than a
number you can read off a test run. The
[Central Model system design](2026-07-25-central-model-system-design.md) records six gaps that are
invisible to every existing test. This spec makes them executable.

## 2. The testable envelope

There is no microphone, no vehicle bus, and no TTS in this repo, so "end to end" means:

> **given** an utterance, optionally a vehicle state, optionally an executor behaviour
> **assert** which tool calls were dispatched with which parameters, and the exact reply string

That is steps 2, 3 and 4 of the workflow. Step 1 and the speaking itself are outside the boundary and
outside this spec.

## 3. Goals and non-goals

**Goals**

1. Every case is traceable to a workflow step, so coverage is auditable against the requirement rather
   than against our module list.
2. Suite A runs in the default `pytest -q` with no model, no network, no GPU.
3. Adding Suite B **must not change any number currently in `RESULTS.md`** (see §5.1 — this is a hard
   constraint, not a preference).
4. When a gap closes, the suite says so by itself rather than waiting for someone to remember.

**Non-goals**

- No ASR, TTS, audio, or serving-host testing — outside the envelope (§2).
- No on-device / SA8797 / soak / memory benchmarking. Blocked on hardware; tracked as gap 7.
- No behaviour changes. This spec adds cases and two test doubles. It does not fix the gaps it
  exposes; the red cases are the specification for that later work.
- Not re-annotating all 328 gold rows with replies. A curated subset only (§5.2).

## 4. Suite A — deterministic scenarios

### 4.1 The pattern (already established, being extended)

`tests/test_reply_e2e.py` solved the hard problem: with a **reduced fixture catalog** and loosened
thresholds, `FakeEmbedder` routes correctly, so the *real* `Pipeline.route()` can be driven end to end
with no model and asserted against exact strings.

```python
FIX = Path(__file__).parent / "fixtures" / "catalog"

def _pipeline():
    cards = load_catalog(FIX)
    cfg = Config.default()
    cfg.thresholds = Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)
    return Pipeline(cards, FakeEmbedder(256), Scorer(cfg.weights, cfg.domain_keywords),
                    ConfidenceGate(cfg.thresholds), cfg)
```

The fixture catalog holds three cards, and they are sufficient for almost the whole failure taxonomy:

| card | params | provokes |
|---|---|---|
| `set_temperature` | `temperature` number **required** 16–32; `position` enum | `out_of_range`, `missing_required` (a *named* param) |
| `set_fan_speed` | `level` integer **required** 1–7; `position` enum | `out_of_range` on an integer |
| `open_window` | `is_open` boolean **required**; `position` enum | `missing_required` (an *unnamed* param), boolean-polarity confirmation |

`is_open` matters: it is one of the 14 required parameter names the `_CLARIFY` question bank does not
know, so it exercises the "请补充更多信息。" fallback path without needing a new card.

**No fixture-catalog changes are required.** Cases that cannot be provoked from natural text — a
`bad_enum` or `type_mismatch`, which the deterministic extractors never produce — are driven through
the LLM path with a scripted `FakeLLMClient`, which already accepts
`{clause_substring: LLMResult}` and can emit an arbitrary or invalid tool call, or `__reject__`.

### 4.2 New test doubles

Two are needed; neither exists today, because `MockExecutor` is the only executor in the repo and it
always succeeds.

```python
class RecordingExecutor:
    """Records dispatched calls so a case can assert WHAT was actuated, not just what was said."""
    def __init__(self): self.calls: list[ToolCall] = []
    def execute(self, tool_call): self.calls.append(tool_call); return {"ok": True, ...}

class FailingExecutor:
    """Reports a vehicle-side failure. The only way to reach requirement 4b's 'the car refused'."""
    def __init__(self, error="device_unavailable"): self.error = error
    def execute(self, tool_call): return {"ok": False, "error": self.error}
```

They live in `tests/e2e/doubles.py`. `RecordingExecutor` is immediately useful; `FailingExecutor` is
what makes the currently-untestable branch expressible, and the cases that use it are red until gap 1
(thread the executor result) is fixed.

### 4.3 Green / red separation

Suite A2 cases use **`@pytest.mark.xfail(strict=True)`** rather than a custom marker.

Rationale: `strict=True` means a case that *starts passing* becomes a **failure**, forcing whoever
closed the gap to promote it into A1. The suite detects its own progress instead of relying on someone
remembering to run `pytest -m spec`. They stay in the default run (reported as `xfailed`, not as
breakage), so `pytest -q` remains a usable merge gate.

Each A2 case carries `reason=` naming the gap-register item it blocks on, e.g.
`reason="gap 2: reply.py never reads validation_errors"`.

*Alternative considered:* a custom `spec` marker excluded via `addopts`. Rejected — it makes the red
set invisible by default and silent when a gap closes, which is the exact failure mode of a to-do list
kept in a document.

### 4.4 File layout

```
tests/e2e/
  doubles.py            # RecordingExecutor, FailingExecutor
  conftest.py           # the _pipeline() factory, parameterised by executor / llm_client / state
  test_s2_recognition.py
  test_s3_execution.py
  test_s4a_confirmation.py
  test_s4b_failure_cause.py   # majority xfail(strict=True)
```

The existing `tests/test_reply_e2e.py` stays where it is (see §8).

## 5. Suite B — graded eval slices

### 5.1 The contamination constraint — new rows go in a separate file

Auditing `eval/metrics.py` for type filters gives a result that dictates the design:

| filtered by row type (safe to extend) | **not** filtered (contaminated by any new row) |
|---|---|
| `recall_at_k`, `multi_intent_set_recall`, `e2e_executable_accuracy`, `ood_false_execution_rate`, `context_false_action_rate`, `avg_llm_calls`, `candidate_gen_recall`, `coverage` | `param_exact_match`, `schema_valid_rate`, `incorrect_execution_rate`, `clarification_rate`, `json_valid_rate`, all four `reply_*` metrics, and the latency percentiles |

So **appending new rows to `gold.jsonl` would silently move eight reported metrics plus P50/P95**, and
every number in `RESULTS.md` would stop being comparable to its own history — the precise failure
Spec 5 went to the trouble of proving it had avoided.

**Decision:** new rows live in **`data/eval/e2e_cases.jsonl`**, loaded and scored alongside gold
exactly as `context_negatives.jsonl` already is (`run_eval.py` builds a second `ctx_records` list and
reports `n_context_rows` next to its metric). `gold.jsonl` is not touched. Consequences: every existing
number stays valid, the new slices report with their own explicit denominators, and the precedent for
"a second scored file" is one this harness already implements.

### 5.2 Row schema

Existing fields are unchanged. New rows use:

```jsonc
{
  "utterance": "把温度调到99度",
  "type": "invalid",                    // new type
  "expected_functions": ["set_temperature"],   // recognition should still succeed
  "expected_execution": false,                 // but nothing may be dispatched
  "expected_cause": "out_of_range",            // the cause 4b must explain
  "expected_reply": "温度只能设置在16到32度之间。"  // curated subset only
}
```

| new type | meaning | required fields |
|---|---|---|
| `invalid` | names a supported function with an unusable value | `expected_functions`, `expected_execution: false`, `expected_cause` |
| `asr_noise` | a perturbation of an existing gold utterance | `expected_functions`, `source_utterance` |

**Hard rule, enforced by the validator: new-type rows must not carry `expected_params`.**
`param_exact_match` is not type-filtered, so an `expected_params` on a new row would contaminate it
even from a separate file. This is the one place where the separate-file decision is not enough on its
own.

`expected_reply` is optional on any row and drives the new metric. Target: **~30 curated
annotations** spanning single confirmation, joined multi-confirmation, question, and each failure
cause — not all 328 rows. This deliberately revisits Spec 5's "Rejected: gold reply annotations",
whose stated objection was annotation cost and brittleness across the full set; a curated subset
carries neither at that scale.

### 5.3 New metrics

Three, all registered in `run_eval.py` beside the existing `reply_*` block. `eval/arms.py::predict`
already exposes `reply`, `responses`, `questions`, `val_errors` and `row`, so **no plumbing changes
are needed** — these are pure additions to `eval/metrics.py`.

| metric | definition | want | empty denominator | can it fail today? |
|---|---|---|---|---|
| `invalid_no_execution_rate` | of `type:"invalid"` rows, fraction where nothing was dispatched | 1.0 | 1.0 | **yes** — a genuine safety metric |
| `reply_exact_match` | over rows carrying `expected_reply`, fraction where `reply` matches exactly | 1.0 | 1.0 | **yes** |
| `n_reply_annotated` | count of rows carrying `expected_reply` | — | — | n/a |

`n_reply_annotated` is not decoration. `reply_action_coverage` shipped reading 1.0000 while being
incapable of failing, and the only reason we know that is a paragraph in `RESULTS.md`. **Every
subset-scoped metric here reports its denominator next to it**, so a 1.0 over zero rows is visibly
vacuous rather than quietly reassuring.

Empty-denominator polarity follows the established convention: want-1.0 metrics return 1.0, want-0
metrics return 0.0 (`schema_valid_rate` vs `reply_question_drop_rate`).

### 5.4 Validator changes

`eval/dataset.py::validate_against_catalog` gains:

- `invalid` rows must carry `expected_cause` and `expected_execution: false`
- `asr_noise` rows must carry `source_utterance`
- **no row of type `invalid` or `asr_noise` may carry `expected_params`** (§5.2)
- `expected_cause`, where present, must be one of the ten known cause codes: the eight from
  `t2f/validate.py` (`not_in_candidates`, `unknown_function`, `unknown_param`, `missing_required`,
  `type_mismatch`, `out_of_range`, `bad_enum`, `llm_no_toolcall`) plus the two from `t2f/state.py`
  (`no_numeric_param`, `missing_state`), plus `executor_failed` once gap 1 introduces it

## 6. Case taxonomy

Every case carries an ID so coverage maps onto the workflow rather than onto our modules. Counts are
targets, to be adjusted during planning if a case turns out to be unprovokable.

### S2 — segmented intent recognition (~14 cases, all A1 green)

| ID | Case |
|---|---|
| S2-01..03 | single intent; explicit value; positional variant |
| S2-04..06 | multi-intent, two and three actions; delimiter and conjunction forms |
| S2-07..08 | context + action (`我有点热，温度调到25度`) — narration suppressed, absent from reply |
| S2-09 | relative operation (`温度调高一点`) |
| S2-10 | fraction (`天窗开到一半` → `percent=50`) |
| S2-11 | **negation / polarity** (`别关车窗`) — asserts the currently-derived polarity, locking in today's behaviour so the known defect can't drift further |
| S2-12 | ambiguous → clarification, no execution |
| S2-13 | OOD chitchat → refusal, no execution |
| S2-14 | OOD in-car-but-unsupported → refusal (distinct from chitchat) |

### S3 — execution (~9 cases; 2 red)

| ID | Case | Polarity |
|---|---|---|
| S3-01 | valid single → exactly one dispatch, asserted via `RecordingExecutor` | green |
| S3-02 | multi-intent, all valid → barrier passes, both dispatched, in order | green |
| S3-03 | **barrier partial** — one action invalid → **only the valid subset dispatches** | green |
| S3-04 | barrier partial → the invalid action is still named in the reply | green |
| S3-05 | relative op resolved against seeded state | green |
| S3-06 | relative op with **no** state → `missing_state`, nothing dispatched | green |
| S3-07 | LLM abstains (`__reject__`) → nothing dispatched | green |
| S3-08 | **`FailingExecutor` → the reply must not claim success** | **red — gap 1** |
| S3-09 | **`FailingExecutor` in a multi-action plan → state must not be committed for the failed action** | **red — gap 1** |

S3-08 is the single most important case in the suite. It is the one that will catch the latent
false-confirmation bug the moment a real vehicle adapter is attached.

### S4a — success confirmation (~7 cases; 1 red)

| ID | Case | Polarity |
|---|---|---|
| S4A-01..02 | single confirmation; value stated (temperature, level) | green |
| S4A-03 | two confirmations sentence-joined, no duplication | green |
| S4A-04 | exact-duplicate confirmations collapse to one | green |
| S4A-05 | confirmation names the position it acted on | green |
| S4A-06 | zero-parameter function → fixed sentence | green |
| S4A-07 | **boolean action states on-or-off** (`开车窗` vs `关车窗` must not produce the same sentence) | **red — gap 4** |

S4A-07 is the 43-of-92-cards defect: today both render `已为您调整{position}车窗状态。`

### S4b — failure cause (11 cases; 10 red)

Each asserts that the spoken reply conveys the *specific* cause. Today all but the first produce
`抱歉，这个操作没能完成。`

| ID | Cause | Provoked by | Polarity |
|---|---|---|---|
| S4B-01 | `missing_required`, *named* param | `把温度调一下` (no value) | green — the one path that works |
| S4B-02 | `missing_required`, *unnamed* param (`is_open`) | bare `车窗` | **red — gap 3** |
| S4B-03 | `out_of_range` above max | `把温度调到99度` | **red — gap 2** |
| S4B-04 | `out_of_range` below min | `把温度调到5度` | **red — gap 2** |
| S4B-05 | `out_of_range` on an integer | `风速调到20档` | **red — gap 2** |
| S4B-06 | `bad_enum` | scripted `FakeLLMClient` | **red — gap 2** |
| S4B-07 | `type_mismatch` | scripted `FakeLLMClient` | **red — gap 2** |
| S4B-08 | `missing_state` | relative op, empty state, plan path | **red — gap 2** |
| S4B-09 | `llm_no_toolcall` | `FakeLLMClient` returning nothing | **red — gap 2** |
| S4B-10 | `executor_failed` | `FailingExecutor` | **red — gaps 1+2** |

Plus one cross-cutting consistency case, currently red:

| ID | Case | Polarity |
|---|---|---|
| S4B-11 | the **same invalid value** produces the same *kind* of reply on the legacy path and the plan path (today: a failure line on one, a clarification request on the other) | **red — gap 2** |

### Suite B slices (~50 new rows)

| slice | rows | purpose |
|---|---|---|
| `invalid` | ~20 | the failure taxonomy, currently absent entirely; scored by `invalid_no_execution_rate` |
| `asr_noise` | ~20 | homophone, filler, dropped-particle perturbations of existing gold utterances; scored as its own recall slice — gap 8 |
| relative-with-state | ~10 | only 6 of 328 gold rows carry `vehicle_state` |
| `expected_reply` annotations | ~30 | spread across the above and existing types |

## 7. What starts red, and what that buys

| Red cases | Gap | What closing it requires |
|---|---|---|
| S3-08, S3-09, S4B-10 | 1 | an `Executor` protocol returning a result; thread it into `ClauseResult` / `PlannedAction`; gate `render_response` on it |
| S4B-02 | 3 | drive the question from `ParamSpec.description`, already parsed and unused |
| S4B-03..09, S4B-11 | 2 | a `ValidationError.code` → Chinese phrase table read by `t2f/reply.py` |
| S4A-07 | 4 | boolean/enum value rendering in `render_response` |

**13 red cases out of 41 in Suite A** (S2 14 + S3 9 + S4a 7 + S4b 11). That ratio is the honest
current answer to "how much of step 3 and step 4 do we meet", and after this lands it is a number a
run prints rather than a claim a document makes.

## 8. Migration of existing tests

`tests/test_reply_e2e.py` (11 tests) already follows the Suite A pattern and all 11 pass. They are
**left in place, not moved**. Moving them would produce a large diff with no behavioural meaning and
would break the `RESULTS.md` Spec-5 test-count record. Instead the new `tests/e2e/` files carry a
header noting that `test_reply_e2e.py` holds the original Spec-5 reply cases, and any *new* case is
written in `tests/e2e/`.

`tests/test_reply_golden.py` (9) is genuinely unit-level — it composes from synthetic `RouteResult`s —
and is out of scope here.

Suite A must not duplicate an existing assertion. During planning, each proposed case is checked
against the 11 existing e2e tests; overlaps are dropped rather than restated.

## 9. Risks and trade-offs

**Exact-reply assertions are brittle.** Any template rewording breaks a batch at once. Accepted
deliberately: the entire product output is a spoken string, so a silent wording regression is
precisely what should fail loudly. Mitigation: expected strings are defined as module-level constants
(the pattern `test_reply_e2e.py` already uses with `WINDOW` / `TEMP25`), so a template change is a
one-line test edit, not a scatter of literals.

**The fixture catalog is not the real catalog.** Suite A proves the *mechanism* (a cause is explained,
a failure is not confirmed) on three cards. It does not prove the 92-card catalog is well-formed —
that is Suite B's job. Neither suite alone is sufficient, which is why both were chosen.

**`xfail(strict=True)` reads as green in CI.** A reviewer skimming "208 passed" will not see 14 xfails.
Mitigation: the coverage table in §7 is mirrored into `RESULTS.md` when this lands, and the red count
is stated in the README status line.

**Suite B needs real models to run.** It cannot gate a commit. That is inherent — accuracy over a
distribution is not measurable with a fake embedder — and is why Suite A exists.

## 10. Success criteria

1. `pytest -q` passes with the new suite present, reporting 13 `xfailed`.
2. Running the eval on `gold.jsonl` produces **numbers identical to the current `RESULTS.md` to four
   decimals** — proving Suite B added no contamination. Same discipline Spec 5 used to prove the reply
   layer was presentational.
3. `invalid_no_execution_rate` reports 1.0 on both arms (nothing dispatches on an unusable value), and
   `reply_exact_match` reports a real number next to a non-zero `n_reply_annotated`.
4. Every case ID in §6 maps to exactly one test or row, and every red case names its gap.
5. Closing any gap in §7 turns its cases from `xfailed` to failing-because-they-now-pass, which is the
   signal to promote them.

## 11. Open question for the plan stage

`asr_noise` perturbations must be authored by hand: we have no ASR to generate real
misrecognitions, so the rows will encode *our belief* about what an ASR gets wrong. That is weaker
evidence than the rest of the suite and should be labelled as such in `RESULTS.md` rather than
reported beside measured numbers as though it carried equal weight.
