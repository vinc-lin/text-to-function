# Spec 4 — Multi-Intent, Context-Aware Routing

Status: design approved 2026-07-24. Builds on Spec 1 (deterministic router + eval), Spec 2
(constrained-LLM fallback + classifier + multi-turn), Spec 3 (learned confidence gate + safety
hardening). On-device SA8797/GGUF port remains deferred.

## 0. Motivation

A single utterance can bundle **narration + several commands**, mixing context, absolute values,
and relative nudges. The canonical case:

> 后排小孩老去按车窗，把车窗锁打开。然后主驾这边窗户再开一点，天窗开到一半。

Correct behavior is **three** executed actions plus **one** non-executed context clause:

| Span | Correct outcome |
|---|---|
| 后排小孩老去按车窗 | CONTEXT — explains *why*; must produce **no** action |
| 把车窗锁打开 | `set_window_child_lock {enabled: true}` |
| 主驾这边窗户再开一点 | `set_window_position {position: driver, percent: current+step, clamped}` |
| 天窗开到一半 | `set_sunroof_position {percent: 50}` |

### What the current system does (measured, not assumed)

A probe on the real Qwen3-Embedding-0.6B pipeline (`config.yaml`, OOD prototypes loaded) shows:

1. **The whole utterance is MEDIUM band** — every segment. The deterministic HIGH path never
   fires; this case is entirely LLM/clarification territory.
2. `segment.py` splits `后排小孩老去按车窗` into its own clause and routes it to
   `open_window (0.766)`. The context clause becomes a spurious action/clarification.
3. `再开一点` parses to `operation=None` and maps to `set_window_position` (which requires an
   absolute `percent` it cannot supply) → fails/clarifies. **No relative representation, no state.**
4. `天窗开到一半`: `一半` is **not** parsed to `percent=50`; `open_sunroof (0.736)` narrowly beats
   the correct `set_sunroof_position (0.730)`.
5. Execution is **eager and per-clause** — there is no "validate the whole plan, then execute"
   barrier, and **all 312 gold cases are `type:"single"`**: multi-intent has never been evaluated.

Polarity is *not* a problem: `把车窗锁打开` already retrieves `set_window_child_lock (0.819)` cleanly
above `open_window (0.779)`.

### Key design finding — context detection (experiment-driven)

We tested three ways to decide "this span is context, don't act on it" against a battery of 8
context spans and 8 action spans on the real pipeline:

| Approach | Context-correct | Action-correct | Verdict |
|---|---|---|---|
| **A** — reuse confidence/OOD gate (LOW = context) | **0.12** | 1.00 | **Rejected** |
| **B** — lexical actionability rule (target-alias + op/value cue) | **1.00** | ~0.88 → ~1.0 with fuller op lexicon | **Adopted** |
| **C** — hand the full utterance to the LLM | n/a (needs LLM anyway) | — | Adopted for the residue |

**Why A fails:** context clauses are *topically in-domain* narration, and the embedder is a
topical-similarity model. `副驾说有点热` sits right next to `set_temperature (0.784, ood 0.216)`;
`后备箱东西很多` next to `open_trunk (0.713)`. OOD negatives don't fire because narration is not
out-of-domain. **The gate cannot separate narration from command; they are topically identical.**
Only 1/8 context spans (`外面在下雨`) went OOD.

**Adopted direction:** a lexical **actionability filter** (Approach B) as the always-on deterministic
context suppressor, escalating multi-intent/medium spans to a **single Qwen3-0.6B plan call**
(Approach C) for context-driven disambiguation and multi-action planning. The OOD gate is retained
for genuinely out-of-scope input, which is what it is actually good at.

### Environment (validated 2026-07-23)
Python 3.10.12; torch 2.11.0+cu130 on an RTX 4060 Ti (CUDA available); transformers,
scikit-learn, joblib, xgrammar, sentence-transformers all installed; `pip` works. Qwen3-0.6B and
Qwen3-Embedding-0.6B cached. `pytest -q` = 117 passed; `pytest -q -m model` = 2 passed. The real
embedder and the real xgrammar-constrained toolcall both run end-to-end here.

## 1. Objective

Extend the router to handle multi-intent, context-aware utterances **without regressing
single-intent behavior and without new false executions**. Concretely:

1. **Suppress context** — narration spans produce no action (target ≈1.0 suppression, ~0 false action).
2. **Plan-then-execute** — assemble and validate the *whole* plan before any execution.
3. **Relative + state** — represent relative operations in the plan; resolve to absolute tool calls
   via an **injectable mock** vehicle-state store (real vehicle APIs stay deferred, per scope).
4. **Partial failure** — execute the valid subset; raise **one** consolidated clarification for the rest.
5. **Evaluate it** — a multi-intent gold set + context negatives + new metrics, with single-intent
   metrics as a hard regression gate.

### Scope (decided)
One Spec 4. The vehicle-state layer is implemented against an **injectable mock** (consistent with
`MockExecutor` and gold/silver eval). Real vehicle-state APIs, real relative vehicle APIs, and the
on-device SA8797/GGUF runtime remain **deferred**.

### PRD requirement coverage (this spec)
- Multi-intent decomposition with context attribution.
- Relative/stateful control (`再开一点`) via state resolution, never LLM-guessed values.
- Child-lock polarity robustness (target vs operation vs context).
- Abstention / no-hallucinated-call preserved under multi-intent (safety-first).

## 2. Enhanced control flow

Current `route()` is a per-clause eager loop. Spec 4 inserts a **plan barrier** between
understanding and execution:

```
route(utterance):
  1. SEGMENT      normalize → split (existing punctuation/conjunction tokenizer)
                  → label each span ACTION | CONTEXT | CONNECTOR   (actionability filter)
                  → attach CONTEXT spans to their neighbor ACTION (carried, never routed)
  2. RETRIEVE     per ACTION span: encode → retrieve → score → gate   (Spec 1-3 internals unchanged)
  3. PLAN PATH    all ACTION spans HIGH & unambiguous & no unresolved relative
                     → deterministic per-action tool calls
                  otherwise (any MEDIUM / multi-intent / context-disambiguation / relative)
                     → ONE Qwen3-0.6B plan call: full utterance + per-span candidate cards,
                       thinking disabled, xgrammar-constrained multi-action JSON,
                       {"name":"__reject__"} per span for context/out-of-scope
  4. RESOLVE      per PlannedAction: if relative → StateResolver(mock state) → absolute ToolCall
  5. VALIDATE-ALL validate every action against its card; NOTHING executes yet    ← the barrier
  6. EXECUTE      run the valid subset in order; write results back to state;
                  collect invalid/ambiguous/missing-state → ONE consolidated clarification
  return RouteResult(plan, executed[], clarification?)
```

Relative actions resolve against the **plan-start state snapshot** (deterministic); the executor
writes confirmed results back for *future* turns, not mid-plan.

## 3. New & changed modules

### `t2f/actionability.py` — new (the context suppressor)
`role(span, feats, cards) -> SpanRole`. A span is **ACTION** iff it contains a target alias
(from card aliases) **and** an operation/polarity/value cue (open/close/set/adjust lexicon, on_off,
percent, level, or a relative cue). Otherwise **CONTEXT** (attached to a neighbor) or **CONNECTOR**
(pure conjunction residue). Fails safe: implied-desire clauses (`我有点冷`) → CONTEXT, never a silent
action. Depends on: card aliases, `lexical`.

### `t2f/segment.py` — changed
Keep the existing punctuation/conjunction split as the tokenizer. Add role labeling + context
attachment on top, returning `list[Span]` instead of `list[str]`. A back-compat helper preserves the
old `split() -> list[str]` for single-intent callers/tests.

### `t2f/state.py` — new (the mock state layer + resolver)
- `VehicleState`: injectable mock store, `get(key)/set(key, value)`, reset-per-utterance.
  **State priority: live (injected) > last-confirmed (executor write-back) > session default.**
- `StateResolver.resolve(action, state) -> ToolCall`: read current value → apply `step × amount`
  in the `operation` direction → **clamp to the card's min/max** → emit the **absolute** tool call.
  Missing state with no default → mark `clarify` (never guess). Depends on: `config.relative_steps`,
  card limits.

### `t2f/plan.py` — new (the barrier)
`ActionPlan` / `PlannedAction` dataclasses + the assemble → validate-all → execute-valid
orchestration. Enforces "no execution until the whole plan is validated," then executes the valid
subset in order and produces one consolidated `ClarificationRequest` (reusing multi-turn
`PendingState`) for the remainder. Depends on: `validate`, `state`, `execute`.

### `t2f/llm/plan_prompt.py` + `t2f/llm/schema.py` — new / extended
Multi-action plan prompt (full utterance + per-span candidate cards, thinking disabled) and an
xgrammar schema `{"actions": [ {name ∈ offered ∪ __reject__, parameters, relative?} ]}`. Reuses the
Spec-2 constrained-decoding path; falls back to strict-parse + one repair when constrained decoding
is unavailable.

### `t2f/params/numerals.py`, `t2f/params/extractors.py` — changed
- Parse fractions: `一半`→50, `三分之一`→33, etc. (fills `percent`; breaks the sunroof near-tie).
- Parse relative cues `再…一点` / `多开点` / `调小一点` / `大一点` → `operation ∈ {increase, decrease}`
  + `amount ∈ {small, medium, large}` (currently `None`).

### `t2f/pipeline.py` — changed
Rewire `route()` to the 6-stage flow; a `PlanResolver` replaces eager per-clause execution while
delegating single-intent, single-action cases to the existing deterministic/LLM resolvers.

### `t2f/types.py` — changed
Add `SpanRole`, `Span`, `RelativeSpec`, `PlannedAction`, `ActionPlan`; extend `RouteResult` with
`plan: Optional[ActionPlan]` (keep `clauses[]` for single-intent back-compat).

### `config.yaml` — changed
Add `relative_steps:` — `default_percent: 10`, per-unit overrides (`celsius: 1`, `level: 1`),
`amount` multiplier (small×1 / medium×2 / large×3). No catalog **function** changes.

## 4. Data

- `data/eval/gold.jsonl`: add `type:"multi"` records with canonical `expected_actions:
  [{function, params}]` (ordered), `context_clauses: [...]` (must yield no action), and optional
  `vehicle_state: {…}` seeding relatives. Single-intent records keep their existing
  `expected_functions`/`expected_params` shape; readers stay back-compatible.
- `data/eval/context_negatives.jsonl` (new): pure narration / implied-desire utterances → expect
  **zero** actions (the probe's context battery + more).
- Target volume: ~40–60 multi-intent cases (2–4 actions each, including relative + polarity + one
  context clause) + ~30 context negatives; dev/test split matching gold. Authored per
  `data/gen/generate_notes.md`.
- `data/catalog/`: add a small polarity-confusable set + relative utterances to `window.yaml`
  (child-lock vs window vs sunroof) as prototypes/hard-negatives; no function changes.

## 5. Eval — new axis + regression gate

`eval/metrics.py` gains:

| Metric | Definition | Role |
|---|---|---|
| Plan exact-match | produced actions == `expected_actions` as an **unordered set** of (fn, params); execution runs in sequence but order is not scored | headline |
| Per-action correctness | fraction of expected actions produced correctly | diagnostic |
| Context-suppression rate | context_clauses producing no action | **safety, target ≈1.0** |
| False-action-on-context | context spans that executed | **safety, hard gate ≈0** |
| Spurious / missing-action rate | over- / under-generation vs expected | diagnostic |
| Relative-resolution accuracy | resolved absolute == current + step (seeded state) | relative correctness |
| Polarity accuracy | child-lock-vs-window confusable set routed correctly | robustness |

All **existing** metrics (recall@1/@3, schema-valid, param-match, incorrect-exec, OOD false-exec,
avg-LLM-calls, P95 latency) are computed on the **single-intent split as a regression gate**.

`eval/arms.py` / `eval/run_eval.py`: `predict()` seeds the mock `VehicleState` from
`row["vehicle_state"]` (reset per row), then scores the produced `ActionPlan` against
`expected_actions` + `context_clauses`. The plan path is exercised through the existing `C_llm` / `D`
arms with the new `PlanResolver`.

### Acceptance targets
- **Hard gates:** false-action-on-context ≈ 0; single-intent regression = 0 (existing 117 tests +
  single-split metrics unchanged); schema-valid on the plan call ≥ Spec-2 level (~0.99).
- **Context-suppression rate ≥ 0.95** on context negatives.
- **Plan exact-match / per-action correctness:** establish a measured baseline first, then set an
  improvement target. No invented number precedes measurement.

## 6. Testing (TDD, model-free where possible)
- Unit: actionability filter (context battery → all CONTEXT, action battery → all ACTION);
  `StateResolver` (relative→absolute, clamp at min/max, missing-state→clarify, priority order);
  plan barrier (partial failure → execute valid + exactly one clarification; all-valid → all execute);
  numerals (`一半`→50, `三分之一`→33); relative-cue parsing.
- Integration (`@model`): the canonical utterance → exactly the 3-action plan with the context clause
  suppressed; multi-intent eval on the dev split; polarity-confusable set.

## 7. Non-goals (explicit)
- Real vehicle-state API and real relative vehicle APIs — deferred (mock only).
- On-device SA8797 / GGUF / Hexagon runtime — deferred.
- Reordering / dependency planning across actions beyond sequential execution.
- Cross-turn plan memory beyond the existing `PendingState` multi-turn mechanism.

## 8. Key risks & mitigations
- **Actionability lexicon brittleness on paraphrase.** Mitigation: derive target aliases from cards
  (not a hand list); keep the op/value lexicon small and value/polarity-anchored; the LLM plan call
  is the safety net for the medium residue; measure on context negatives.
- **LLM over-generates actions from context.** Mitigation: per-span `__reject__` escape + validate-all
  barrier + false-action-on-context hard gate; the actionability filter strips most context before
  the LLM sees it.
- **Relative resolution wrong when state is stale/absent.** Mitigation: explicit state priority;
  missing state → clarify, never guess; resolve against a single plan-start snapshot for determinism.
- **Multi-intent changes regress single-intent.** Mitigation: single-intent path preserved via
  back-compat `split()`/`clauses[]`; single-split metrics are a hard regression gate.
- **Sunroof/window near-ties.** Mitigation: `一半`→percent parsing + param-compat signal break the
  tie toward the position-setter; covered by the polarity/tie eval subset.
