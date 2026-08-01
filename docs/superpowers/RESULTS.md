> **Read this first:** these are per-spec *measured* results, **recorded when each spec shipped and
> left as written** — a section here says what was true then, not what is true now. Where a later
> spec moved an earlier number, the later section says so. **Current** measured figures live in
> [`../TEST_REPORT.md`](../TEST_REPORT.md), which is the only place they are maintained; for what
> the system covers against the Central Model business workflow, see
> [`2026-07-25-central-model-system-design.md`](specs/2026-07-25-central-model-system-design.md).
> **No number anywhere below was measured on the 87 platform.**
>
> The last section is not a spec. Sensed signals, staleness and `intake/` landed after Spec 9 and are
> recorded here in the same form.

# Spec 1 — Evaluation Results

**Date:** 2026-07-21
**Embedder:** `Qwen/Qwen3-Embedding-0.6B` (FP16, transformers, GPU), MRL dim 512, last-token pooling
**Dataset:** `data/eval/gold.jsonl` — 312 hand-verified rows; **calibrated on the dev split (128), reported on the test split (184)**
**Catalog:** 92 functions across 10 domains
**Fusion weights:** `embedding 0.88 · keyword_alias 0.04 · param_compat 0.05 · domain_prior 0.03`
**Gate thresholds (dev-calibrated):** `high_top1 0.35 · high_margin 0.12 · low_top1 0.15`

> Latency below is a **dev-machine (x86 + discrete GPU, CUDA FP16) number, not SA8797**. No memory
> figure is reported anywhere in this document, or measured anywhere in the repo — `psutil` is a
> declared dependency that nothing imports. On-device latency/memory/crash benchmarking was assigned
> to the SA8797 port, which remains **deferred pending hardware** (Spec 3 shipped as accuracy & safety
> hardening instead). The `<1500 ms` target below is a self-set engineering inference, not an
> 87-platform standard.

## Headline metrics (test split, n=184)

| metric | Arm C (hybrid) | Arm baseline (pure-emb) | target | met? |
|---|---|---|---|---|
| Function Recall@1 | **0.822** | 0.805 | ≥ 0.90 | ✗ |
| Function Recall@3 | **0.907** | 0.907 | ≥ 0.97 | ✗ |
| Multi-intent set recall | 0.929 | 0.946 | — | — |
| Parameter exact-match | 0.356 | 0.351 | — | ✗ (see below) |
| OOD false-execution rate | **0.000** | 0.000 | ≈ 0 | ✓ |
| Incorrect-execution rate | **0.039** | 0.048 | ≈ 0 | ~ |
| Schema-valid rate (all assembled calls) | 0.498 | 0.500 | ≥ 0.99 | ✗ (see below) |
| e2e executable (deterministic-only) | 0.070 | 0.056 | ≥ 0.80 | ✗ (by design) |
| e2e executable (LLM-ceiling) | **0.845** | 0.845 | — | — |
| Avg LLM calls / single-intent req | 1.16 | 1.15 | ≤ 0.5 | ✗ |
| P50 / P95 latency (ms) | 52 / 72 | 53 / 72 | < 1500 | ✓ |

Fake-embedder Arm C (harness sanity check, no model): recall@1 0.337 / recall@3 0.397 — confirms the pipeline and every metric compute correctly end-to-end; the gap to the real embedder is the embedder's contribution.

**On schema-valid rate:** every call the router *executes* is valid by construction — the deterministic path validates before executing and never executes an invalid call (this is the safety guarantee behind incorrect-execution ≈ 0). The 0.50 figure is over *all assembled candidate calls including the un-executed medium band*, and reflects that about half the medium-band calls still lack a required parameter — exactly the residual that Spec-2's LLM param-completion is meant to fill. The ≥0.99 target properly applies to Spec-2's LLM output.

## What the numbers mean

**Retrieval-first is validated, and the hybrid helps (barely).** Qwen3-Embedding-0.6B alone gets 0.805 recall@1; the embedding-dominant hybrid lifts it to 0.822 without regressing recall@3. This is the workhorse of the system.

**The coarse lexical signals do not beat a strong embedder.** A dev-set weight sweep drove `keyword_alias` and `param_compat` toward zero: because `param_compat` rewards *every* function that shares a parameter type (e.g. all temperature-taking functions score 1.0 on "调到25度"), it is not discriminative and, at the plan's original weights (0.25), actively *hurt* recall@1 (0.771 < 0.805 baseline). We therefore made the fusion embedding-dominant. The signals are retained (small weights) because they still feed the gate's `param_compat` feature and would help a weaker on-device backend. **Lever:** redesign `param_compat` to give *negative* evidence (penalize functions whose required param types are absent) rather than uniform positive credit.

**Safety holds.** OOD false-execution is 0.000 and incorrect-execution is 0.039 — the system almost never executes the wrong function, which was priority #1.

**The deterministic/LLM split is the central Spec-1 result.** To keep the HIGH band ≥98% precise (for near-zero incorrect execution), calibration must require a large top1−top2 margin (0.12). On confusable clusters (set_temperature vs set_fan_speed vs set_seat_heating; set_volume vs set_fan_speed) that margin is small, so only ~7% of traffic clears the HIGH bar and the rest defers to the medium band. Hence:
- `e2e_deterministic` (executed clauses only) = 0.07 — this is *not* the system's accuracy; it is the fraction resolvable with **zero** LLM at ~98% precision.
- `e2e_ceiling` = **0.845** — the fraction the Spec-2 LLM can execute correctly by picking the function from the top-3 it is handed (and completing missing params). This is the realistic end-to-end number once the fallback exists, and it comfortably clears the 0.80 target.
- `avg_llm_calls` ≈ 1.16 > 0.5 — a direct consequence: most requests route to the LLM. (It exceeds 1.0 because the conservative segmenter splits some comma/conjunction-containing "single"-labeled utterances into 2 clauses; `avg_llm_calls` is a per-request upper bound.)

## Gaps vs targets and the levers to close them

| Gap | Root cause | Lever (spec) |
|---|---|---|
| Recall@1 0.82 vs 0.90; Recall@3 0.91 vs 0.97 | 0.6B embedder ceiling on a hard, anti-leakage, colloquial, 92-function set with confusable clusters | More/more-diverse colloquial prototypes per card; hard-negative-aware prototypes; **the Spec-2 supervised classifier (Arm D)** which learns to separate confusable clusters; optional embedder fine-tune |
| Deterministic coverage low → avg LLM calls > 0.5 | Precision-over-coverage gate needed for near-zero incorrect execution at 0.82 recall@1 | Raising recall@1 (above) widens the HIGH band at fixed precision; **Spec-2 LLM** resolves the medium band so the target shifts to "≤0.5 *without hurting accuracy*" |
| Parameter exact-match 0.36 | Strict full-dict equality; gold `expected_params` include values not always recoverable from a paraphrase; extractor gaps | Per-parameter F1 instead of exact-dict; extractor coverage for more phrasings; **Spec-2 LLM param completion** for the residual |
| avg_llm_calls > 1.0 artifact | Segmenter splits some "single"-labeled utterances containing commas/conjunctions | Refine multi-intent labels, or make the segmenter re-merge clauses that route to the same function |

## Known limitations (Spec 1, for Spec 2 to address)

- **Relative operations not yet parameterized.** `LexFeatures` detects increase/decrease/max/min (调高/调低/最大/最小), but no parameter extractor consumes them, so a command like "温度调高一点" finds no value and falls through to a clarification. A card modeling relative change (e.g. an `operation` enum or a signed `delta`) plus a matching extractor would close this.
- **Only position enums are extracted.** The schema-driven dispatcher maps units (celsius/percent/level) and position enums; a generic (non-position) enum or free `string` param currently falls to the numeric extractor and cannot be filled deterministically. A generic enum-value/alias matcher is the natural addition.
- **Segmenter can split a comma-preceded "single" utterance** into two clauses (e.g. "有点热，把温度调低"); each clause routes independently. This is correct behavior but inflates `avg_llm_calls` against single-labeled rows.

## Bottom line

Spec 1 delivers a working, tested (72 automated tests), fully-measured retrieval-first router. It **meets the safety and latency targets** (OOD/incorrect execution ≈ 0; P95 72 ms) and demonstrates the core thesis: strong retrieval + a calibrated gate can *safely* separate "execute now with no LLM" from "hand a tight top-3 to the LLM." The **LLM-ceiling e2e of 0.845 clears the ≥0.80 executable target**, confirming the retrieval+params foundation is strong enough for Spec 2 to build on. It **does not yet meet the recall@1/@3 and avg-LLM-call targets**, which require Spec 2 (LLM fallback + supervised classifier) and richer catalog/eval data — the levers are identified above, not hand-waved.

---

# Spec 2 — LLM Fallback + Classifier + Multi-Turn Results

**Date:** 2026-07-21
**Fallback LLM:** `Qwen/Qwen3-0.6B` (FP16, transformers, GPU), single-shot, **xgrammar JSON-schema-constrained** decoding
**Same** 92-function catalog, embedder, gold split (calibrate on dev 128, report on test 184), plus 96 OOD prototypes and a 53-row multi-turn `followups.jsonl`.

## Metrics (test split, n=184)

| metric | Spec 1 (Arm C) | Arm C+LLM | Arm D (clf+LLM) | target | note |
|---|---|---|---|---|---|
| Recall@1 / @3 | 0.822 / 0.907 | 0.814 / 0.907 | 0.814 / 0.907 | 0.90 / 0.97 | retrieval unchanged |
| candidate_gen_recall@3 | — | 0.907 | 0.907 | — | classifier union didn't lift the pool |
| multi-intent set recall | 0.929 | 0.929 | **0.964** | — | Arm D helps multi-intent |
| **JSON-valid rate (LLM)** | n/a | **~1.0** (0.83*) | ~1.0 (0.83*) | — | *xgrammar guarantees valid JSON; 0.83 counts LLM **rejects** as "not a tool-call" |
| **schema-valid rate** | 0.50 | **0.995** | 0.995 | ≥ 0.99 | ✅ constrained + validated |
| **param exact-match** | 0.356 | **0.633** | 0.608 | — | LLM completes params rules miss |
| **e2e executable (real, with LLM)** | 0.07 (det-only) | **0.458** | 0.458 | ≥ 0.80 | 6.5× over Spec-1 deterministic; below target (recall ceiling) |
| **multi-turn follow-up success** | n/a | **1.000** (n=46) | 1.000 | — | ✅ rules-first completion works |
| OOD false-execution | 0.000 | **0.321** | 0.321 | ≈ 0 | ✗ — see safety analysis |
| incorrect-execution | 0.039 | 0.344 | 0.337 | ≈ 0 | ✗ — same root cause |

*(All correctness/e2e/param metrics for the LLM arms are scored against the function the LLM actually **executed** — which it selects from the top-2/3 — not the retrieval top-1.)*
| avg LLM calls / single | 0.0 | 1.16 | 1.11 | ≤ 0.5 | most traffic uses the LLM |
| P50 / P95 latency (ms) | 52 / 72 | 835 / 1184 | 932 / 1390 | < 1500 | ✅ LLM in the loop, still < 1.5 s |

## What Spec 2 delivered
- **Guaranteed-valid tool calls.** xgrammar makes syntactically-invalid JSON impossible; combined with the reused Spec-1 validator, **schema-valid rate hits 0.995** (met the ≥0.99 target) — the constrained 0.6B never emits an out-of-schema call.
- **Parameter completion.** The LLM fills params the deterministic rules can't (relative operations, phrasings the extractors miss): **param exact-match 0.36 → 0.63**.
- **The medium band now resolves.** End-to-end executable accuracy went from **0.07 (Spec-1 zero-LLM) to 0.46** with the real LLM — a 6.5× lift, realizing much of the 0.845 retrieval ceiling.
- **Multi-turn works.** Rules-first follow-up completion resolves **100%** of the 46 in-scope clarification follow-ups.
- **Arm D** (classifier candidate union) lifted **multi-intent set recall 0.929 → 0.964** but did **not** raise single-intent candidate recall — consistent with Spec-1's finding that the strong embedder is hard to beat; the classifier partly replicates it (silver-leakage caveat).

## The safety cost, and how it was mitigated (the central Spec-2 finding)
Executing the medium band via a **constrained** LLM introduces a risk Spec 1 didn't have: the decoder is *forced* to emit one of the candidate calls, so an out-of-domain utterance that reaches MEDIUM is executed. Naively this drove **OOD false-execution to 1.0**. Three stacked, spec-aligned mechanisms brought it down to **0.32**:
1. **LLM reject escape hatch** — a `{"name":"__reject__"}` schema option + prompt instruction lets the model decline (~17% of medium-band calls decline; this is why the raw JSON-valid metric reads 0.83 rather than 1.0 — a reject is valid JSON, just not a tool-call).
2. **OOD negative prototypes** — 96 chitchat/unsupported prototypes seeded in the index; when the `__ood__` marker wins retrieval the gate rejects. Captures **57%** of test OOD at **0.8%** in-domain false-rejection.
3. **OOD-aware conservative gate calibration** — when the medium band executes, `low_top1` becomes an OOD floor tuned to keep OOD out of the executing bands (biasing toward rejection, per the PRD).

**Residual gap (honest):** OOD 0.32 and incorrect-execution 0.34 still miss "≈ 0". The residual is the hard tail of OOD/wrong-top-1 inputs whose retrieval score overlaps the in-domain range (in-domain median 0.74 vs OOD median 0.56, overlapping ~0.65–0.71), which a single threshold can't separate and a 0.6B can't reliably reject. **Levers to close it:** (a) more/broader OOD prototypes — capture scaled 46%→57% just by 54→96, and keeps scaling; (b) a dedicated binary in-domain/OOD detector (a natural extension of the Arm-D classifier); (c) a conservative operating point (raise `low_top1`, trading e2e for safety — the PRD's stated "clarify rather than incorrectly execute" preference); (d) a larger fallback (Qwen3-1.7B) whose reject/tool-calling is far more reliable (1.38%→16.88% multi-turn BFCL, per the Spec-1 research). For a production safety-critical deployment, (b)+(c) are recommended before trusting medium-band LLM execution.

## Bottom line
Spec 2 turns the retrieval router into an executing system: **guaranteed-valid, param-complete tool calls, 6× end-to-end executable accuracy, and flawless multi-turn completion**, all with the LLM in the loop under 1.5 s. Its honest cost is that executing the medium band reintroduces OOD/incorrect-execution risk; three principled mechanisms cut OOD false-execution 3× (1.0→0.32), and the remaining gap has clear, identified levers rather than hand-waving. Retrieval recall and the ≤0.5-LLM-call target remain Spec-1-bounded and need the classifier/data work already scoped.

---

# Spec 3 — Accuracy & Safety Hardening Results

**Date:** 2026-07-22
**Mechanism:** a learned **execution-confidence gate** (logistic regression over 11 cheap routing features → `P(top-1 correct)`), calibrated on gold-**dev** (+ OOD prototype queries), reported on gold-**test** (n=184). Replaces the hand-tuned score thresholds; abstains (clarifies) below a calibrated confidence.

> **Wiring caveat (added 2026-07-25).** `ConfidenceModelGate` and `ExecutionConfidence` are implemented,
> trained (`models/confidence.joblib`) and integration-tested, but **no eval arm constructs them** — all
> four builders in `eval/arms.py` hardcode the plain threshold `ConfidenceGate`, and the driver that
> produced the frontier below was never committed. Treat this section as a reproducible-in-principle
> research result, not as shipped behaviour. Spec 4's deterministic arm C subsequently reported
> *better* safety with the plain gate (OOD 0.000 / incorrect 0.031 / P95 73 ms).

## The safety/coverage frontier (learned gate, deterministic HIGH-only, **zero LLM**)

| τ_high | OOD false-exec | incorrect-exec | coverage |
|---|---|---|---|
| 0.4 | 0.357 | 0.060 | 0.712 |
| 0.5 | 0.321 | 0.051 | 0.669 |
| 0.6 | 0.214 | 0.058 | 0.585 |
| **0.7** | **0.107** | **0.067** | **0.508** |
| 0.8 | 0.107 | 0.070 | 0.364 |

The learned confidence separates cleanly and monotonically — every operating point is available by choosing τ. Incorrect-execution stays **0.05–0.07 across the whole curve** (the confident set is 93–95% correct).

## Two operating points vs Spec 2

| metric | Spec 2 (heuristic gate + LLM) | **Spec 3 safe (det. τ=0.7, no LLM)** | Spec 3 balanced (calibrated 2-band + LLM) |
|---|---|---|---|
| OOD false-execution | 0.321 | **0.107** | 0.321 |
| incorrect-execution | 0.344 | **0.067** | 0.252 |
| coverage (fully executed) | ~0.46 | 0.508 | 0.455 |
| e2e executable | 0.458 | 0.508* | 0.366 |
| avg LLM calls / single | 1.16 | **0.000** | **0.447** (≤0.5 ✅) |
| P95 latency (ms) | 1184 | **~275** | 1126 |

\* deterministic e2e = coverage here (every executed clause is a confident, validated call).

## Findings
- **The learned gate is a strict improvement in separation.** At the recommended **safe** point (deterministic, τ_high=0.7), OOD false-execution drops **3× (0.32→0.107)** and incorrect-execution **5× (0.34→0.067)** versus Spec 2, while still confidently executing **~half of all requests with zero LLM calls** at ~275 ms P95. This is the PRD-aligned "clarify rather than incorrectly execute" operating point.
- **Executing the medium band via the LLM is the OOD liability.** The balanced two-band point keeps a medium/LLM zone (τ_low=0.45, τ_high=0.79), which recovers coverage and cuts incorrect-execution to 0.252 and hits the **avg-LLM-calls ≤ 0.5 target (0.447)** — but OOD false-execution returns to 0.321, because the constrained LLM still executes most OOD it is handed (its reject option + OOD prototypes only partly help). This confirms the Spec-2 finding: the medium-band LLM, not retrieval, is where OOD leaks.
- **Recommendation:** for a safety-critical automotive deployment, run the confidence gate **deterministically at a high τ** (execute only high-confidence, clarify the rest), reserving the LLM for a *narrow* high-confidence-medium zone or disabling it — trading raw executable coverage for near-target OOD/incorrect execution and 4× lower latency.
- **Recall / hard negatives:** mining (`data/analysis/hard_negatives.md`) found only diffuse singleton confusions — no cluster worth targeting — so no prototype additions were made and recall@1/@3 stay 0.814 / 0.907. The recall lever remains a larger/fine-tuned encoder (future work), not spot-fixes.

## Residual gap & levers
OOD false-execution plateaus at **0.107** deterministically (≈3 test OOD rows carry spuriously high confidence). Closing the last gap toward ≈0 needs a stronger OOD signal — more OOD prototypes (capture scaled 46%→57% just by 54→96), an OOD-specific feature/embedding, or a larger fallback whose reject is more reliable — and/or accepting the deterministic safe point's coverage cost. The confidence-gate frontier makes that policy choice explicit and tunable per deployment, which is the deliverable.


# Spec 4 — Multi-Intent, Context-Aware Routing Results

Design: `docs/superpowers/specs/2026-07-24-multi-intent-context-aware-routing-design.md`. Adds context-vs-action span classification (lexical actionability filter), a plan-then-execute barrier, relative-op resolution against an injectable mock vehicle-state store, and a multi-intent + context eval axis. Suite: **144 core + 3 model tests** (incl. the canonical multi-intent end-to-end).

## Headline metrics (gold test split, n=192; + 7 context negatives)

| metric | Spec 3 | Spec 4 C_llm (balanced) | Spec 4 C (deterministic, zero-LLM) |
|---|---|---|---|
| recall@1 | 0.814 | **0.856** | 0.864 |
| recall@3 | 0.907 | **0.941** | 0.941 |
| multi_intent_set_recall | — | **0.819** | 0.819 |
| param_exact_match | 0.507 | **0.718** | 0.267\* |
| schema_valid_rate | 0.964 | **0.995** | 0.505\* |
| e2e_deterministic | — | 0.613 | 0.100 |
| e2e_ceiling | — | 0.753 | 0.893 |
| ood_false_execution | 0.321 | 0.321 | **0.000** |
| context_false_action | — | 0.857 | **0.000** |
| incorrect_execution | 0.252 | 0.287 | **0.032** |
| p95 latency (ms) | 1126 | 1019 | **73** |

\* param/schema at the deterministic point are computed only over the small subset that executes without the LLM, so they read low by construction.

## What Spec 4 delivered
- **Multi-intent routing works.** `multi_intent_set_recall` **0.819**. The canonical utterance 「后排小孩老去按车窗，把车窗锁打开。然后主驾这边窗户再开一点，天窗开到一半。」 executes exactly the three intended actions — `set_window_child_lock{enabled:true}`, `set_window_position{percent:40}` (relative, resolved from seeded state 30 +10), `set_sunroof_position{percent:50}` (一半→50) — with the narration clause suppressed (verified by the `@model` test).
- **No single-intent regression — net improvement.** recall@1 0.814→0.856, recall@3 0.907→0.941, param_exact_match 0.507→**0.718**, schema_valid 0.964→**0.995** (from fraction/relative param parsing + polarity/relative catalog prototypes).
- **Context suppression (plan path)** proven on the canonical case. The lexical actionability filter (target-alias ∪ domain-keyword + operation cue) was chosen over the confidence/OOD gate after a probe measured **0.12 vs 1.00** context separation — the gate cannot tell in-domain narration from commands (它们 topically identical).
- **Relative control + state** works: `再开一点`/`一半` resolve to absolute calls via `StateResolver` against the mock `VehicleState` (priority live > confirmed > session), clamped to card min/max; the LLM never invents the current value.

## The central finding — context/OOD is the medium-band-LLM residual
`context_false_action_rate` is **0.857** at the C_llm point but **0.000** deterministically. The context negatives are all **single-clause narration** (副驾说有点热 …) with zero action spans, so they route through the **legacy** path, where the medium-band LLM executes in-domain narration — the same liability Spec 2/3 identified for OOD (`ood_false_execution` unchanged at 0.321). The actionability filter suppresses context only inside the **plan path** (multi-intent), not for a lone context clause. At the deterministic zero-LLM point both context and OOD false-execution are **0** (incorrect 0.032, p95 73 ms) — the safe operating point, consistent with Spec 3.

## Design pivot (user-approved, during implementation)
The spec's single multi-action LLM call empirically failed with Qwen3-0.6B (wrong functions, hallucinated params, under-generation — e.g. it picked `open_window{is_open:false, position:passenger}` for 再开一点, or emitted only 1 of 3 actions). Replaced by **per-span confirm-or-reject retrieval top-1**: one `complete_tool_call` per action span offering only that span's retrieval top-1 (+ `__reject__`), so the weak model fills params / abstains but cannot substitute a different function; relative spans are restricted to numeric-param candidates. This lifted the canonical case from 1/3 to **3/3** correct.

## Residual gap & levers
Single-clause pure-context (and OOD) false-execution at the C_llm point is the open gap. Levers, by leverage: (1) route 0-action / all-context utterances to a conservative clarify instead of the legacy LLM (trades a little command-shape coverage for safety); (2) more context/OOD negative prototypes; (3) a binary context/OOD classifier feeding the gate; (4) a stronger fallback whose reject is more reliable. The safe deterministic point (context/OOD = 0, p95 73 ms) is available today for a safety-critical deployment.

---

# Spec 5 — Utterance-Level Reply Results

Design: `docs/superpowers/specs/2026-07-25-utterance-level-reply-design.md`. Plan: `docs/superpowers/plans/2026-07-25-utterance-level-reply.md`. Adds a single spoken `RouteResult.reply`, composed deterministically from what the router already produced, plus four contract metrics that make the eval harness enforce it on every run. Suite: **208 core + 3 model tests**.

## What the gap was

The pipeline routed, validated, and executed — but never produced the one string a voice assistant speaks. Confirmations existed only per clause (`ClauseResult.response`), and `_route_plan` attached the **same** `ClarificationRequest` object to *every* unresolved clause, so a caller concatenating naively would ask the identical question two or three times.

## Headline metrics (gold test split, n=192; + 7 context negatives)

| metric | arm C (deterministic) | arm C_llm (balanced) | want |
|---|---|---|---|
| reply_action_coverage | **1.0000** | **1.0000** | 1.0 |
| reply_single_question | **1.0000** | **1.0000** | 1.0 |
| reply_nonempty_rate | **1.0000** | **1.0000** | 1.0 |
| reply_question_drop_rate | **0.0000** | 0.0104 | 0.0 |

## Zero routing change — proven, not asserted

The regression gate was checked by running both arms on `main` (pre-Spec-5) and on the branch and diffing every metric. **Every routing metric is identical on both arms**, to four decimals:

| metric | arm C main → branch | arm C_llm main → branch |
|---|---|---|
| recall@1 | 0.8644 → 0.8644 | 0.8559 → 0.8559 |
| recall@3 | 0.9407 → 0.9407 | 0.9407 → 0.9407 |
| multi_intent_set_recall | 0.8194 → 0.8194 | 0.8194 → 0.8194 |
| param_exact_match | 0.2733 → 0.2733 | 0.7248 → 0.7248 |
| schema_valid_rate | 0.5079 → 0.5079 | 0.9948 → 0.9948 |
| e2e_deterministic | 0.1067 → 0.1067 | 0.6200 → 0.6200 |
| ood_false_execution | 0.0000 → 0.0000 | 0.3214 → 0.3214 |
| context_false_action | 0.0000 → 0.0000 | 0.8571 → 0.8571 |
| incorrect_execution | 0.0312 → 0.0312 | 0.2850 → 0.2850 |

Only latency moved (run-to-run timing noise: C P95 71.7→75.6 ms, C_llm 1092→1085 ms). This is what "presentational" is supposed to mean, and it is now measured rather than claimed.

> **Note on the Spec 4 table above.** Its arm-C/C_llm figures predate commit `095a0aa` ("explicit value + relative verb is absolute"), which landed on `main` after those numbers were recorded. The small differences (e.g. `param_exact_match` 0.718 → 0.7248, `e2e_deterministic` 0.613 → 0.6200) are attributable to that fix, not to Spec 5 — confirmed by the `main` baseline runs above.

## The canonical reply, on real models

`pytest -m model` asserts the exact string end-to-end, with real Qwen3-Embedding-0.6B and real xgrammar-constrained Qwen3-0.6B:

```
后排小孩老去按车窗，把车窗锁打开。然后主驾这边窗户再开一点，天窗开到一半。
  → 已为您调整车窗儿童锁状态。已将主驾车窗开度调整到40%。已将天窗开度调整到50%。
```

Three confirmations, sentence-joined; the narration clause appears nowhere in the reply.

## The defect the metrics could not see — caught in final review

All four contract metrics read healthy while the composer was doing something genuinely unsafe. The original rule 5 said "no confirmations, no question, no failure → `好的。`". But a clause routinely reaches the reply layer having produced **nothing at all** (`response=None`, `clarification=None`, `validation_errors=[]`): on the plan path a MEDIUM-band span never enters `plan.actions` when there is no LLM client (`t2f/pipeline.py:200-201`), and on the legacy path `NullMediumResolver` deliberately does not execute, so it sets no response. Reproduced with MEDIUM-forcing thresholds:

| utterance | reply BEFORE the fix | reply AFTER |
|---|---|---|
| `把空调调到25度` | `好的。` ← nothing happened | `抱歉，这个操作没能完成。` |
| `后排小孩老去按车窗，温度调到25度` | `好的。` ← nothing happened | `抱歉，这个操作没能完成。` |
| `开车窗，温度调到25度` | `已为您调整当前区域车窗状态。` ← temperature request vanished | `已为您调整当前区域车窗状态。抱歉，这个操作没能完成。` |

Telling a driver "OK" for work that never happened is the worst failure mode this feature could have, and arm C's `e2e_deterministic` of 0.1067 means the MEDIUM band is the common case, not an edge case. **Why no metric caught it:** `reply_nonempty_rate` passes because `好的。` is non-empty; `reply_action_coverage` is vacuous because the clause produced no confirmation to cover; `reply_question_drop_rate` needs two distinct recorded questions and this clause records none.

The fix is confined to `t2f/reply.py`: `_has_failure` no longer requires `validation_errors`, so any clause that neither spoke nor asked counts as a failure. `好的。` is now reachable **only** when there are no clauses at all. This deviates from the brainstormed "soft ack when nothing acted" — that choice was made for *narration*, and it is not safe to extend to a *failed command*.

The lasting lesson: three contract metrics at 1.000 proved the reply was well-formed, not that it was true. Only reading the composition rules against real pipeline states found this.

## Two findings that changed the design

**Sentence-join, not comma-join.** The design originally specified stripping each confirmation's `。` and joining with `，`. Grounding the golden strings in the real catalog showed the templates are *self-contained* `已…` sentences (`已为您调整车窗儿童锁状态。`, `已将{position}车窗开度调整到{percent}%。`), so comma-joining produced `已…，已…，已…` — three `已` in one breath. Concatenating the sentences as-is reads better and needs no template rewrite.

**`reply_single_question` is not punctuation-based.** The obvious implementation counts `？` in the reply. But `build_plan_clarification` returns `关于「…」我还需要确认一下，请补充信息。` — **no question mark at all**. A `？`-counting metric would have read 1.0 forever while measuring nothing. It instead counts how many *distinct recorded question strings* occur as substrings of the reply, ignoring any that are contained in another (so a template that is a prefix of a longer one does not read as two).

## Known limitation — dropped clarifications on the legacy path

Composition rule 3 is "distinct questions → the first wins", which guarantees one question per reply. That is correct on the **plan path**, where `build_plan_clarification` already consolidates every pending span into a single question naming each. The **legacy multi-clause path** has no such consolidation, so a second clause's genuinely different question is silently dropped:

```
车里闷，换外循环吸点新鲜空气
  questions = ['请补充更多信息。', '您想调到几档？']
  reply     = '请补充更多信息。'          ← the second need is never voiced
```

`reply_single_question` cannot see this — exactly one question *is* spoken, which is all it checks. `reply_question_drop_rate` was added to measure it honestly: **0.0000 on arm C** and **0.0104 (2/192) on arm C_llm**. (Under `--fake --permissive`, where far more clauses land in a clarifying band, it reaches 0.0183 = 6/328 — a property of that threshold setting, not of the real pipeline.) Closing it means giving the legacy path the plan path's consolidation, which is a behavior change and belongs in its own spec.

## What `reply_action_coverage` does and does not prove

Stated plainly, because a contract metric that reads 1.0 for structural reasons is worth less than it looks: `compose_reply` builds the reply *by concatenating* the very `response` strings this metric checks for, so today it cannot read below 1.0 without `t2f/reply.py` itself being broken. It is a **regression tripwire** — it would catch a future change that starts omitting a confirmation — not an independent behavioral check. The composition rules themselves are pinned by the 26 unit tests, 9 golden tests, and 11 end-to-end tests.

## Rejected: gold reply annotations

Annotating a gold `reply` on the 64 `multi_intent` rows was considered and rejected. A gold reply bakes in *which actions the arm executed*: arm C executes a subset of what C_llm executes, so C would score near zero even when composing perfectly from what it did execute. That measures routing again, not composition — and every template tweak would invalidate 64 annotations. Exact-wording protection is bought far more cheaply by 8 golden tests over fixed inputs.

## Bottom line

The router now returns something speakable on every path — plan, legacy, rejection, and validation failure — with at most one question, verified by 62 new tests (26 unit + 9 golden + 11 end-to-end + 16 metric) and four harness metrics, and with both eval arms proven bit-identical to the pre-Spec-5 baseline on every routing axis.

---

# Spec 6 — End-to-End Test Cases Results

**Date:** 2026-07-26
**Suite A:** `tests/e2e/` — the real `Pipeline.route()` over the 3-card fixture catalog with `FakeEmbedder`; no model, no GPU
**Suite B:** `data/eval/e2e_cases.jsonl` — 54 labelled rows scored as a **separate** slice
**Arm:** C (deterministic, real embedder), gold test split n=192

## The gap this closed

Before Spec 6 the repo had 208 tests and 328 labelled rows, and **not one case asserted what the driver
hears**, nor **exercised a failure**. `gold.jsonl` carries no reply field, and a scan for unusable
parameter values found **zero** rows — every case ever scored either routed correctly or was refused.
Step 4 of the business workflow was graded only by contract metrics that cannot fail by construction.

## What was added

| | count |
|---|---|
| Suite A end-to-end cases | **36** (25 green, 11 red) |
| eval unit tests (validator + metrics) | 17 |
| Suite B rows | 54 — 22 `invalid`, 20 `asr_noise`, 12 relative-with-state |
| `expected_reply` annotations | 37 |
| **Full suite** | **250 passed, 11 xfailed, 3 deselected** |

Green cases characterise behaviour as it is; **red cases use `xfail(strict=True)`**, so a case that
starts passing becomes a *failure* and forces promotion. The suite reports its own progress rather
than waiting to be asked. Every green literal was measured against the running pipeline, not assumed.

## Headline metrics (arm C, real embedder)

| metric | value | denominator | reading |
|---|---|---|---|
| `invalid_no_execution_rate` | **1.0000** | 22 | nothing unusable reached the vehicle — the safety half of 4b holds |
| `reply_exact_match` | **0.0811** | 37 | the driver hears the correct sentence in **3 of 37** cases |
| `n_e2e_rows` | 54 | — | |

`reply_exact_match` is reported beside its denominator deliberately. `reply_action_coverage` shipped
in Spec 5 reading 1.0000 while being incapable of failing, and only a paragraph of prose recorded
that. Every subset-scoped metric here prints its own denominator, so a vacuous 1.0 is visible.

**0.0811 is the honest state of step 4**, and it is not to be improved by softening the annotations.

## Four defects the existing 208 tests could not see

1. **A failed actuation is spoken as success, and commits vehicle state.** With an executor returning
   `{"ok": False}`, `route("把空调调到25度")` returns `已将当前区域温度设置为25°C。` and writes the
   confirmed state layer. `execute()`'s return value is discarded at all four call sites
   (`t2f/plan.py:43`, `t2f/pipeline.py:64`, `:104`); `t2f/plan.py:48` marks
   `"executed"` unconditionally. **This is the case that matters most** — it is harmless only because
   `MockExecutor` cannot fail, and becomes a live safety defect the day a vehicle adapter is attached.
2. **`别关车窗` ("don't close the window") dispatches `is_open=False` and closes it.** Polarity is
   keyword-derived with no negation handling (`t2f/lexical.py:70-73`), so the system executes the
   inverse of the instruction.
3. **Opening and closing the window produce byte-identical replies.** 43 of 92 cards confirm an action
   without stating the value set (`render_response` humanizes only `position`).
4. **Every failure cause collapses to one constant.** `bad_enum` carries the card's enum list and
   `type_mismatch` carries `temperature must be numeric`; both reach the reply layer on
   `ClauseResult.validation_errors` and both are spoken as `抱歉，这个操作没能完成。`

## A fifth gap, surfaced by authoring the rows

**A single-clause relative command can never be resolved against vehicle state.** `StateResolver` is
constructed only in `PlanExecutor.__init__` (`t2f/plan.py:17`) and called only at `t2f/plan.py:26`,
reachable only via `_route_plan`. A bare `温度再调高一点` goes down `_route_legacy` and clarifies
instead of resolving, however much state is available. This was invisible because all 6 state-carrying
gold rows are `multi_intent`; 8 of the 10 new relative rows expose it.

## Zero contamination — proven, not asserted

Eight metrics (`param_exact_match`, `schema_valid_rate`, `incorrect_execution_rate`,
`clarification_rate`, `json_valid_rate`, and all four `reply_*`) plus the latency percentiles have
**no row-type filter**, so appending to `gold.jsonl` would have moved numbers throughout this
document. The new rows therefore live in their own file, scored as `context_negatives.jsonl` already is.

Arm C was run three times — before the change, with the wiring and no rows, and with all 54 rows
loaded. **Every gold metric is identical to four decimals across all three:** recall@1 0.8644,
recall@3 0.9407, multi-intent set-recall 0.8194, param exact-match 0.2733, schema-valid 0.5079,
e2e deterministic 0.1067, e2e ceiling 0.8933, OOD 0.0000, context 0.0000, incorrect 0.0312,
clarification 0.0000, avg-LLM-calls 1.0175, json-valid 0.3266, all four reply metrics unchanged,
candidate-gen recall@3 0.9407. Only latency moved (P95 78.4 → 83.9 ms, run-to-run noise).

## Honest limits

- **`asr_noise` rows encode our belief about ASR errors, not measured misrecognitions.** We have no
  ASR. They are weaker evidence than the rest of this document and should not be quoted beside
  measured numbers as though they carried equal weight.
- **Suite A proves mechanisms on 3 cards, not the 92-card catalog.** It shows *that* a cause is
  unexplained; Suite B shows how often. Neither is sufficient alone.
- **Two of the 22 `invalid` rows claim `bad_enum` but fire `missing_required`** — `氛围灯调成粉色`
  and `导航走最省油的路线`. The extractor drops the unrecognised enum value, so "you named an
  unsupported colour" is indistinguishable from "you named no colour". The label records the true
  cause; the divergence is the finding, not an error.
- **`xfail(strict=True)` reads as green in CI.** A reviewer skimming "250 passed" will not see the 11
  red cases. That is why the count appears in the README status line.

## Bottom line

"How much of the workflow do we meet?" is now a number a test run prints rather than a claim a
document makes: **11 red cases and `reply_exact_match` 0.0811** are the measured distance between what
the Central Model does and what the business workflow specifies. Steps 2 and 4a are in good shape;
step 3's actuation half and step 4b are not, and both are now guarded so that closing them is
self-reporting.

---

# Simplification pass — what left `t2f/`, and where it went

**Date:** 2026-07-26

The runtime package had accumulated code no production path could reach. On a target whose
requirement says *minimize hardware resource consumption as much as possible*, that is not a
tidiness problem. This pass applied one rule — **if `Pipeline.route()` cannot reach it, it is not
runtime** — and one guard: **deleting code must not delete evidence.**

Nothing here changes behaviour. No number above moves.

## Moved to `research/` (measured, not shipped, not packaged)

| What | Backs which published number | Why it moved |
|---|---|---|
| `t2f/safety/` + `ConfidenceModelGate` | the Spec-3 safety/coverage frontier | no eval arm ever constructed it — **and `t2f/gate.py` imported `confidence_features` at module level**, so the whole path loaded on every gate import |
| `t2f/classify/` | Spec-2 Arm D | Arm D only; `candidate_gen_recall@3` 0.907, identical to Arm C — no measured gain |
| `t2f/dialog.py` | Spec-2 multi-turn follow-up **1.000 (n=46)** | never imported by `Pipeline`; the only non-test constructor was `eval/run_followups.py` |

All three remain runnable and their tests still pass in place. `eval/` may import `research/` — neither
ships — so **Arm D and the follow-up harness stay reproducible**. `t2f/` imports neither.

The Spec-3 entry deserves its own sentence, because it is the one that could mislead: this is not
promising work parked for later. Spec 4's deterministic Arm C reports OOD false-execution 0.000,
incorrect-execution 0.031 and P95 73 ms — **better than the learned gate's best published operating
point** (0.107 / 0.067 / ~275 ms). The simpler approach overtook it. Reviving it needs a reason
beyond "it exists".

## Moved to `eval/` (offline tooling that was living in the shipped package)

- `calibrate_gate` — 80 lines, the largest function in the runtime package, reachable only under
  `--calibrate`. On the vehicle, thresholds arrive pre-baked in `config.yaml`.
- `t2f/tools/` → `eval/tools/` — offline hard-negative mining.

## Deleted outright

| What | Why |
|---|---|
| `Span.attached_context` | written for every context span, read by nothing but two test assertions. The attachment pass went with it. |
| `FunctionCard.hard_negatives` | parsed from every card, read by nothing. The YAML keeps the data. |
| `PendingState`, `SessionState`, `ClarificationRequest.pending` | `build_clarification` constructed a `PendingState` on the live route path that only `dialog.py` ever read. Now owned by `research/dialog.py`. |
| `psutil` | declared as a hard dependency, imported by zero files. It can return with the memory benchmarking it was meant for. |

## Result

`t2f/` now contains only what `route()` reaches, and imports nothing from `eval/` or `research/` —
a layering violation that existed until this pass (`t2f/classify/train.py` imported `eval.dataset`).

**249 passed, 11 xfailed** (from 250 — one test asserted a deleted field's default). Arm C
re-measured after the move: every metric unchanged.

**Not deleted, deliberately:** `models/clf_charngram.joblib` is **185 MB** for a component with no
measured recall gain. It is gitignored, so it never was part of the repo, but it must not enter a
vehicle image. Regenerate with `python3 -m research.classify.train` if Arm D is ever revisited.

---

# Spec 7 — SQLite Vehicle Simulator Results

**Date:** 2026-07-26
**What:** `sim/` — a SQLite-backed simulated vehicle behind the existing `execute()` seam.
**Headline:** step 3's actuation half is now covered in simulation, requirement 4b gained its third
failure category, and the red count went **11 → 9**.

## What existed before

`MockExecutor` was six lines returning `{"ok": True}`, and its return value was discarded at all three
runtime call sites. `t2f/plan.py` marked every dispatched action `executed`, committed vehicle state
and rendered a confirmation regardless of what the vehicle said. `VehicleState`'s `live` layer had no
producer at all.

## The design

**The DB is the car.** Rows are **signals** — `(entity, attribute)` — not functions. That single
decision is what makes it a simulation rather than a dictionary: `open_window` and
`set_window_position` address the same physical window and write `window.driver/window_position`,
instead of the car holding two contradictory beliefs about one window.

Four tables: `signal` (current condition + physical limits), `operation_log` (every attempt and how it
ended), `device` (availability), `precondition` (as data, not code).

One operation resolves to signals → checks device availability → checks preconditions → checks the
signal's **physical limits** → writes every signal in one transaction → logs either way.

**Physical limits are deliberately separate from the card's.** A card says a window is 0–100; a jammed
window is 0–60. Validation and actuation are different questions, and 4b's third branch exists only
because they can disagree. Proven: `validate_tool_call("set_window_position", {"percent": 90})`
returns a tool call with zero errors, and the simulator refuses it.

## Measured, end-to-end through `route()` on the real 92-card catalog

| | result |
|---|---|
| `把主驾温度调到25度` | `climate.driver/temperature` **24.0 → 25.0**, reply `已将主驾温度设置为25°C。` |
| same words, A/C off | **24.0 → 24.0 (untouched)**, reply `空调尚未开启。` |
| `开车窗,主驾温度调高一点`, car at 18 | **18 → 28**, resolved against real vehicle state |
| mixed plan, one refused | `已为您调整当前区域车窗状态。空调尚未开启。` |

The relative case carries a negative control — the same utterance without the state snapshot gives
`missing_state` and leaves the car unmoved — so resolution is proven state-driven, not coincidental.

## Five defects found while building, four of them in the plan

The plan was written from reading the code, not running it. Building it found:

1. **25 signal collisions across 54 of 92 functions.** The plan derived the signal attribute from the
   primary *parameter* name, which is generic — `enabled`, `level`, `direction`. `seat.<pos>/level`
   was shared by seat heating, ventilation, massage and lumbar support. Turning on heating would have
   overwritten the massage level. Worse: seeding writes limits last-writer-wins, so that row would
   have carried lumbar's 0–4 and a **valid** `set_seat_massage{level:5}` would have been refused as
   physically impossible. Fixed by deriving the attribute from the *function*; now guarded by a
   catalog-wide test plus a test that the guard cannot pass vacuously.
2. **A dead precondition.** The plan referenced `window.all/child_lock`; the real attribute is
   `window_child_lock`. The plan's own test only asserted non-emptiness, so dead config would have
   passed and then never fired. The replacement guard is mutation-tested.
3. **Seeded signals with no limits.** `set_signal`'s `ON CONFLICT` refreshes value but not limits, so
   card-by-card seeding left both aliased pairs with *no physical limit at all* — making the
   `out_of_range` branch a silent no-op exactly where the alias design put the interesting behaviour.
4. **A false-green test.** The plan narrowed a seeded row with `set_signal(limits=(0,60))`, which does
   not narrow limits. Run as written, the window moved to 90 and the test passed anyway.
5. **Branch-order fragility** in `_route_plan`: with `failed` checked after the generic clarification
   branch, a refused action is spoken as a question about a request that was understood perfectly.
   Mutation-tested. Note it is invisible in a *uniformly* failing plan — phase 3's `pending` filter
   excludes `failed`, so no clarification exists to mis-speak — and only a **mixed** plan exposes it.

## What this did not do

- **The ten validation causes are still dropped.** This spec scoped itself to *executor* causes. The
  `out_of_range` / `bad_enum` / `type_mismatch` table remains gap 2 and its 7 red cases stay red.
  `reply_exact_match` is unchanged at **0.0811**.
- **A single-clause relative command still cannot resolve.** `StateResolver` is reachable only from
  the plan path, so `温度调高一点` alone still clarifies. Documented in the test, not asserted.
- **16 catalog functions write no signal.** Eight are correctly momentary (`next_track`,
  `take_screenshot`). Eight bear real state and do not record it — `lock_doors`/`unlock_doors` is the
  notable one, since nothing can express "the doors are locked" as a precondition. Needs a
  constant-write mechanism `resolve_writes` does not have.
- **One reply-composition nuance:** if one clause is refused *with* a detail and another is silently
  unresolved, the generic line is suppressed, so the second failure is not separately enumerated. No
  false affirmation results — the driver is still told something failed.

## Regression

`t2f/` gained the executor-result contract only. The eval harness constructs no executor, so it uses
`MockExecutor`, which returns `ExecResult(ok=True)` unconditionally and cannot reach any new failure
branch. Arm C re-measured after the change: unchanged.

**305 passed, 9 xfailed** (from 293/11 before this spec, 250/11 before Spec 6).

---

# Spec 8 — Interactive Session Results

**Date:** 2026-07-29
**What:** `cli/` — `python3 -m cli`, a terminal session where a person types Chinese and watches all
four workflow steps run against the SQLite-simulated car. Plus `t2f/build.py`, one place that
assembles a Pipeline.
**Headline:** no metric, by design. This spec produces a *tool*, and its value is what the tool
surfaces. 46 tests, almost all over the pure `Turn → text` renderer.

## Why a hand-testing tool earns a spec

The eval harness reports 20-odd numbers and none of them says what a driver would experience. Nine
consecutive utterances in an early demo produced nine generic apologies — that is `e2e_deterministic
0.1067` made visible, and no metric had ever made it visible before. A session that shows the
recognition, the row that moved in the vehicle database, and the sentence spoken, all three at once,
is the only artifact in this repo where the gap between "the numbers are fine" and "this is unusable"
is legible.

## The seam it closed

`t2f/build.py` is now the single place a Pipeline is constructed, and both the session and eval arms
C / C_llm call it. Before this, `eval/` was the only assembler — recorded as gap 6 in the system
design — which meant the thing a person tried by hand and the thing the metrics described could drift
apart with nothing to notice. They now cannot.

## Three defects found while building the renderer

All three are cases where the display would have lied, and all three are now regression-tested:

1. **"Nothing moved" and "no state to move" rendered identically.** `打开空调` against an already-on
   A/C is a success that changes no signal; `next_track` has no signal at all. Showing both as a bare
   `executed` made a working command look broken.
2. **`resolved by LLM` appeared above `unresolved`.** `NullMediumResolver` sets `needs_llm` even with
   no model attached, so a span arrived flagged as escalated when nothing had seen it. The block
   contradicted itself.
3. **A question that WAS the reply printed twice.** An out-of-scope utterance asks the driver to
   rephrase and that question is the whole reply, so the same sentence appeared on two adjacent lines
   and read as two separate questions.

## What it did not do

- **No routing change of any kind.** `t2f/` gained `build.py` and nothing else; every eval number was
  unchanged across this spec.
- **`--fake` misroutes badly** and the guide says so. Its embedder has no semantics; the mode exists
  to check plumbing after a code change, not to judge the system.

**498 passed, 1 xfailed** at the close of this spec.

---

# Spec 9 — Scene Engine Results

**Date:** 2026-07-30
**What:** `scene/` — a proactive subsystem beside `t2f/` and `sim/`. Structured perception in, at
most one spoken question out, and the vehicle moves only after the driver's explicit consent.
**Headline:** `scene_false_speech_rate` **0.000** and `scene_false_consent_rate` **0.000** on arm S,
over denominators of 9 and 4. 635 lines of `scene/`, 111 tests plus 2 model-backed.

## The shape, and why it is this shape

The router's own architecture pointed at perception: deterministic rules decide the clear cases, an
xgrammar-constrained Qwen3-0.6B sees only near-misses and observations no rule anticipated, and
everything else is silence. **Arbitration order is what enforces "the LLM never overrides the rules"**
— control flow, not prompt wording. A rule at MATCH means the model is never constructed a prompt.

`scene/` has no path into `Pipeline.route()`. The two subsystems meet at exactly one seam,
`execute(ToolCall) -> ExecResult`, so a scene-generated call gets the same validation, preconditions,
physical limits and operation-log entry as one the driver asked for — and no scene change can move a
router metric.

## Measured

Arm **S** (rules only) and **S_llm** (fallback attached) differ in nothing but the client. The scene
path uses no embedder at all, which is why these numbers do not carry the `--fake` caveat the
router's do.

| Metric | arm S | arm S_llm | denominator |
|---|---|---|---|
| `scene_false_speech_rate` | **0.0000** | **0.0000** | 9 silent rows |
| `scene_recall` | **1.0000** | **1.0000** | 4 speaking rows |
| `scene_false_consent_rate` | **0.0000** | **0.0000** | 4 must-not-consent rows |
| `avg_llm_calls_per_event` | 0.0000 | 0.1538 | 13 rows |

`scene_false_speech_rate` is the number this design optimises for — the proactive analogue of
`ood_false_execution_rate`. A proactive system's worst failure is not being wrong; it is being
uninvited. Arm S_llm consults the model on exactly two rows (a 0.62-confidence near-miss and an
observation no rule mentions) and **declines to speak on both**: the fallback earning its place by
staying quiet.

## The interaction it demonstrates

```
/scene rear_occupant=child conf=0.9  →  后排有小孩，要打开儿童锁吗？
好                                    →  window.all/window_child_lock  False → True
                                         已为您打开车窗儿童锁。
开车窗                                 →  refused · 车窗儿童锁已开启 · nothing changed
```

The third line is the point. `sim/seed.py:38` already declared `open_window` to require
`window_child_lock == False`, so a proactive action changes what a later driver-initiated command is
permitted to do, and the refusal is explained. Both entry points, all four workflow steps, one car.

## Safety properties, asserted over every rule

`tests/scene/test_contract_sweep.py` follows `test_s8_contract_sweep.py`: a property asserted over the
whole rule set cannot be satisfied by a lucky special case. **Eight mutations were applied one at a
time and every one was caught by its intended test** — the sweep is not vacuous.

- no rule match ever produces a ToolCall — consent is the only path to the car
- every `ask` carries a proposal that validates *before* the question is spoken
- every `Signal` condition names a row the seeded car actually holds (a typo reads as `None`, which
  is a REJECT, which is silence — a misspelled rule would be permanently and undetectably mute)
- a rule never fires for what is already true, never bypasses its cooldown, and never talks over a
  question the router is waiting on

**Consent is exact membership in a closed lexicon, never substring.** A sweep of 6,882 driver-facing
strings in this repo's own data found zero collisions — and found three real commands (`行李箱`,
`打开行李箱`, `开行李箱`) that a substring test would have read as consent, because `行` is an
affirmative. The driver asks for the trunk and the car locks the windows.

## Five defects the happy-path tests would not have caught

Each was found by an adversarial pass, and each now carries a regression test:

1. **`VehicleFacts` held the write path its own docstring forbade** — it stored the whole
   `SqliteVehicle`, leaving `set_signal` one attribute access away from any rule.
2. **`resolve()` could raise.** The exception guard was on the perception path while the consent path
   — the one that touches the car — could propagate a traceback mid-actuation.
3. **A `notify` could carry a question.** `{"decision": "notify", "reply_intent": "ask_..."}` was
   schema-valid: the car asks, no consent is pending, the driver answers into the void. Fixed by
   keying the grammar on the decision, which moved three checks out of Python.
4. **`/reset` did not reset the engine** — `好` after a reset re-opened a lock on a car nobody had
   been asked about.
5. **A docstring claimed a mechanism its assertion could not distinguish**, found by mutating the
   sweep: the silence it attributed to one cause was over-determined by two.

## It also closed the last red case

`set_window_child_lock` confirmed identically whether the lock went on or off — worse in a proactive
flow, where the driver never named a direction and hears a sentence that does not say what happened.
`render_response` now humanises booleans and 38 of 39 boolean cards state their direction
(`spray_washer` is a momentary trigger, not a state). **Red count across the project: 11 → 9 → 1 → 0.**
Arm C was re-measured after the change and no routing metric moved; `reply_exact_match` stayed
0.0811, because none of the 37 free-form annotations covers a boolean confirmation.

## What this did not do

- **The gold file is authored, not observed.** `scene_recall 1.000` measures agreement with our own
  beliefs about what a camera would report. It is a contract test wearing a metric's clothes — the
  same caveat this repo already applies to its `asr_noise` rows.
- **`persist_for` ships with no end-to-end consumer.** The shipped rule sets it to 0.0, so the
  mechanism is unit-tested and unused.
- **The closed consent lexicon drops oblique agreement.** `开吧` is a yes a person would say; it is
  routed as a command instead, and in the gold row it opens the defroster. That cost is recorded
  rather than argued about.
- **One rule exists.** Priority and tie-breaking are implemented and swept, but never contended.
- **No vision.** `/scene` is a person typing what a camera would have reported. Parsing real captions
  is deferred, and the natural next step is this project's own thesis applied to perception —
  retrieval over scene prototypes, with its own calibration and gold.

**624 passed, 5 deselected, 0 xfailed** (from 498/1 before Spec 8).

---

# After Spec 9 — Sensed Signals, Staleness and Intake

**Date:** 2026-08-01
**What:** two pieces of work that are deliberately **not** numbered specs. They add no capability to
the router; they change what the car can know and how facts reach the modules that read them.
`sim/` gained signals it senses but cannot command, `scene/` gained a second rule, and `intake/` —
packaged — became the composition root and the one door every input comes through.
**Headline:** **no measured number moved, and that is the result.** Both proof obligations came back
byte-identical, with the scene arms re-run over **two** rules instead of one.

## Why this is not a spec

Every numbered spec above answers "what can the system now do". These answer "what can the system now
*know*, and can it tell the difference between knowing and having known". A capability claim is
measured against gold; this is measured against **the absence of change** in gold that already exists.
Filing it as Spec 10 would have implied a metric to move, and the only honest metric here is that
nothing moved.

## The defect underneath both

`updated_at` had been written on every signal write since Spec 7 and was **read by nothing** — a
repo-wide grep found it in a schema line and two comments. So `SignalAbove("vehicle.all",
"speed_kph", above=5.0)` fired the animal warning off a speed frozen ten minutes ago exactly as
readily as off a live one. **A dead bus and a stationary car were indistinguishable to every rule.**

Perception had received precisely this discipline in Spec 9 — `Observation.is_live`, read-time expiry,
no sweeper — and the car never had. The asymmetry was invisible because nothing in the repo could
express it: the two stores were separated by a string prefix (`inside.` / `outside.` / `vehicle.`),
which is a convention, not a type.

## Sensed signals, and the second rule

`sim/` had modelled exactly what the 92 cards can write. That was right until a rule needed to read
something no card produces, and the correct response was not to weaken the guard that asserts it:
`tests/sim/test_seed.py` now asserts the car holds exactly the writable signals **plus the declared
sensed ones**, both directions still enforced, and a sensed signal nobody declared is still a failure.

Motion needed a condition form the closed vocabulary did not have. `SignalAbove` is a **third shape**
rather than an operator on `Signal`, because the closed vocabulary is what lets the contract sweep walk
every rule and assert properties over all of them — three trivially inspectable forms stay inspectable;
one form with a comparator does not.

`ANIMAL_AHEAD` proposes nothing: no vehicle function makes an animal in the road safe, so there is
nothing for consent to authorise. It outranks the child-lock question (90 vs 50) and is readier to fire
(0.70 vs 0.80), because a missed animal is worse than a spurious warning while a spurious question is
merely annoying. **It is the first pair of shipped rules that can contend**, so it is also the first
time the arbitration code has had anything real to arbitrate — the suppressed rule reports `outranked
by animal_ahead` and does not spend its own cooldown.

Setting a sensed signal is a **simulator control, not a Central Model action**. Telling the simulator
the car is doing 45 is the world changing, the same category as a camera seeing a child, so it lives
outside `ACTIONS` in a disjoint `CONTROLS` table with its own route — `/control/`, not `/action/`.
Anyone later asking "how does the page reach the car" finds two lists with different names and
different justifications rather than one list with a quiet sixth entry.

## Intake, and a view that owns nothing

One envelope: `Input(source, at, payload)`. **There is no `kind` field** — the payload's type *is* the
kind, and a kind beside a payload is two statements about one fact that eventually differ. Sources
declare what they produce, so `Input(source="cabin_cam", payload=SignalWrite(...))` cannot be
constructed; before this, `source` was decoration that everything defaulted to `"cabin_cam"`,
including vehicle-namespace observations, and nothing noticed.

`WorldView` is read-through and **owns nothing**. A hub that stored would recreate exactly the problem
signal-keyed state was built to prevent: `open_window` and `set_window_position` are keyed by signal so
they cannot hold two contradictory beliefs about one window, and a hub holding a copy of
`window_child_lock` rebuilds that one level up, with its own staleness, so the car and the hub can
disagree about a lock. That it holds nothing writable is asserted by walking the instance, inheriting
the property `VehicleFacts` carried before it was absorbed.

`intake/` ships. The composition root — the only thing that assembled router, scene engine and car —
had lived in `cli/session.py`, which `pyproject` deliberately excludes, so a real integration had to
reimplement wiring the CLI already worked out.

Staleness falls out: a sensed signal past its `max_age` reads as `None`, identical to a signal the car
does not hold, so every condition rejects and names both ages
(`vehicle.all/speed_kph is stale (40.0s > 2.0s)`). **Actuated signals never expire** — a window
position holds until commanded, a speed is a measurement whose absence means the bus stopped. And
`max_age` is declared on the **signal**, not the rule, so **no rule can forget to ask**.

## Five silent failures found by building it

Each would have shipped looking correct:

1. **Staleness failed open.** The session was on a monotonic clock (~756 thousand) while the car
   stamps `time.time()` (~1.79 billion), so the age came out at about **minus 1.78 billion seconds** —
   under every `max_age`, so every stale signal read as fresh. The discipline had tests passing around
   it and did nothing.
2. **The seeded speed kept its seed stamp**, so two seconds into any session the animal rule blamed
   the bus for a car that was simply parked.
3. **`/reset` left a held 45 kph to republish** into a freshly seeded vehicle, arriving perfectly live
   while `/car` reported the car as seeded.
4. **`SceneEngine.reset()` rebound the perception store**, which would have stranded every `WorldView`
   built over it on the discarded instance — reading empty perception forever, with nothing raising.
   It clears in place now.
5. **`intake/hub.py` reaches `sim.seed` through a guarded import**, so renaming `sensed_max_age` would
   have made staleness a permanent no-op **with no test failing**. There is now a test that fails on
   exactly that, verified by performing the rename.

The contract sweep was re-verified by mutation rather than assumed: eleven mutations, each caught by
the property meant to catch it. The animal rule's signal typo is caught by exactly one test in the
suite — the sweep property whose own docstring warns that the failure is "indistinguishable from
working correctly."

## Measured

| obligation | result |
|---|---|
| `run_eval --arm C --fake --permissive` | byte-identical but for `p50/p95_latency_ms`, which is host jitter larger than most real changes |
| `run_scene_eval --arm S` | **byte-identical**, over two rules instead of one |

Arm S, re-run 2026-08-01: `scene_false_speech_rate` **0.0000** (9), `scene_recall` **1.0000** (4),
`scene_false_consent_rate` **0.0000** (4), `avg_llm_calls_per_event` **0.0000** (13). `t2f/` was not
touched, so no routing figure in any section above can have moved.

## What this did not do

- **It added no observed data.** The rule set doubled and the gold file did not grow by a row.
  `animal_ahead` has **no gold row at all** — it is covered by unit tests and the contract sweep and by
  nothing in the measured column. The gold was deliberately not rewritten to keep a number stable.
- **`max_age = 2.0` for speed is a guess.** No measurement supports it; it is a plausible number for a
  10 Hz signal, one constant in one declaration, and provisional.
- **The pump is only as good as its callers.** A consumer that forgets to pump sees everything stale.
  That is the safe direction — stale reads as absent, so the failure is silence rather than a wrong
  action — but it is a real footgun, and it belongs in the module docstring rather than in a reader's
  memory.
- **`live_facts()` is not wired into the fallback prompt.** It now covers both stores, which is what
  would finally make "complex relationships between multiple context states" — named as a fallback job
  in the original scene-engine brief — expressible at all. Wiring it would move arm S_llm, which
  neither proof obligation covers, so it deserves its own measurement instead of arriving under a
  byte-identical arm S.
- **Arm S_llm was not re-measured.** Its column dates from 2026-07-30, when one rule shipped.
- **No caption parsing, no cross-modal reasoning, no real CAN adapter.** "VLM output" is still a
  structured `Percept` rather than a sentence; a rule still cannot condition on what the driver said;
  `can0` is a declared source with no hardware behind it. All three are named as deferred in the
  design, and the envelope makes each of them a new *source* later rather than a new *door*.

**860 passed, 1 skipped, 5 deselected, 0 xfailed** (from 624/5 at the close of Spec 9). The one skip
is `test_every_proposal_validates[animal_ahead]`: a notify-only rule has no proposal to validate, so
the sweep skips that property for it rather than passing vacuously. `t2f/` untouched. No new
dependency.
