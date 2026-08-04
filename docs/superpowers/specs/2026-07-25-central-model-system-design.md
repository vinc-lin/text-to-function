# Central Model — System Design & Requirement Coverage

**Date:** 2026-07-25
**Status:** Framing document. Supersedes nothing; it re-states Specs 1–5 against the stakeholder's
business workflow and records, honestly, where the code does and does not meet it.

> This document exists because the project had five *technical* specs and no single statement of
> **what the product is supposed to do end-to-end**. Everything below is grounded in code that was
> read, not in intent. Every coverage claim carries a `file:line`. Where the answer is "we do not do
> this", it says so plainly.

---

## 1. The business workflow

The core system is the **Central Model**. The in-vehicle voice interaction is:

| Step | Description |
|---|---|
| **1** | The user **speaks** a voice command inside the car. |
| **2** | The Central Model performs **segmented intent recognition** on the utterance. |
| **3** | The **corresponding operations are executed** from the recognized intents. |
| **4** | The system **feeds back execution results** via simple dialogue: **(4a)** on success, state the completed action; **(4b)** on failure, **explain the specific cause**. |

**Deployment constraint.** The target is vehicle-side. Linux is a *simulation* of the vehicle
environment at this stage. All performance standards must ultimately be benchmarked against **stable
operation on the 87 platform (Qualcomm SA8797)**, and the solution must minimize hardware resource
consumption.

---

## 2. What the Central Model is, and where it ends

The Central Model as built is a **text-in → (tool calls + one spoken reply)-out** component.

```
                    ┌───────────────── Central Model (this repo) ─────────────────┐
 microphone         │                                                             │        speaker
    │               │  normalize → segment → recognize per span → gate            │           │
    ▼               │      → validate → plan barrier → execute → compose reply    │           ▼
 ┌───────┐  text    │                                                             │  text  ┌───────┐
 │  ASR  │─────────▶│  Pipeline.route(utterance: str) -> RouteResult              │───────▶│  TTS  │
 └───────┘          │                                    .reply: str              │        └───────┘
   NOT IN           │                          .clauses[].tool_call ──┐           │         NOT IN
   THIS REPO        └────────────────────────────────────────────────┼───────────┘         THIS REPO
                                                                     ▼
                                                            Executor.execute(tool_call)
                                                            ── vehicle bus adapter ──
                                                                  NOT IN THIS REPO
```

**The single entry point** is `Pipeline.route(utterance: str) -> RouteResult`
(`t2f/pipeline.py:126`). **The single output** is `RouteResult.reply: str` (`t2f/types.py:158`) plus
the per-clause `tool_call`s that were dispatched.

Three interfaces sit at the boundary and are **not implemented here**:

| Boundary | Contract | Status |
|---|---|---|
| **ASR → Central Model** | a finalized, whole-utterance Chinese transcript as a Python `str` | consumer implemented; **producer absent** |
| **Central Model → vehicle** | `execute(ToolCall) -> dict` (`t2f/execute.py:5`) | seam exists and is injectable at all four dispatch sites; **only a mock implementation ships** |
| **Central Model → TTS** | one non-empty reply string, at most one question | producer implemented and metric-enforced; **consumer absent** |

This boundary is a deliberate scope decision, but until now it was **nowhere written down**. Stating
it is the main purpose of this document.

---

## 3. Coverage, step by step

### Step 1 — the user speaks · **NOT COVERED (by design, upstream)**

There is no audio anywhere in the repository — no capture, ASR, wake word, VAD, or endpointing. An
exhaustive search across every `.py`/`.md`/`.yaml`/`.jsonl` file for audio and speech terms returns
zero hits; `pyproject.toml:9` declares only `numpy`, `pyyaml`, `psutil`.

The system consumes what an ASR would emit. That is a reasonable division of labour — speech
recognition is a platform service, not a router concern — but two consequences must be owned:

- **No serving host.** The repo ships no server, daemon, CLI, or console script. The only two
  `__main__` blocks are the eval runner (`eval/run_eval.py:140`) and the classifier trainer
  (`t2f/classify/train.py:47`), and **the only non-test constructors of `Pipeline` in the entire
  repo are in `eval/arms.py:9,14,21,32`**. Production wiring decisions — is the LLM attached, are
  OOD prototypes loaded, is the classifier attached — currently live in the *evaluation* package.
- **No ASR-error robustness, and none measured.** `normalize()` (`t2f/normalize.py:7-12`) does NFKC
  folding, punctuation mapping, lowercasing and whitespace collapse — no homophone repair, no
  disfluency stripping. There is no `asr_noise` slice in the eval set, so every accuracy number in
  `RESULTS.md` is an upper bound measured on clean text. Mitigating: Chinese numeral parsing is real
  (`t2f/params/numerals.py`), colloquial fillers appear in both gold and silver data, and the fusion
  is embedding-dominated (`config.yaml`: embedding 0.88 vs keyword_alias 0.04), so degraded input
  tends to depress scores into the clarify/reject band rather than flip to a wrong action.

### Step 2 — segmented intent recognition · **COVERED, with a measured recall ceiling**

This is the most complete part of the system.

| Stage | Module | What it does |
|---|---|---|
| split | `t2f/segment.py` | punctuation + a 6-item conjunction list → fragments |
| label | `t2f/actionability.py` | `ACTION` iff the fragment has **both** a target word and an operation/polarity/value cue; else `CONTEXT` |
| recognize | `t2f/retrieve.py` → `t2f/score.py` | multi-prototype max-sim over 92 cards, then hybrid rescore |
| decide | `t2f/gate.py` | band HIGH / MEDIUM / LOW; `__ood__` top-1 forces LOW |
| route | `t2f/pipeline.py:135-138` | ≥2 action spans, or action+context → plan path; else legacy path |

Measured on the gold test split (n=192), deterministic zero-LLM arm C: **recall@1 0.8644**,
**multi-intent set-recall 0.8194**, **OOD false-execution 0.0000**, **context false-action 0.0000**,
**incorrect-execution 0.0312**, **P95 73 ms**.

Two honest limits:

- **Mixed-utterance recall ceiling.** On the plan path, when at least one `ACTION` and one `CONTEXT`
  span coexist, a genuine command that the lexical filter mislabels `CONTEXT` is **silently
  dropped** — never decided (`t2f/pipeline.py:198`), never executed, and never mentioned in the
  reply, because `Span.attached_context` (`t2f/segment.py:49` — the field no longer exists
  anywhere in the tree as of 2026-08-04; this paragraph describes the code as it stood on
  2026-07-25) has no production reader. This
  affects 10 of the 36 multi-intent test rows and caps set-recall at 0.8611 against 0.8194 achieved.
  The root cause is narrow and fixable: 8 rows lack an `_OP_CUES` entry (放 / 切到 / 切成 / 喷 / 息 /
  往上调) and 2 lack an alias-index target (HUD, 屏幕). Utterances with *zero* action spans are
  correctly rescued by the legacy fallback and are not affected.
- **Recognition is closed-set.** 92 functions across 10 domains. An in-car request outside the
  catalog is rejected, not attempted — the safe behaviour — but the rejection is generic
  (`t2f/respond.py:44`) and never says *which* capability is unsupported. That is a step-4b problem,
  recorded below.

### Step 3 — execute the operation · **PARTIALLY COVERED — the decision layer is real, the actuator is a stub**

Distinguish two things that are easy to conflate:

**(a) The decision-and-safety layer is genuinely implemented.** `PlanExecutor.finalize`
(`t2f/plan.py:19-54`) resolves relative operations against vehicle state, validates **every** action
before **any** action runs, executes only the fully valid subset in order, and writes each executed
numeric value back into the confirmed state layer. `validate_tool_call` (`t2f/validate.py`) enforces
candidate-set membership, unknown params, missing required, type, enum, and min/max. Nothing invalid
is ever dispatched.

**(b) The actuator does not exist.** The only executor is six lines:

```python
class MockExecutor:
    def execute(self, tool_call: ToolCall) -> dict:
        return {"ok": True, "name": tool_call.name, "parameters": tool_call.parameters}
```

No bus, no SOME/IP, no VHAL, no IPC anywhere in the repo. This is a declared non-goal and the
stakeholder's own framing ("Linux simulates the vehicle at this stage") accommodates it. The seam is
injectable at all four dispatch sites, so an adapter drops in without touching the router.

**What is *not* acceptable to leave as-is** is the wiring around that seam:

> **The return value of `execute()` is discarded at every one of the four call sites** —
> `t2f/plan.py:43`, `t2f/pipeline.py:64`, `t2f/pipeline.py:104`, `t2f/dialog.py:42` — and none of
> them is inside a `try`. `t2f/plan.py:48` then marks the action `"executed"` and commits vehicle
> state unconditionally, and `t2f/pipeline.py:219` renders a success confirmation unconditionally.

Consequence: **the day a real executor is attached, a failed actuation will be spoken to the driver
as a success.** No dataclass carries a vehicle-reported outcome (`t2f/types.py:124-158`). Requirement
4b's "the car refused" case is not merely unimplemented — there is no channel for it to travel on.
This is a wiring defect in the router, independent of who writes the vehicle adapter, and it should
be fixed before the adapter is written rather than after.

Related: `VehicleState` (`t2f/state.py`) is a real three-layer store and *is* populated in-process by
executed actions (`t2f/plan.py:44-47`), giving genuine cross-utterance session state. Only the
`live` telemetry layer has no producer, which affects relative commands with no prior value; those
fail safe to a clarification (`t2f/state.py:56-58`).

### Step 4 — feed back the result · **4a COVERED · 4b NOT MET**

**4a — success.** `render_response` (`t2f/respond.py:24-32`) fills each card's
`response_template` with localized parameter values; `compose_reply` (`t2f/reply.py:66-79`) joins all
distinct confirmations into one spoken string with at most one question. Four contract metrics score
**1.0000 on both arms** (action coverage, single question, non-empty), so every routing path returns
something speakable.

The residual: **43 of 92 cards confirm an action without ever stating the value that was set.**
Of the 55 templates with no `{placeholder}`, 12 are zero-parameter functions whose fixed sentence is
fully specific; the other 43 (34 with a boolean parameter, 9 with a required enum) say things like
`已为您调整空调开关状态。` — "I have adjusted the A/C power state" — without saying *on or off*.
Since polarity is keyword-derived with no negation handling (`t2f/lexical.py:70-73` maps 别关车窗 to
`is_open=False`), an inverted action is currently **unhearable** in the confirmation.

**4b — failure. This is the clearest gap against the requirement.**

The system computes rich, machine-readable causes and then throws them away before speaking:

| Cause available internally | Where produced | What the driver hears |
|---|---|---|
| `out_of_range` (e.g. `temperature > 32`) | `t2f/validate.py:33-36` | `抱歉，这个操作没能完成。` |
| `bad_enum` | `t2f/validate.py:41-42` | `抱歉，这个操作没能完成。` |
| `type_mismatch` | `t2f/validate.py:29,32,39,45` | `抱歉，这个操作没能完成。` |
| `unknown_param` / `unknown_function` / `not_in_candidates` | `t2f/validate.py:10-18` | `抱歉，这个操作没能完成。` |
| `llm_no_toolcall` | `t2f/pipeline.py:59` | `抱歉，这个操作没能完成。` |
| `no_numeric_param` / `missing_state` | `t2f/state.py:54,58` | a span-naming question with no cause |
| `missing_required` | `t2f/validate.py:21` | a real question — **but only for 3 parameter names** |
| vehicle-reported actuation failure | *does not exist* | success |

**`t2f/reply.py` never reads `ClauseResult.validation_errors`.** The field is populated
(`t2f/pipeline.py:27,59,76,77,103`), carried to the reply layer on `ClauseResult`
(`t2f/types.py:146`), and read by exactly one consumer in the repo — the eval harness
(`eval/arms.py:70`). Likewise `PlannedAction.error`, whose own docstring reads "short reason when not
executed" (`t2f/types.py:132`), is written at `t2f/plan.py:29,34` and read by nobody.

The one cause-specific path is the missing-parameter question, and its vocabulary is three words:
`_CLARIFY` (`t2f/respond.py:7-9`) knows `position`, `temperature`, `level`. The catalog has **76
required-parameter slots across 17 distinct names**; those three cover **10 of 76**. The remaining 66
— `enabled` (31 slots), `is_open` (7), `mode` (6), `direction` (6), `percent` (5), `contact`,
`destination`, `theme`, … — all produce `请补充更多信息。` ("please provide more information"), which
does not name what is missing. `ParamSpec.description` is parsed (`t2f/cards.py:23`) and consumed by
nothing.

Two further inconsistencies worth recording:

- The same invalid value is reported **differently by different paths**: on the legacy path it is a
  failure (`t2f/pipeline.py:103` → generic line); on the plan path it becomes a *clarification
  request* (`t2f/plan.py:33,52`).
- `t2f/reply.py:75-78` suppresses the failure line entirely whenever any sibling clause asked a
  question. On the plan path the failed span is still named in the consolidated question, so nothing
  vanishes silently there; a MEDIUM-band span skipped by the deterministic planner can, however, go
  unmentioned.

**Assessment: 4b is unmet.** Every failure collapses to one of three canned lines and none states a
cause. Crucially, this is *not* an architectural problem — the data already exists and reaches the
reply layer. It is a missing code→message mapping in one file plus a result contract at the executor
seam.

---

## 4. Performance and the 87 platform · **NOT BENCHMARKED**

The requirement makes 87-platform stability the standard for all performance claims. Against that:

| Requirement | Status |
|---|---|
| Benchmarked on SA8797 | **No.** Zero measurements on target hardware. |
| Latency measured | Yes, but on an **x86 dev box with a discrete GPU (RTX 4060 Ti, CUDA FP16)** — architecturally unrelated to SA8797. P50/P95 instrumented end-to-end (`t2f/pipeline.py:161,164,228` → `eval/metrics.py:171`). |
| Memory / footprint measured | **No.** Not RSS, not peak, not model size. `psutil` is declared at `pyproject.toml:9` and **imported by zero files**. |
| Cold start / model load measured | **No.** |
| CPU/NPU utilisation, power, thermal, soak stability, crash rate | **No.** "Stable operation" is entirely unmeasured. |
| On-device runtime implemented | **No.** `GgufEmbedder` (`t2f/embed.py:114`) and `GgufLLMClient` (`t2f/llm/client.py:82`) are 6-line classes whose every method raises `NotImplementedError`. |
| Quantisation | **No.** Q8_0 / Q4_0 exists as prose; both models run FP16. |
| Prototype embeddings precomputed | **No.** All 1,395 prototype texts are re-encoded at every `Pipeline` construction (`t2f/retrieve.py:33`). Bounded one-time cost (~1.5 s), does not affect per-request latency. |

This is a **disclosed, hardware-blocked deferral**, not a hidden hole — `RESULTS.md` labels its own
numbers as non-SA8797, and the GGUF seam is designed in. But the honest summary is: **no performance
claim in this repository has been validated against the requirement's stated standard.** The `<1500
ms` budget the docs measure against is a self-set engineering inference, not an 87-platform figure.

On "minimize hardware resource consumption", the architecture does pull the right direction — the
recommended deterministic operating point uses **zero LLM calls**, MRL-512 brute-force cosine over 92
cards, and logistic regression over cheap features. One artifact contradicts it loudly:
**`models/clf_charngram.joblib` is 184 MB** (a 2^18-feature hashing vectorizer) for a component whose
measured candidate-generation recall is indistinguishable from arm C's. It is gitignored, so it is a
local artifact rather than a committed one — but if arm D were ever promoted to the vehicle build it
would cost 184 MB for no measured gain.

---

## 5. Question 2 — what is in the repo that the workflow does not ask for

Of ~2,115 lines in `t2f/` (plus ~560 in `eval/` — the figure originally given here conflated the
two), the split is:

> **Superseded 2026-07-26 by the simplification pass.** `t2f/` is now **1,637 lines**; the classifier,
> the learned confidence gate and the multi-turn resolver moved to `research/`, and the offline
> calibrator and mining tool moved to `eval/`. See the pass's record in `RESULTS.md`. The
> classification below is what *drove* that pass, and is kept as written.

**On the runtime path and directly implementing steps 2–4** (~18 modules): normalize, segment,
actionability, lexical, numerals, embed, retrieve, score, signals, params, gate, validate, state,
plan, execute, respond, reply, pipeline, plus the 92-card catalog.

**Supporting runtime — not named by the 4 steps, but defensible** (keep):
- **OOD rejection** (`data/ood/prototypes.txt`, the `__ood__` marker, `__reject__` in the LLM
  schema). The workflow says "execute the recognized intent" and is silent on what to do with an
  unrecognizable one. Executing a guess in a vehicle is the worst outcome; this machinery is why
  OOD false-execution is 0.0000 on the deterministic arm.
- **The LLM fallback** (`t2f/llm/`, ~180 LOC). Optional by construction; the recommended safe
  operating point runs without it.
- **The plan barrier and relative-op resolution.** Both are prerequisites for doing step 3 correctly
  on multi-intent and 再开一点-style utterances.

**Genuinely outside the stated scope — the "ask the driver a question back" mode.** The workflow is
recognize → execute → report success or failure. It never says the system may ask a question. But
the system has a full third response mode:

| Piece | Size |
|---|---|
| `t2f/respond.py` clarification builders | ~19 LOC |
| `t2f/dialog.py` multi-turn follow-up resolver | 52 LOC |
| `t2f/types.py` `ClarificationRequest` / `PendingState` / `SessionState` | ~16 LOC |
| clarification branches in `t2f/pipeline.py` | ~18 LOC across 5 branches |
| question path in `t2f/reply.py` | ~16 LOC |
| eval harness, metrics, dataset, tests | ~160 LOC + 53 rows |

**Verdict, split two ways:**
- **The missing-required-parameter question is in-scope-but-unstated.** You cannot execute
  `set_temperature` without a temperature; asking is strictly better than guessing or failing. Keep.
- **The multi-turn session machinery is not part of this product today.** `FollowUpResolver` is
  **never imported by `Pipeline`** — `t2f/pipeline.py` has no dialog import, and the only non-test
  constructor in the repo is `eval/run_followups.py:9`, which hand-builds a `SessionState` and calls
  the resolver directly. The reported 1.0 follow-up score measures a component that is not connected
  to the product. Similarly `PendingState` is *written* by `build_clarification` and never *read* on
  the `route()` path — carried through the runtime and dropped.

**Development scaffolding that must not be counted as delivered product** (~1,200 LOC + 7,600 doc
lines): the whole `eval/` package, `t2f/classify/` (arm D — `Pipeline` defaults
`classifier_source=None`, `t2f/pipeline.py:119`), `t2f/tools/`, `calibrate_gate` (an 80-line offline
grid search living inside the shipped `t2f/gate.py`), the fake embedder/LLM, gold/silver datasets,
and the test suite.

Two items in this bucket deserve the stakeholder's attention:

- **The repo ships four systems, not one.** `eval/arms.py` builds arms C, baseline, C_llm and D, and
  they differ materially in safety: arm C has OOD false-execution 0.0000 and context false-action
  0.0000; arm C_llm has 0.3214 and 0.8571. **Which arm is the candidate build is a decision that has
  not been recorded anywhere**, and it is the single most consequential open question in the project.
  This document's recommendation is **arm C** — it is the only arm that meets the safety posture, and
  it is also the one the 87-platform resource constraint favours (zero LLM calls, P95 73 ms).
- **The Spec-3 learned confidence gate is not wired into anything.** `ConfidenceModelGate` and
  `ExecutionConfidence` are constructed only in tests; all four arm builders hardcode the plain
  `ConfidenceGate`. Its published frontier came from an ad-hoc run whose driver was never committed.
  *(As built, 2026-08-04: this finding was acted on — the code and its tests were deleted rather
  than wired, the plain gate having measured better in Spec 4. The frontier stays in `RESULTS.md`.)*

---

## 6. Coverage summary

| Step | Verdict | One-line reason |
|---|---|---|
| **1 — user speaks** | not covered (upstream) | no audio/ASR anywhere; boundary now documented, no serving host exists |
| **2 — segmented intent recognition** | **covered** | full segment → recognize → gate stack; set-recall 0.8194, OOD/context false-action 0.0000 |
| **3 — execute** | partial | decision + validation + barrier real; actuator is a 6-line mock and its result is discarded |
| **4a — report success** | **covered** | one composed reply, contract-metered at 1.0000; 43/92 cards omit the value set |
| **4b — explain failure cause** | **not met** | 10 machine-readable causes exist; all collapse to one of three canned lines |
| **87-platform performance** | not benchmarked | no SA8797 numbers, no memory measurement at all, GGUF path unimplemented |

---

## 7. Gap register

Ordered by what a vehicle programme should fix first, not by size.

| # | Gap | Fix shape | Blocked on |
|---|---|---|---|
| 1 | Executor result discarded at 4 sites; success spoken unconditionally | define an `Executor` protocol returning a result type; thread it into `ClauseResult` / `PlannedAction`; gate `render_response` on it | nothing |
| 2 | 4b speaks no cause | a `ValidationError.code` → Chinese phrase table read by `t2f/reply.py`, using bounds already in the cards | nothing |
| 3 | Clarification vocabulary is 3 of 17 required-param names | drive the question from `ParamSpec.description`, already parsed and unused | nothing |
| 4 | 43/92 confirmations omit the value set (boolean polarity unhearable) | boolean/enum value rendering in `render_response` | nothing |
| 5 | Mixed-utterance action spans silently dropped | add 6 missing `_OP_CUES` + 2 alias targets; give `attached_context` a reader or delete it | nothing |
| 6 | No serving host; production wiring lives in `eval/arms.py` | extract a `build_pipeline()` factory into `t2f/`; add a process entry point | arm decision (§5) |
| 7 | No 87-platform benchmark; no memory measured anywhere | implement `GgufEmbedder`/`GgufLLMClient`; measure latency, RSS, cold start, soak | **SA8797 board + Qualcomm toolchain** |
| 8 | ASR-error robustness unmeasured | add a perturbed-input eval slice (homophone, filler, dropped particle) | nothing |
| 9 | Multi-turn machinery unreachable from `route()` | decide: wire it in, or move it out of `t2f/` | product decision |
| 10 | `psutil` declared, never imported; 184 MB arm-D artifact | drop the dep or implement the memory metric; keep arm D out of any vehicle image | nothing |

Items 1–5 are contained, additive changes inside files that already exist. Item 7 is the only one
blocked on something this team does not have.

---

## 8. Non-goals of this document

It does not change any behaviour, re-run any evaluation, or revise the Spec 1–5 records, which stand
as dated accounts of what was built when. Where a historical spec contains a factual error, an errata
note is added there rather than the text being rewritten.
