# Specs — what each one covers, and whether it shipped

Fourteen design documents, named by date. That makes them sort correctly and tells you nothing
else, so this index says what each is about and — the part that matters when you are deciding
whether to trust one — **whether the code actually does what it describes**.

A spec is a design record written before or during the work. Where the implementation went
somewhere else, the spec keeps its original reasoning and carries an as-built note; it is not
rewritten to match, because the thing that was rejected is often the useful part.

Measured results are **not** here. `docs/TEST_REPORT.md` owns current figures and
`docs/superpowers/RESULTS.md` owns the dated per-spec record.

## The numbered specs

| | spec | shipped? |
|---|---|---|
| 1 | [Deterministic router + eval harness](2026-07-20-text-to-function-routing-design.md) | yes — `t2f/`, `eval/` |
| 2 | [LLM fallback, classifier, multi-turn](2026-07-21-spec2-llm-classifier-multiturn-design.md) | yes — `t2f/llm/`, xgrammar-constrained |
| 3 | [Accuracy & safety hardening](2026-07-22-accuracy-safety-hardening-design.md) | **no** — the code was deleted 2026-08-04; see below |
| 4 | [Multi-intent, context-aware routing](2026-07-24-multi-intent-context-aware-routing-design.md) | yes — `t2f/plan.py`, `t2f/actionability.py` |
| 5 | [Utterance-level reply](2026-07-25-utterance-level-reply-design.md) | yes — `t2f/respond.py` |

## After the numbered specs

| spec | shipped? |
|---|---|
| [End-to-end test cases](2026-07-26-e2e-test-cases-design.md) | yes — `tests/e2e/` |
| [Interactive session](2026-07-29-interactive-session-design.md) | yes — `cli/` |
| [Scene Engine](2026-07-30-scene-engine-design.md) | yes — `scene/` |
| [Sensed signals and animal-ahead](2026-07-31-sensed-signals-design.md) | yes — `SignalAbove`, `animal_ahead` |
| [Intake and WorldView](2026-08-01-intake-and-worldview-design.md) | yes — `intake/` |
| [The store](2026-08-02-the-store-design.md) | yes — `intake/store.py`, `sim/migrate.py`. **The vehicle path is deliberately not flipped onto it**; that decision is gated on §9 and still open. |
| [The model tier](2026-08-02-the-model-tier-design.md) | yes — `tests/e2e/conftest.py`, `test_s9_shipped_gate_cost.py` |

## Not implementations

| document | what it is |
|---|---|
| [Central Model — system design & requirement coverage](2026-07-25-central-model-system-design.md) | A framing document, not a spec. It re-states Specs 1–5 against the stakeholder's business workflow and records where the code did and did not meet it. Scoped to Specs 1–5 and dated 2026-07-25 — it describes a smaller system than the one that exists now. |
| [端云仲裁逻辑图](2026-07-26-edge-cloud-arbitration-design.md) | **A diagram, and nothing was built from it.** It says so itself: 「不含实现计划——本文不改动任何代码」. There is no edge/cloud arbitration in this repo and no module implements one. Kept because the arbitration logic it draws is the intended shape, not because it describes anything that runs. |

## Two specs whose implementation went elsewhere

Neither is a defect; both are the design record differing from what got built, which is worth
knowing before you go looking for a file.

- **Spec 3's learned confidence gate no longer exists.** `ConfidenceModelGate`,
  `ExecutionConfidence`, their 11 tests and the trained model were deleted on 2026-08-04. No eval
  arm had ever constructed one — all four builders in `eval/arms.py` use the plain threshold
  `ConfidenceGate` — and Spec 4's deterministic arm reported *better* safety with the plain gate
  than the learned one's best published point. **The spec and its measured frontier stay**:
  `RESULTS.md` is the record, and deleting an implementation does not delete evidence. Read the
  spec as a documented experiment, not as a description of code.
- **Spec 4 proposed `t2f/llm/plan_prompt.py`; it was never created.** The per-span plan path
  lives in `t2f/pipeline.py::_llm_plan`, and the prompt building stayed in `t2f/llm/prompt.py`.
  The spec also cites `Span.attached_context`, which no longer exists anywhere in the tree.
