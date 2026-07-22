# Spec 3 — Accuracy & Safety Hardening

**Date:** 2026-07-22
**Status:** Approved
**Builds on:** Spec 1 (deterministic router) + Spec 2 (LLM fallback + classifier + multi-turn), both complete (`docs/superpowers/RESULTS.md`).
**Note:** The SA8797 on-device port (GGUF/llama.cpp Hexagon NPU) is a separate spec, **deferred** until target hardware + the Qualcomm toolchain are available (none exist in the current x86 environment). This spec is fully buildable and testable here against the existing gold set.

---

## 0. Motivation

Spec 2's LLM fallback turned the router into an executing system (schema-valid 0.995, param-match 0.63, e2e 0.07→0.46, multi-turn 1.0), but executing the medium band reintroduced execution risk that three stacked heuristics only partly contained:

- **OOD false-execution 0.32** (target ≈ 0) — out-of-domain requests still execute something.
- **Incorrect-execution 0.34** (target ≈ 0) — the executed function is wrong.
- **Recall@3 0.91** (target 0.97) — 9% of the time the correct function isn't even in the candidate pool handed to the LLM, which *forces* a wrong execution.

**Key insight:** OOD false-execution and incorrect-execution are the same failure — *"executed something wrong."* Both are best handled by one mechanism: a **learned execution-confidence gate** that predicts whether executing the top candidate would be correct and **abstains (clarifies) below a safety-calibrated threshold**, replacing Spec-1's hand-tuned score thresholds and Spec-2's OOD-marker special-casing with a calibrated model. Recall hardening then enlarges the confident set so abstaining costs less coverage. The PRD explicitly prefers "clarify rather than incorrectly execute," so biasing toward abstention is aligned.

---

## 1. Objective

Drive **OOD false-execution and incorrect-execution toward ≈ 0** while preserving as much executable coverage as possible, and **lift recall@3 toward 0.97**. Deliverables are lightweight and on-device-friendly (logistic regression over cheap features; no new large models), so they port cleanly in the later SA8797 spec.

---

## 2. Unifying design — the learned execution-confidence gate

Compute `P(top-1 candidate is the correct function)` from cheap, interpretable routing features. Band on that probability:
- `P ≥ τ_high` → **execute** (deterministic HIGH, or LLM-confirmed).
- `τ_low ≤ P < τ_high` → **medium** → LLM fallback (Spec 2), still gated by execution-confidence before executing the LLM's output.
- `P < τ_low` → **abstain → clarify / reject** (never execute).

One calibrated model governs both OOD rejection (OOD → low P, since no candidate is the gold function) and incorrect-execution (wrong-pick features → low P). `τ` is a documented operating point on a **safety/coverage frontier**, configurable per deployment.

---

## 3. Components

### 3.1 Confidence features — `t2f/safety/features.py`
`confidence_features(clause, candidates, lex, classifier_probs=None) -> dict[str, float]`. Pure function; features:
- `top1_score`, `margin` (top1 − top2), `top3_spread`
- `ood_marker_sim` — similarity to the `__ood__` marker if present in candidates (0 otherwise)
- `top1_param_compat` — top-1 candidate's `param_compat` signal
- `classifier_prob`, `classifier_entropy` — from `classifier_probs` (0 / max-entropy if absent)
- `n_candidates`, `query_len`
- `has_required_params` — 1.0 if all of the top-1's required params are extractable from the clause
- `domain_kw_hit` — 1.0 if a domain keyword for the top-1's domain appears

Unit-tested on fixtures (deterministic; no model).

### 3.2 Execution-confidence model — `t2f/safety/confidence.py`
`ExecutionConfidence`:
- `fit(feature_rows, labels)` — logistic regression (scikit-learn) over the feature dict (stable feature ordering).
- `predict_proba(features) -> float` — `P(correct)`.
- `calibrate(dev_examples, route_fn, target_error=0.05) -> (tau_low, tau_high)` — routes each dev row once (**LLM-independent**), builds features, and picks thresholds: `τ_high` = lowest P such that executed-error (wrong ∪ OOD among P≥τ_high) ≤ `target_error` while maximizing executed coverage; `τ_low` = the OOD floor (P below which OOD dominates). Returns a `ConfidenceThresholds` dataclass.
- `save/load` (joblib).

**Training label** (LLM-independent): for each gold-**dev** clause, `label = 1` if the retrieval top-1 function is a gold function, else `0`; OOD rows are `0`. This predicts routing correctness, which is the execution-correctness proxy — trainable with no LLM calls.

### 3.3 Integration — `t2f/gate.py` (+ pipeline)
A `ConfidenceModelGate` implementing the same `decide(candidates, features, cards_by_name) -> Decision` interface as `ConfidenceGate`, but banding on `ExecutionConfidence.predict_proba(...)` with `(τ_low, τ_high)`. It records `P(correct)` in the decision features. The existing OOD-marker short-circuit and the LLM `__reject__` path remain as defense-in-depth. `Pipeline`/arms accept an optional confidence gate; when absent, behavior is unchanged (Spec-1/2 `ConfidenceGate`). The medium-band `LLMResolver` additionally checks execution-confidence of the LLM's *chosen* function before executing (abstain if below τ_low).

### 3.4 Recall hardening — `t2f/tools/mine_hard_negatives.py`
Offline analysis: route gold-**dev**; for every clause where the correct function ranks below a sibling, record the (gold, distractor) confusion. Emit `data/analysis/hard_negatives.md` (confusable clusters + counts) and a suggested-prototype list. Author a small number of **discriminative** prototypes/aliases into `data/catalog/*.yaml` for the top confusable clusters (dev-guided; no test leakage). Optionally add a `param_compat` refinement that *penalizes* functions whose required param types are absent (the Spec-1 finding: uniform-positive param_compat didn't discriminate). Re-measure recall@1/@3 on test.

---

## 4. Data
Reuse gold (dev = train/calibrate, test = report only), silver, and the 96 OOD prototypes (now also feature inputs and negative training samples). Hard-negative mining may add small, dev-guided prototype/alias additions to the catalog. **No test-set leakage** — all training/calibration is on dev + silver + OOD prototypes.

---

## 5. Eval — the safety/coverage frontier
- New metrics in `eval/metrics.py`: `coverage` (fraction of in-scope requests executed) and a `frontier(records, taus)` helper.
- `eval/run_eval.py`: a `--confidence` mode that builds the learned gate; report OOD false-execution, incorrect-execution, e2e-executable, and clarification-rate **across a sweep of τ**, plus the recommended operating point.
- Re-run Arm C, C+LLM, D with the learned gate; add a **"Spec 3"** section to `RESULTS.md` with the frontier table.
- **Targets:** at the recommended τ, OOD false-execution ≤ 0.05 and incorrect-execution ≤ 0.05, with the resulting e2e-executable and clarification-rate reported honestly (the coverage cost). Recall@3 lift from hard-negative hardening reported vs the Spec-2 baseline. If a target isn't met, note the gap + lever (do not silently pass).

---

## 6. Non-goals
No on-device/GGUF/NPU (deferred SA8797 spec). No new large models — no cross-encoder reranker, no embedder fine-tune (keep it LR over cheap features, on-device-friendly). No new LLM or new dataset collection. No changes to the Spec-1/2 public metric definitions (only additions).

---

## 7. Key risks & mitigations
- **Learned gate overfits dev** → few interpretable features, calibrate with a margin, report only on held-out test, and compare against the Spec-2 heuristic gate as a baseline (must not regress).
- **Coverage collapse** (abstaining too much) → the frontier makes the OOD/incorrect vs coverage tradeoff explicit; recall hardening enlarges the confident set; τ is configurable per deployment.
- **Recall hardening is data authoring** → strictly dev-guided, small targeted additions (not wholesale rewrites), measured on test; report the honest lift even if modest.
- **Feature/label leakage** → training label uses retrieval top-1 vs gold on dev only; OOD prototypes are index/feature inputs, never copied into gold; hard-negatives mined from dev only.
