# Spec 2 — LLM Fallback + Supervised Classifier + Multi-Turn Clarification

**Date:** 2026-07-21
**Status:** Approved
**Builds on:** Spec 1 (`2026-07-20-text-to-function-routing-design.md`) — the deterministic router + eval harness, complete and committed (72 tests, results in `docs/superpowers/RESULTS.md`).

---

## 0. Context & motivation

Spec 1 delivered a retrieval-first router that safely resolves the **high-confidence** band with zero
LLM calls but defers the large **medium** band (recall@1 0.82, recall@3 0.91; e2e-ceiling **0.845**).
Spec 2 makes the deferred paths actually resolve:

1. **LLM fallback** — a Qwen3-0.6B *single-shot, schema-constrained* tool-call resolver for the medium
   band, converting "hand a tight top-3 to the LLM" into real executions.
2. **Supervised classifier (Arm D)** — lightweight classifiers that *add* candidate functions
   retrieval misses (attacking the recall@3 gap) and contribute a scoring signal.
3. **Multi-turn clarification** — complete a pending call from a later user reply instead of treating
   it as a new query.

**Core principle (unchanged from Spec 1):** retrieval-first, LLM-optional, no hard domain/classifier
gate. The classifier only *adds* candidates and a signal; it never removes the correct function.

### Environment (validated)
- Python 3.10, system interpreter; `pip install --user` works (no PEP 668 block). CUDA GPU available.
- `transformers` works once the broken `torchvision` is neutralized (`sys.modules["torchvision"] = None`),
  as done by Spec 1's `TransformersEmbedder`. `sentence-transformers` remains broken (unused).
- No constrained-decoding libs installed yet (`xgrammar`, `outlines`, `lm-format-enforcer` absent);
  `llama-cpp-python` absent and hard to build here (no `cmake`/`nvcc`) — hence the transformers path.
- Qwen3-0.6B (the LLM, distinct from the embedding model) is not cached (~1.2 GB first-run download).

---

## 1. Objective

Extend the Spec 1 `Pipeline` so:
- the MEDIUM band is resolved by a real `LLMResolver` (was a stub),
- the candidate pool is augmented by a supervised classifier (Arm D),
- a clarification can be completed across turns,

reusing Spec 1's strict validator, parameter extractors, confidence gate, response templates, and
metrics harness unchanged wherever possible.

### PRD requirement coverage (this spec)

| PRD req | Spec 2 |
|---|---|
| 4 Lightweight function classifier | ✅ char-ngram-LR + embedding-LR, trained + evaluated |
| 5 Confidence gate (medium → LLM) | ✅ MEDIUM band routes to the LLM resolver |
| 6 LLM param completion | ✅ LLM completes params rules can't extract |
| 7 LLM scope (clause + top-2/3 + compact schemas + extracted params only) | ✅ enforced by prompt builder |
| 7 Compact JSON output, no XML | ✅ xgrammar schema-constrained JSON |
| 9 Multi-turn clarification resolution | ✅ bounded single-slot follow-up |
| Eval: JSON-valid, schema-valid (LLM), clarification-follow-up success | ✅ new metrics |

---

## 2. Enhanced control flow

```
utterance (+ optional SessionState with pending call)
→ if SessionState.pending exists:
     FollowUpResolver: is-follow-up? → extract missing params (rules → LLM) → complete
       ├ completed & valid → execute → template response (clears pending)
       ├ still missing & within max_turns → re-clarify (keep pending)
       └ looks like a new query → clear/park pending, fall through to routing
→ normalize → segment → per clause:
   → retrieve embedding top-k  ∪  classifier top-k         [Arm D candidate generation; union only]
   → lexical features → hybrid rescore (embedding + classifier_prob + Spec-1 signals)
   → confidence gate:
       HIGH   → deterministic params → validate → execute → template     (zero LLM; Spec 1 path)
       MEDIUM → LLMResolver:
                  build compact prompt {clause, top-2/3 candidate compact schemas, extracted params}
                  → xgrammar schema-constrained single-shot JSON
                  → Spec-1 validate_tool_call:
                       ├ valid & complete → execute → template
                       ├ missing required → clarification (+ PendingState)
                       └ invalid → one bounded retry; else reject/clarify (never execute invalid)
       LOW    → clarification / reject                                    (Spec 1 path)
→ if a clarification was emitted, return an updated SessionState for the next turn
```

---

## 3. New modules

### `t2f/llm/` — the constrained LLM resolver
- **`client.py`** — `LLMClient` ABC: `complete_tool_call(clause, candidate_cards, extracted_params) -> LLMResult`,
  where `LLMResult` is `{tool_call: ToolCall|None, clarification: str|None, raw: str, error: str|None}`.
  Backends:
  - `TransformersXGrammarClient` — loads Qwen3-0.6B via transformers on GPU (torchvision-safe), applies
    the Qwen chat template, and constrains decoding to the JSON schema via `xgrammar`. Lazily imported.
  - `FakeLLMClient` — returns scripted `LLMResult`s keyed by clause substring (deterministic, no model);
    the default in tests.
  - `GgufLLMClient` — stub (`NotImplementedError`, Spec 3 llama.cpp/GBNF).
- **`prompt.py`** — `build_prompt(clause, candidate_cards, extracted_params) -> messages`. Includes only:
  the original clause, the top-2/3 candidate **compact** schemas (name, param names+types+enums+ranges),
  and already-extracted params. No domain names, no query rewriting, no XML. (PRD §7.)
- **`schema.py`** — `candidates_to_json_schema(candidate_cards) -> dict`: a JSON Schema with
  `oneOf` over one object per candidate `{name: const, parameters: {typed props, enums, ranges, required}}`.
  This is compiled by xgrammar to force valid output and by construction guarantees `name ∈ candidates`.

### `t2f/classify/` — the supervised classifier (Arm D)
- **`features.py`** — `CharNgramVectorizer` (deterministic char 2–4 gram hashing/counts → sparse vector).
- **`classifiers.py`** — `Classifier` ABC `fit(texts, labels)`, `predict_topk(text, k) -> [(fn, prob)]`,
  `save/load`. Implementations `CharNgramLRClassifier` (LR over char n-grams) and `EmbeddingLRClassifier`
  (LR over Qwen3 query embeddings; takes an `Embedder`). Uses scikit-learn.
- **`train.py`** — CLI: load `silver.jsonl` + gold **dev** split, fit both classifiers, persist to
  `models/classifier_*.joblib`. Records train size + per-classifier dev accuracy.
- **`source.py`** — `ClassifierCandidateSource(classifier, k)`: `augment(candidates, clause) ->
  candidates`, unioning the classifier's top-k functions into the retrieval candidate pool (dedup by
  name, seeding a `classifier_prob` on each candidate). Never removes a candidate.

### `t2f/dialog.py` — multi-turn
- `SessionState{pending: PendingState|None, turn_count: int}`.
- `FollowUpResolver(extractor, llm_client, max_turns)`:
  - `is_followup(session_state, utterance) -> bool` — heuristic: pending exists AND (utterance is short /
    contains a value for a missing param / lacks a competing high-confidence function).
  - `resolve(session_state, utterance, cards_by_name) -> ClauseResult` — extract missing params from the
    reply (rules first; LLM fallback for the residual), merge into `pending.known_parameters`, validate;
    execute if complete, else re-clarify while `turn_count < max_turns`, else give up.

---

## 4. Changes to Spec 1 modules (targeted, minimal)

- **`t2f/pipeline.py`** — `DeterministicResolver`'s MEDIUM branch is replaced by a pluggable
  `MediumResolver`; a new `LLMResolver` implements it (the Spec-1 stub behavior remains available as a
  `NullMediumResolver` for the no-LLM arms). `Pipeline.__init__` gains optional `llm_client`,
  `classifier_source`; `Pipeline.route(utterance, session_state=None)` handles the follow-up entry path
  and returns `(RouteResult, SessionState)`.
- **`t2f/score.py`** — add a `classifier_prob` term to the weighted fusion (0 when no classifier).
- **`t2f/retrieve.py`** — retrieval unchanged; the pool union happens in the pipeline via
  `ClassifierCandidateSource` before rescoring.
- **`config.yaml`** — add `llm` (model_id, max_candidates=3, max_retries=1, max_new_tokens),
  `classifier` (enabled, paths, topk), `dialog` (max_turns=2), and a `classifier_prob` weight.
- **Reused unchanged:** `validate.py` (LLM output flows through the *same* strict validator — the safety
  guarantee), `params/`, `gate.py`, `respond.py` (+ `PendingState`), `normalize.py`, `segment.py`,
  `lexical.py`, `cards.py`, `embed.py`.

---

## 5. Data

- **`data/eval/followups.jsonl`** — ~40–60 hand-authored triples:
  `{initial_utterance, expected_clarification_param, followup_reply, expected_tool_call}`. Covers the
  common missing-param cases (position, temperature, level) plus a few "reply is actually a new query".
- Classifier training reuses `data/eval/silver.jsonl` (weak labels) + `gold.jsonl` **dev** split.
  **Leakage caveat:** silver rows derive from card prototype utterances, so a silver-trained classifier
  partly replicates retrieval; all classifier metrics are reported on the gold **test** split only.

---

## 6. Eval additions (`eval/`)

- **Arms:** `Arm C+LLM` (Spec-1 C with a real medium-band LLM) and `Arm D` (classifier ∪ retrieval +
  hybrid + LLM). Spec-1 arms retained for comparison.
- **New metrics** (`eval/metrics.py`): `json_valid_rate` and `llm_schema_valid_rate` (validity of LLM
  output before/after the Spec-1 validator), `clarification_followup_success` (on `followups.jsonl`),
  `candidate_gen_recall@k` (recall of the *union* pool — does the classifier lift it past retrieval's
  0.91?). Re-run end-to-end with the real LLM for **actual** (not ceiling) executable accuracy and
  **actual** avg-LLM-calls.
- **Headline questions:** how far does Arm D+LLM close the gap from e2e 0.07 toward the 0.845 ceiling;
  how much does the classifier lift candidate recall; what is the real JSON/schema-valid rate of a
  constrained 0.6B model.

---

## 7. Testing

`FakeLLMClient` (scripted results) + a fitted-on-fixtures classifier make the entire enhanced pipeline
testable with **no model or network**. Unit tests: `prompt.build_prompt` content limits; `schema`
JSON-schema shape + `oneOf`/const/enum/range; each classifier `fit`/`predict_topk`/`save`/`load` on a
tiny fixture catalog; `ClassifierCandidateSource.augment` (union, never removes, dedup); `FollowUpResolver`
(is-followup heuristic + rules completion + re-clarify bound + new-query handoff); `LLMResolver` valid /
missing-param / invalid-retry paths against `FakeLLMClient`. Model-marked tests: real
`TransformersXGrammarClient` emits schema-valid JSON, and one end-to-end medium-band resolution.

---

## 8. Non-goals

No GGUF/llama.cpp/NPU (Spec 3); no reply-LLM for chitchat/summarization beyond templates; no full
multi-intent dialogue manager (single pending slot, bounded turns); no embedder fine-tuning; Arm A
(legacy 3-stage) / Arm B (domain-embedding) remain out of scope (no such systems exist here).

---

## 9. Key risks & mitigations

- **xgrammar install/integration** → isolated behind `LLMClient`; if xgrammar won't install/work, swap
  to `lm-format-enforcer` or a JSON logits-processor without touching callers. `FakeLLMClient` keeps the
  suite green regardless of the real backend.
- **Qwen3-0.6B reliability / 1.2 GB download** → schema-constrained decoding + the Spec-1 strict
  validator + one bounded retry; never execute invalid; measure JSON/schema-valid rates rather than
  assume them.
- **Classifier silver-leakage** → evaluate on gold test only; report the caveat. The classifier is
  purely additive (candidate generator + signal), so its failure mode is "no lift," not "regression."
- **Latency** → the medium-band LLM call is the one expensive step; single-shot + short `max_new_tokens`
  keep it bounded; report real P50/P95 with the LLM in the loop (still dev-GPU, not SA8797).
