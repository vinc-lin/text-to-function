# Spec 1 — Deterministic Text-to-Function Router + Eval Harness

**Date:** 2026-07-20
**Status:** Approved
**Scope:** First slice of the Text-to-Function Routing system for in-vehicle control.

---

## 0. Context

We are building an **on-device text-to-function router for in-vehicle voice control**: it maps a
colloquial Chinese utterance directly to a concrete vehicle-control function call (name + validated
parameters), executes it, and returns a templated confirmation — using a small LLM only as an
*optional fallback*, never as the primary router.

**Core principle — retrieval-first, LLM-optional.** The target output is a concrete function, not a
domain name. Domain is a ranking feature only; a wrong domain guess must never remove the correct
function from the candidate pool.

The full system targets Qualcomm **SA8797** (Snapdragon automotive cockpit SoC, Android) running
Qwen3-Embedding-0.6B (retrieval) and Qwen3-0.6B (fallback). This spec is **Spec 1 of three**:

- **Spec 1 (this doc):** the deterministic, non-LLM fast path + eval harness, in Python, mirroring
  on-device constraints. Delivers PRD **Arm C** (target router) + **Arm C-baseline** (naive retrieval).
- **Spec 2 (later):** Qwen3-0.6B single-shot constrained-decoding fallback + lightweight supervised
  classifier (full Arm D) + multi-turn clarification resolution.
- **Spec 3 (later):** SA8797 port (GGUF/llama.cpp Hexagon NPU) + on-device latency/memory benchmarking.

### Deployment constraints that shaped this design (from research, 2025–2026)

These are validated findings that Spec 1 mirrors so the later port is a port, not a rewrite:

- **Runtime:** on-device path is **llama.cpp** (GGUF) on the Hexagon NPU (HTP), directly or via
  Qualcomm **GenieX**. The Hexagon backend is officially *experimental* and SA8797P is not on the
  tested NPU-arch list — must be validated on the board (Spec 3 concern).
- **Quantization (asymmetric):** embedding → **Q8_0** (embeddings are quant-sensitive; ~99% retained
  at INT8; no official Q4 shipped). LLM → **Q4_0 + imatrix** (Q4_K_M falls back to CPU on HTP).
- **Embedding serving:** precompute function-card prototype embeddings **offline**; on device encode
  only the short query; **brute-force cosine** over the small catalog (no HNSW/faiss needed).
  Qwen3-Embedding-0.6B supports **Matryoshka** dims (32–1024) → vectors truncatable for cheaper search.
- **Latency/memory:** NPU gives ~10× prefill advantage on short inputs; cold-start load is ~10× slower
  than warm → keep models resident. Avoid extreme KV-cache quantization (degrades tool-calling).
  The <1.5 s target is a reasonable engineering inference but is **not measured on SA8797P**.
- **Tool-calling:** Qwen3-0.6B **collapses on multi-turn tool-calling (1.38% BFCL)**; single-shot is
  only serviceable (~59–68%). Therefore the LLM (Spec 2) is restricted to **one grammar-constrained
  JSON tool-call per invocation** and retrieval must carry the load. **We commit to Qwen3-0.6B** and
  compensate with strong retrieval + strict validation + clarification.

---

## 1. Objective for Spec 1

A Python reference implementation of the **non-LLM fast path** that takes a colloquial Chinese
utterance and produces validated `tool_call`(s) — or a clarification — with **zero LLM calls**, plus a
metrics harness that measures it against the PRD acceptance criteria.

### PRD requirement coverage

| PRD requirement | Spec 1 | Deferred |
|---|---|---|
| 1 Multi-intent split | ✅ rule-based | — |
| 2 Function-level multi-prototype retrieval | ✅ | — |
| 3 Hybrid scoring | ✅ | — |
| 4 Lightweight classifier | interface hook only | ✅ training = Spec 2 |
| 5 Confidence gate | ✅ calibrated, multi-feature | — |
| 6 Rule param extraction | ✅ | LLM param completion = Spec 2 |
| 7 LLM scope | stub/interface only | ✅ Qwen3-0.6B = Spec 2 |
| 8 Schema validation | ✅ | — |
| 9 Multi-turn clarification | pending-state *model* + emit clarification | ✅ follow-up resolution = Spec 2 |
| 10 Template responses | ✅ | reply-LLM = Spec 2 |

---

## 2. Architecture — pipeline stages

Each stage is an isolated module with a clean interface, independently testable.

```
raw utterance
 → Normalizer            normalize CN text (full/half-width, punctuation, whitespace)
 → IntentSplitter        rule-based split on CN punctuation + conjunctions → [clauses]
 for each clause:
 → Embedder(interface)   Qwen3-Embedding-0.6B; query instruction prefix; FP backend now
 → Retriever             cosine vs ~1k precomputed prototype vectors → top-K functions
                          (max-sim per function; MRL-truncatable dims)
 → LexicalFeatures       extract numbers/units/positions/operations/on-off/min-max once
 → SignalExtractors      keyword+alias · param-compat · number/unit/position/op · domain prior
 → Scorer (fusion)       weighted linear fusion → ranked candidates + per-signal breakdown
 → ConfidenceGate        features {top1, top1−top2 margin, param-compat, OOD} → high/med/low
 → ParamExtractor        deterministic params from schema + lexical features → {params, missing}
 → SchemaValidator       strict: exists · required · no-unknown · types · enums · ranges · safety
 → Resolver              high→execute · medium→needs-LLM(stub) · low→clarification
 → Executor(mock) + ResponseTemplate → CN response text
 → aggregate → RouteResult[] with full trace
```

### Data flow notes

- The full per-clause **trace** (candidates, per-signal scores, gate decision, extracted params,
  validation result, response) is a first-class output — required for eval and debugging.
- Prototype embeddings are computed once and persisted; only the query is encoded per request.

---

## 3. Module boundaries (package `t2f/`)

- **`cards.py`** — `FunctionCard` model: name, domain, description, params (type/enum/range/required),
  aliases, prototype utterances (typical + colloquial), hard-negatives. Loaded from `data/catalog/*.yaml`.
- **`normalize.py`** — `normalize(str) -> str`. Full/half-width folding, punctuation unification,
  whitespace, latin lowercasing.
- **`segment.py`** — `split(str) -> list[str]`. Conservative rule-based multi-intent split on CN
  punctuation (，、。；！) + conjunctions (和 / 还有 / 然后 / 并 / 同时 / 把…和…). Guards against
  over-splitting (e.g. must not split inside "调到25度").
- **`embed.py`** — `Embedder` interface (`encode(texts, is_query) -> np.ndarray`);
  `SentenceTransformerEmbedder` (FP, used now) + `GgufEmbedder` stub (Spec 3). Applies the Qwen3
  query instruction prefix to queries; prototypes encoded without it.
- **`retrieve.py`** — `PrototypeStore` (build/persist prototype embeddings for all cards) + `Retriever`
  (brute-force cosine, function-level **max-sim** aggregation, top-K, MRL dim truncation option).
- **`signals/`** — one module per hybrid signal, each `score(clause, features, card) -> float`:
  `keyword_alias.py`, `param_compat.py`, `lexical.py` (number/unit/position/operation matching),
  `domain_prior.py` (soft prior — never a gate).
- **`score.py`** — `Scorer.fuse(embedding_score, signals) -> ranked candidates + per-signal breakdown`
  with config-driven weights; plus `EmbeddingOnlyScorer` for the baseline arm.
- **`gate.py`** — `ConfidenceGate.decide(candidates, features) -> Decision{band, chosen, candidates,
  ood_score}`. Features = {top-1 score, top1−top2 margin, param-compatibility of top-1, OOD score}.
  Calibrated on the gold **dev** split. **Not** a single global cosine threshold. OOD detection combines
  a score threshold with OOD/chitchat negative prototypes in the index.
- **`params/`** — deterministic extractors, each keyed to a schema parameter type:
  CN + Arabic numerals (二十五→25, 两/俩), temperature (度/°C, 最高/最低), percentage (百分之/%),
  position (主驾/副驾/主驾驶/副驾驶/左/右/前/后排/全车), direction, on-off (打开/关/开), increase/decrease
  (调高/调低/大一点/小一点), level (档/级/挡), min/max (最大/最小/最高/最低). Output `{params, missing}`.
- **`validate.py`** — `validate(fn, params, candidate_set) -> ToolCall | Errors`. Strict: function name
  must be in candidate set, required params present, no unknown params, type match, enum valid, numeric
  in range, safety/conflict rules pass. Invalid output is never executed.
- **`respond.py`** — template response generation (Chinese), keyed by function + params; also the
  `PendingState` model (`pending_function`, `known_parameters`, `missing_parameters`) and clarification
  templates.
- **`pipeline.py`** — orchestrator wiring all stages; returns `RouteResult` (list, one per clause) with
  full trace. Resolver policy is pluggable: `DeterministicResolver` for Spec 1; LLM resolver = Spec 2.
- **`config.yaml`** — signal weights, gate thresholds, embedding model id, MRL dims, instruction templates.

**Design rule:** each module answers "what does it do / how do you use it / what does it depend on"
in isolation. Consumers depend on interfaces, not internals.

---

## 4. Data plan (`data/`)

- **`catalog/*.yaml`** — **80+ function cards** across ~8 domains (climate, windows, seats, media,
  lights, doors/locks, navigation, misc). Card schema documented in `cards.py`. Deliberately includes
  **confusable clusters** (e.g. set-temperature vs fan-level vs seat-heating) to stress hybrid scoring
  and hard-negative handling at production scale.
- **Generation pipeline** (`data/gen/`, scripts + prompts, reproducible) — a strong LLM generates per
  function: colloquial utterances, hard-negatives, multi-intent combinations, and OOD/chitchat. These
  populate card prototypes **and** form a larger *silver* eval set.
- **`data/eval/gold.jsonl`** — **~300–500 hand-verified** examples, labeled:
  `{utterance, expected_functions[], expected_params, type ∈ {single, multi_intent, ood, ambiguous}}`.
  Split into **dev** (gate + weight tuning) and **test** (reported metrics). Silver and gold metrics
  are reported separately.

---

## 5. Eval harness (`eval/`)

- **`metrics.py`** — Function Recall@1/@3, multi-intent function-set recall, parameter exact-match,
  schema-valid rate, end-to-end executable accuracy, out-of-domain false-execution rate, incorrect-
  execution rate, clarification rate, average LLM calls per request, P50/P95 latency, peak/resident
  memory (`tracemalloc`/`psutil`), crash/timeout rate. *(JSON-valid rate and clarification-follow-up
  success are Spec 2.)*
- **Medium-band handling (best-practice split):** report **two** end-to-end numbers —
  1. *deterministic-only* — only high-band executes (medium/low do not) → measures how far we get with
     **zero** LLM;
  2. *LLM-ceiling* — medium band credited if the correct function is within top-3 → the ceiling a
     perfect LLM tie-break could reach.
  This directly informs the "≤0.5 LLM calls per single-intent request" acceptance criterion.
- **`arms.py`** — pluggable routing arms: **Arm C** (target: multi-prototype retrieval + hybrid
  rescoring + calibrated gate) and **Arm C-baseline** (pure embedding + single threshold). Extensible
  for Arms B/D later.
- **`run_eval.py`** — runs an arm over a dataset → JSON + Markdown report against acceptance thresholds.
  Latency is recorded but **flagged as dev-machine, not SA8797**.

### Acceptance targets (measured by the harness; on-device numbers are Spec 3)

Function Recall@1 ≥ 90% · Recall@3 ≥ 97% · e2e executable ≥ 80% · schema-valid ≥ 99% ·
incorrect-execution ≈ 0% · OOD false-execution ≈ 0% · deterministic-only path resolves enough
high-confidence traffic to keep projected LLM calls ≤ 0.5 per single-intent request.

---

## 6. Testing (TDD)

- Unit tests per stage: normalizer, splitter, every parameter extractor, validator, scorer, gate.
- Golden pipeline tests on a small fixture catalog (deterministic, no network).
- The eval harness runs as an integration test emitting the metrics report.
- Chinese numeral/position parsing gets dedicated edge-case tables.
- Tests must not require network or model downloads by default (embedder is mockable via the interface;
  a marked slow test may exercise the real FP model).

---

## 7. Non-goals (explicit)

No LLM invocation (resolver is a stub); no classifier training; no multi-turn follow-up loop
(pending-state model only); no SA8797 / GGUF / NPU (interfaces only); no real vehicle execution
(mock executor); no reply-LLM (chitchat/summarization).

---

## 8. Key risks & mitigations

- **Synthetic-data optimism** → gold set hand-verified; silver/gold reported separately; hard-negatives
  + OOD kept in eval.
- **FP-vs-Q8_0 numerics gap** → flagged; `Embedder` interface allows GGUF swap; recall may shift on port.
- **Gate overfitting to dev split** → held-out test split; keep the gate to a few features.
- **Chinese parsing edge cases** → comprehensive extractor unit tables + per-extraction confidence flags.
- **Retrieval at 80+ functions** → confusable clusters intentionally included in the catalog and eval so
  hybrid scoring is stressed, not flattered.
