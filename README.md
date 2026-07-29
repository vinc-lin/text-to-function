# Central Model — In-Vehicle Text-to-Function Router

The **Central Model** turns a colloquial **Chinese** in-car utterance into concrete vehicle-control
function calls (name + validated parameters), dispatches them, and returns **one spoken reply**. It is
**retrieval-first and LLM-optional**: a small LLM is used only as a fallback, never as the primary
router. Targets on-device deployment (Qualcomm SA8797 / "87 platform", Qwen3-Embedding-0.6B +
Qwen3-0.6B).

> **Status:** Specs 1–7 complete (305 automated tests + 3 model-backed), **plus 9 red cases** that
> encode what the business workflow still does not meet. Step 3's actuation half is now covered by a
> SQLite-simulated vehicle, which took the red count from 11 to 9 — the cases are `xfail(strict=True)`,
> so closing a gap makes the suite say so rather than waiting to be asked. No performance number has been measured on the 87 platform. Start with
> **[the Central Model system design](docs/superpowers/specs/2026-07-25-central-model-system-design.md)**.

## Try it yourself

```bash
python3 -m cli
```

Type Chinese, watch the workflow run against a simulated car — what it recognised, the row that
moved in the vehicle database, and what the driver would hear. Switch the LLM and the confidence
gate mid-session to compare the two candidate builds against the same car.
**[Guide →](docs/TRYING_IT.md)**

## The business workflow

1. The user **speaks** a command in the car.
2. The Central Model performs **segmented intent recognition** on the utterance.
3. The **corresponding operations are executed**.
4. The system **feeds back the result** in simple dialogue — on success, the completed action; on
   failure, the specific cause.

| Step | Status | Note |
|---|---|---|
| 1 — user speaks | **upstream** | no audio/ASR here; the Central Model consumes an ASR transcript |
| 2 — segmented intent recognition | **covered** | multi-intent set-recall 0.819; OOD & context false-action 0.000 |
| 3 — execute | **covered in simulation** | validation + plan barrier + a SQLite-simulated car whose state each operation actually changes; a refusal writes nothing and is never spoken as success |
| 4a — report success | **covered** | one composed reply on every path, metric-enforced; 43/92 cards omit the value set (1 red case) |
| 4b — explain failure cause | **partial** | the car's own refusals are explained (`空调尚未开启。`); the ten *validation* causes are still computed and dropped — `reply_exact_match` **0.081** over 37 annotations (7 red cases) |
| 87-platform performance | **not benchmarked** | all figures are dev-machine (x86 + discrete GPU) |

### Scope boundary

```
 ┌───────┐   text    ┌──────────── Central Model (this repo) ────────────┐   text   ┌───────┐
 │  ASR  │──────────▶│  Pipeline.route(utterance: str) -> RouteResult    │─────────▶│  TTS  │
 └───────┘           └───────────────────────┬──────────────────────────┘          └───────┘
  not here                                   │ execute(ToolCall) -> dict            not here
                                             ▼
                                    vehicle bus adapter — not here (MockExecutor only)
```

## Why

The naïve approach — hand every utterance to an LLM to pick a tool — is slow, hallucination-prone, and
hard to run on an automotive SoC. This project shows that **strong embedding retrieval + a calibrated
confidence gate** can route most requests correctly with **zero LLM calls**, reserve a small LLM for
genuinely ambiguous cases, and — critically for a safety-critical system — **abstain rather than
execute the wrong thing.**

## Pipeline

```
utterance (ASR transcript)
  → normalize → segment into spans, label ACTION / CONTEXT
  → embed (Qwen3-Embedding-0.6B) → retrieve (multi-prototype, max-sim)  ∪  classifier candidates
  → hybrid rescore → confidence gate
      high   → deterministic param extraction → strict schema validation → execute → template reply
      medium → Qwen3-0.6B single-shot, JSON-schema-constrained tool-call → validate → execute / clarify
      low    → clarify / reject (never execute)
  → plan barrier (multi-intent): validate the whole plan, then execute the valid subset
  → compose ONE spoken reply: confirmations sentence-joined + at most one clarification
```

Execution dispatches through an injectable `execute(ToolCall) -> ExecResult` seam. `sim/` implements
it as a **SQLite-backed simulated vehicle**; a real car swaps in a bus adapter and nothing else changes.

**The DB is the car.** Rows are *signals* — `(entity, attribute)`, e.g. `window.driver/window_position`
— not functions, so `open_window` and `set_window_position` move the same physical window instead of
the car holding two contradictory beliefs about it. An operation resolves to signals, checks device
availability, then preconditions, then the signal's **physical limits** (which may be tighter than the
card's: a card says a window is 0–100, a jammed window is 0–60), writes every signal in one
transaction, and logs the attempt either way.

That makes the simulator able to **refuse** — which is the only source of requirement 4b's third
failure category, "I tried and the car refused". The driver hears `空调尚未开启。` rather than a
generic apology, and the car is left exactly as it was.

## Which build ships

`eval/arms.py` builds four configurations, and they differ materially in safety. **Arm C
(deterministic, zero LLM) is the recommended candidate build**: it is the only arm with OOD
false-execution and context false-action at 0.000, and it is also the cheapest on the target SoC.

| arm | LLM | OOD false-exec | context false-action | incorrect-exec | P95 |
|---|---|---|---|---|---|
| **C** (recommended) | none | **0.000** | **0.000** | 0.031 | 73 ms |
| C_llm | Qwen3-0.6B on medium band | 0.321 | 0.857 | 0.285 | ~1085 ms |

Arm C_llm buys parameter accuracy (param exact-match 0.72 vs 0.27, e2e 0.62 vs 0.11) at a safety cost
that is not acceptable for a vehicle without further work. Arm D adds a supervised classifier for no
measured recall gain and a 184 MB artifact; it should not enter a vehicle image.

## The specs

| Spec | What | Key result (gold test split) |
|---|---|---|
| **1 — Deterministic router** | retrieval + hybrid scoring + calibrated gate + rule param-extraction + strict validation + eval harness | recall@1 0.82 / @3 0.91; OOD & incorrect execution ≈0; P95 **72 ms**; LLM-ceiling e2e **0.845** |
| **2 — LLM fallback + classifier + multi-turn** | Qwen3-0.6B via **xgrammar**-constrained decoding; supervised classifier (Arm D); bounded multi-turn; `__ood__` prototypes + reject option | schema-valid **0.995**, param-match **0.63**, e2e **0.46**, multi-turn follow-up **1.0**; but executing the medium band via LLM leaks OOD (0.32) |
| **3 — Accuracy & safety hardening** | **learned execution-confidence gate** (LR over cheap routing features) → tunable safety/coverage frontier | safe point (τ=0.7, no LLM): OOD **0.107** (3×↓), incorrect **0.067** (5×↓), coverage 0.51, ~275 ms; balanced point hits avg-LLM-calls **0.447 (≤0.5)** |
| **4 — Multi-intent, context-aware routing** | lexical actionability filter (context suppression), plan-then-execute barrier, relative-op resolution against injectable mock vehicle state, multi-intent eval axis | multi-intent set-recall **0.819**; deterministic point: context & OOD false-action **0.000**, incorrect **0.031**, P95 **73 ms** |
| **5 — Utterance-level reply** | one spoken `RouteResult.reply` composed from what the router already produced; four contract metrics enforce it every eval run | coverage / single-question / non-empty all **1.000** on both arms; zero routing change (arm C byte-identical to baseline) |
| **7 — SQLite vehicle simulator** | the DB *is* the car: signal-keyed state, physical limits, preconditions, transactional writes, an operation log — and the ability to **refuse** | operations demonstrably change state (24.0 → 25.0); a refusal changes nothing and is spoken with its cause; red count **11 → 9** |
| **6 — End-to-end test cases** | 36 e2e cases asserting *both* what was dispatched and the exact reply, 11 of them red (`xfail(strict=True)`); 54 new eval rows carrying the failure taxonomy gold never had | `invalid_no_execution_rate` **1.000** (22 rows) — nothing unusable reaches the vehicle; `reply_exact_match` **0.081** (37 rows) — the measured distance to the workflow; gold metrics byte-identical |

Note: the Spec-3 *learned* gate is measured in `RESULTS.md` but is **not wired into any eval arm** —
all four arms construct the plain threshold `ConfidenceGate`. Treat its frontier as a research result,
not as shipped behaviour.

Full analysis and the safety/coverage frontier are in **[`docs/superpowers/RESULTS.md`](docs/superpowers/RESULTS.md)**; each spec's design and TDD plan live under `docs/superpowers/specs/` and `docs/superpowers/plans/`.

## Layout

```
t2f/          # the shipped runtime. Everything here is reachable from Pipeline.route().
  normalize · segment · embed · retrieve · score · gate · params/ · validate · respond · pipeline
  llm/        # LLMClient interface + xgrammar-constrained Qwen3-0.6B + FakeLLMClient (Spec 2)
  actionability · state · plan   # context filter, mock vehicle state, plan barrier (Spec 4)
  reply.py    # utterance-level reply composition (Spec 5)
  execute.py  # MockExecutor — the vehicle-adapter seam, stub only
sim/          # the simulated vehicle — the thing on the FAR side of the executor seam
  schema.sql · vehicle.py · mapping.py · seed.py · executor.py
research/     # measured, NOT shipped and NOT packaged — see research/README.md
  safety/     # Spec-3 learned confidence gate (no arm constructs it; the plain gate measures better)
  classify/   # Spec-2 char-ngram + embedding classifiers (Arm D only; no measured recall gain)
  dialog.py   # Spec-2 multi-turn follow-up resolver (never reachable from Pipeline.route())
data/
  catalog/    # 92 function cards across 10 domains (YAML)
  eval/       # hand-verified gold.jsonl (328) + context_negatives.jsonl (14)
              # + generated silver.jsonl + followups.jsonl
  ood/        # 100 out-of-domain / chitchat negative prototypes
eval/         # all PRD metrics, pluggable arms (C, baseline, C+LLM, D), runner
              # NOTE: also the only place a Pipeline is constructed — see gap 6 in the system design
docs/superpowers/  # specs, plans, RESULTS.md
```

## Setup & test

Core deps: `numpy pyyaml pytest`. Real models add `transformers torch` (embedder + LLM),
`scikit-learn joblib` (classifier + confidence model), `xgrammar` (constrained decoding).
(`psutil` is declared in `pyproject.toml` but currently unused — it was intended for the memory
benchmarking that is still outstanding.)

```bash
python3 -m pytest -q            # core suite (no network / no model)
python3 -m pytest -m model -q   # model-backed tests (load Qwen3 models; need GPU/network)
```

## Run the evaluation

```bash
# Fast harness sanity check (fake embedder, no model):
python3 -m eval.run_eval --arm C --dataset data/eval/gold.jsonl --fake --permissive

# Real models (calibrate the gate on dev, report on test):
python3 -m eval.run_eval --arm C        --dataset data/eval/gold.jsonl --calibrate   # Spec 1
python3 -m research.classify.train --embedding                                            # Spec 2: train classifiers
python3 -m eval.run_eval --arm C_llm    --dataset data/eval/gold.jsonl --calibrate   # Spec 2: + LLM fallback
python3 -m eval.run_eval --arm D        --dataset data/eval/gold.jsonl --calibrate   # Spec 2: classifier + LLM
```

The real embedder/LLM run via `transformers` (GPU if available). Note: on some boxes a mismatched
`torchvision` breaks `sentence-transformers`; the transformers backend neutralizes it
(`sys.modules["torchvision"] = None`) and is the default.

## Design principles

- **Retrieval-first, LLM-optional** — the target is a concrete function, never a domain name; a wrong
  domain guess never removes the correct function from the candidate pool.
- **Never execute low-confidence** — the LLM's output flows through the *same* strict validator as the
  deterministic path; the confidence gate abstains rather than execute the wrong function.
- **On-device-friendly** — small models, brute-force cosine over a small catalog, logistic regression
  over cheap features; the FP `transformers` path mirrors the eventual GGUF/llama.cpp on-device port.

## Performance posture — read this before quoting a number

Every latency figure in this repository was measured on an **x86 dev machine with a discrete GPU
(CUDA, FP16)**. None was measured on SA8797. Memory is **not measured at all** — there is no RSS,
peak, cold-start, power, thermal, or soak-stability figure anywhere. The `<1500 ms` budget the docs
compare against is a self-set engineering inference, not an 87-platform standard.

## Deferred: SA8797 on-device port

The design ports to Qualcomm SA8797 via GGUF/llama.cpp on the Hexagon NPU (GBNF replaces xgrammar for
constrained decoding) plus on-device latency/memory/crash benchmarking. It is **deferred** pending the
target hardware + Qualcomm toolchain; `GgufEmbedder` / `GgufLLMClient` mark the seam and currently
raise `NotImplementedError`. Quantisation (Q8_0 / Q4_0) is designed but not implemented — both models
run FP16 today.

---

*Built with [Claude Code](https://claude.com/claude-code) via iterative brainstorm → spec → TDD plan →
subagent-driven execution → adversarial review cycles. Evaluation numbers are measured, not estimated;
gaps vs. targets are documented with levers rather than hidden.*
