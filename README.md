# Central Model — In-Vehicle Text-to-Function Router

The **Central Model** turns a colloquial **Chinese** in-car utterance into concrete vehicle-control
function calls (name + validated parameters), dispatches them, and returns **one spoken reply**. It is
**retrieval-first and LLM-optional**: a small LLM is used only as a fallback, never as the primary
router. Targets on-device deployment (Qualcomm SA8797 / "87 platform", Qwen3-Embedding-0.6B +
Qwen3-0.6B).

> **Status:** Specs 1–9 complete, plus work that came after and is deliberately **not** a numbered
> spec — **sensed signals** and the `animal_ahead` scene, then `intake/` and `WorldView`
> ([below](#after-spec-9-what-the-numbered-specs-do-not-cover)). **860 automated tests + 5
> model-backed**, and **no red cases left**: the count went 11 → 9 → 1 → 0, and because every red
> case was `xfail(strict=True)`, closing a gap made the suite say so rather than waiting to be asked.
> No performance number has been measured on the 87 platform. Start with **[the Central Model system
> design](docs/superpowers/specs/2026-07-25-central-model-system-design.md)**; current measured
> figures live in **[`docs/TEST_REPORT.md`](docs/TEST_REPORT.md)**.

## Try it yourself

```bash
python3 -m cli
```

Type Chinese, watch the workflow run against a simulated car — what it recognised, the row that
moved in the vehicle database, and what the driver would hear. Switch the LLM and the confidence
gate mid-session to compare the two candidate builds against the same car. `python3 -m ui` is the
same session in a browser, on :8770. **[Guide →](docs/TRYING_IT.md)**

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
| 4a — report success | **covered** | one composed reply on every path, metric-enforced; a boolean confirmation states which way it went (`已为您打开车窗儿童锁。` / `已为您关闭车窗儿童锁。`). 10 of 92 cards still confirm without naming the value chosen — nine enum switches and `spray_washer` ([TEST_REPORT §10](docs/TEST_REPORT.md)) |
| 4b — explain failure cause | **covered** | all three categories are spoken with their cause — didn't understand, value unusable (`目标温度只能设置在16到32度之间。`), the car refused (`空调尚未开启。`); `reply_cause_coverage` **1.000** over 15 annotations |
| 87-platform performance | **not benchmarked** | all figures are dev-machine (x86 + discrete GPU) |

Every figure in that table is measured, and **[`docs/TEST_REPORT.md`](docs/TEST_REPORT.md) is where
they are maintained** — it carries the denominators, the re-measurement dates, and what each number
does *not* establish.

### Scope boundary

```
 ┌───────┐   text    ┌──────────── Central Model (this repo) ────────────┐   text   ┌───────┐
 │  ASR  │──────────▶│  Pipeline.route(utterance: str) -> RouteResult    │─────────▶│  TTS  │
 └───────┘           └───────────────────────┬──────────────────────────┘          └───────┘
  not here                                   │ execute(ToolCall) -> dict            not here
                                             ▼
                       vehicle bus adapter — not here. Behind the seam: `sim/` (a SQLite
                       car that really changes state and can refuse) or `MockExecutor`
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
| **C** (recommended) | none | **0.000** | **0.000** | **0.000** | 73 ms |
| C_llm | Qwen3-0.6B on medium band | 0.321 | 0.857 | 0.285 | ~1085 ms |

Arm C_llm buys parameter accuracy (param exact-match 0.72 vs 0.41, e2e 0.62 vs 0.13) at a safety cost
that is not acceptable for a vehicle without further work. Arm D adds a supervised classifier for no
measured recall gain and a 184 MB artifact; it should not enter a vehicle image.

Arm C's row is current — re-measured after the extractor and negation fixes and again after every
change since ([TEST_REPORT §5](docs/TEST_REPORT.md)). **Arm C_llm's has not been re-measured since**;
its row predates those fixes, so read the gap as a ceiling on the difference, not a current reading.

**Arms S and S_llm are not a third and fourth candidate here.** They score the Scene Engine (Spec 9),
a second top-level entry point that never routes an utterance — a build picks one row from the table
above *and*, separately, whether the proactive layer ships with its fallback attached. `scene/` has
no path into `Pipeline.route()`, so attaching or detaching the scene fallback cannot move any number
in that table.

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
| **8 — Interactive session** | `python3 -m cli` — type Chinese, watch the four workflow steps run against a session-persistent simulated car; LLM and confidence gate switchable mid-session without resetting the car | no metric: a hand-testing tool, not a shipped path. 37 tests, most over the pure `Turn → text` renderer |
| **9 — Scene Engine** | a **proactive subsystem beside the router, not a router change**: perception → declarative rules → arbitration → at most a spoken question, with the driver's consent the only path to the car; constrained LLM fallback for near-misses; its own arms **S** / **S_llm** | `scene_false_speech_rate` and `scene_false_consent_rate` **0.000**, `scene_recall` **1.000**, on both arms. Gold is hand-authored — it encodes our beliefs about perception, not measured perception |

Each row records what that spec measured **when it shipped**; current figures live in
**[`docs/TEST_REPORT.md`](docs/TEST_REPORT.md)**, which also states what the suite does *not* cover.

Note: the Spec-3 *learned* gate is measured in `RESULTS.md` but is **not wired into any eval arm** —
all four arms construct the plain threshold `ConfidenceGate`. Treat its frontier as a research result,
not as shipped behaviour.

Full analysis and the safety/coverage frontier are in
**[`docs/superpowers/RESULTS.md`](docs/superpowers/RESULTS.md)** — a per-spec record written as each
spec shipped and left as written; each spec's design and TDD plan live under
`docs/superpowers/specs/` and `docs/superpowers/plans/`.

## After Spec 9: what the numbered specs do not cover

Two pieces of work landed after Spec 9. Neither is a numbered spec — they change how facts reach the
system rather than adding a capability to the router — and both carry the same proof obligation:
`run_eval --arm C` and `run_scene_eval --arm S` come back **byte-identical**, so no routing decision
and no rule outcome moved.

**Sensed signals, and a second rule.** The car now holds a category of signal it *knows* and nothing
*commands*, starting with `vehicle.all/speed_kph`. `sim/` had until then modelled exactly what the
92 cards can write, which was right until a rule needed to read something no card produces — so the
seed guard widened from "exactly the writable signals" to "the writable signals **plus** the declared
sensed ones", both directions still enforced. A third closed condition form, `SignalAbove`, expresses
motion without giving `Signal` an operator, and `animal_ahead` (notify-only — no vehicle function
makes an animal in the road safe) is the second shipped rule. It outranks the child-lock question,
which is the first time arbitration has had anything real to arbitrate. Setting a sensed signal is a
**simulator control**, not a Central Model action: `/signal vehicle.all/speed_kph=45` in the CLI and a
Vehicle pane in the browser, both on a `CONTROLS` table kept disjoint from `ACTIONS`.

**`intake/` — one door in, one view out.** Three inputs arrived through three unrelated doors with
three different disciplines. Now every input is one `Input(source, at, payload)` whose payload type
*is* its kind, from a source that declares what it may produce, and `WorldView` is a single
read-through view over perception **and** the car — owning nothing, because a hub that stored would
rebuild the two-contradictory-beliefs problem signal-keyed state exists to prevent. `intake/` is
packaged: it is the composition root that used to live only in `cli/session.py`, which `pyproject`
deliberately does not ship.

That unblocked the defect underneath both: `updated_at` had been written on every signal write since
Spec 7 and **read by nothing**, so a dead bus and a stationary car were indistinguishable to every
rule. Sensed signals now declare a `max_age`; past it a signal reads as **absent** — identical to one
the car does not hold — so every condition on it rejects and names both ages
(`vehicle.all/speed_kph is stale (40.0s > 2.0s)`). Actuated signals never expire, and the asymmetry is
the point: a window position holds until something commands it otherwise, while a speed is a
measurement whose absence means the bus stopped. The bus is **pumped, not threaded** (SQLite thread
affinity forces it), so a live bus is fresh whenever you look and `/bus off` is what makes age
accumulate.

Design and reasoning: **[sensed signals](docs/superpowers/plans/2026-07-31-sensed-signals.md)** ·
**[intake and WorldView](docs/superpowers/specs/2026-08-01-intake-and-worldview-design.md)**.

## Layout

```
t2f/          # the shipped runtime. Everything here is reachable from Pipeline.route().
  normalize · segment · embed · retrieve · score · gate · params/ · validate · respond · pipeline
  llm/        # LLMClient interface + xgrammar-constrained Qwen3-0.6B + FakeLLMClient (Spec 2)
  actionability · state · plan   # context filter, mock vehicle state, plan barrier (Spec 4)
  reply.py    # utterance-level reply composition (Spec 5)
  execute.py  # MockExecutor — the vehicle-adapter seam, stub only
  build.py    # the ONE place a Pipeline is assembled — the session and eval arms C/C_llm share it,
              # so what you try by hand and what the metrics describe cannot drift apart
sim/          # the simulated vehicle — the thing on the FAR side of the executor seam
  schema.sql · vehicle.py · mapping.py · seed.py · executor.py
              # rows are signals: those the 92 cards can write, PLUS the declared sensed ones the
              # car knows and nothing commands (speed), which carry a max_age and can go stale
scene/        # the proactive Scene Engine (Spec 9) — a SECOND top-level entry, packaged like t2f/
  context.py · rules.py · engine.py · consent.py · llm.py · speech.py
              # perception in, at most a question out; consent is the only path to the car, and
              # the two subsystems meet only at execute(ToolCall). It cannot reach Pipeline.route()
              # two rules ship: animal_ahead (notify-only) and rear_child_window_lock (asks)
intake/       # one door in, one view out — packaged, and the composition root a real integration
  envelope.py · sources.py · hub.py · ingest.py
              # every input arrives as one Input(source, at, payload) and is handed to the module
              # that owns the decision; WorldView is the single read-through view over perception
              # AND the car, which is what lets a sensed signal go stale instead of being believed
              # forever. Assembling router + scene engine + car used to live only in cli/session.py
cli/          # python3 -m cli — the hand-testing session (Spec 8); see docs/TRYING_IT.md
  __main__.py · session.py · render.py    # loop · utterance→Turn · pure Turn→text
              # a dev tool: run from the repo, NOT packaged
              # (pyproject ships t2f/ eval/ sim/ scene/ intake/)
ui/           # python3 -m ui — the same session in a browser; see docs/TRYING_IT.md
  state.py · actions.py · server.py · page.html
              # snapshot · the five actions and two controls · routes · one self-contained page
              # stdlib only, single-threaded on purpose, and NOT packaged either
research/     # measured, NOT shipped and NOT packaged — see research/README.md
  safety/     # Spec-3 learned confidence gate (no arm constructs it; the plain gate measures better)
  classify/   # Spec-2 char-ngram + embedding classifiers (Arm D only; no measured recall gain)
  dialog.py   # Spec-2 multi-turn follow-up resolver (never reachable from Pipeline.route())
data/
  catalog/    # 92 function cards across 10 domains (YAML)
  eval/       # hand-verified gold.jsonl (328) + context_negatives.jsonl (14)
              # + generated silver.jsonl + followups.jsonl
              # + scenes.jsonl (20) — hand-authored scene events, beliefs about perception
  ood/        # 100 out-of-domain / chitchat negative prototypes
eval/         # all PRD metrics, pluggable arms (C, baseline, C+LLM, D), runner
              # run_scene_eval.py + scene_metrics.py — the S / S_llm arms, a separate runner
              # arms C and C_llm now call t2f/build.py; baseline and D stay here — this package
              # builds experiments, t2f/build.py builds the product (closes gap 6)
docs/
  TRYING_IT.md   # the hands-on guide to the interactive session
  TEST_REPORT.md # THE HOME OF CURRENT MEASURED NUMBERS — the 131 end-to-end cases, what a driver
                 # actually hears, and what is NOT covered
  OVERVIEW.zh.md # the system in two minutes, in Chinese — hand-written
  superpowers/   # specs, plans, RESULTS.md (per-spec records, kept as each spec shipped)
```

## Setup & test

Core deps: `numpy pyyaml pytest`. Real models add `transformers torch` (embedder + LLM),
`scikit-learn joblib` (classifier + confidence model), `xgrammar` (constrained decoding).

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

# The Scene Engine has its own runner and its own gold file; no embedder on this path:
python3 -m eval.run_scene_eval --arm S        # Spec 9: rules only
python3 -m eval.run_scene_eval --arm S_llm    # Spec 9: + the constrained fallback
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
