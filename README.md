# Text-to-Function Router for In-Vehicle Voice Control

A **retrieval-first, LLM-optional** pipeline that maps a colloquial **Chinese** in-car utterance
directly to a concrete vehicle-control function call (name + validated parameters), executes it, and
returns a templated confirmation — using a small LLM only as an *optional fallback*, never as the
primary router. Targets on-device deployment (Qualcomm SA8797, Qwen3-Embedding-0.6B + Qwen3-0.6B).

> **Status:** Specs 1–3 complete (119 automated tests). The SA8797 on-device port is designed but
> deferred pending hardware. Built as a reference implementation with honest, reproducible evaluation.

## Why

The naïve approach — hand every utterance to an LLM to pick a tool — is slow, hallucination-prone, and
hard to run on an automotive SoC. This project shows that **strong embedding retrieval + a calibrated
confidence gate** can route most requests correctly with **zero LLM calls**, reserve a small LLM for
genuinely ambiguous cases, and — critically for a safety-critical system — **abstain (ask for
clarification) rather than execute the wrong thing.**

## Pipeline

```
utterance
  → normalize → multi-intent split
  → embed (Qwen3-Embedding-0.6B) → retrieve (multi-prototype, max-sim)  ∪  classifier candidates
  → hybrid rescore → learned confidence gate
      high   → deterministic param extraction → strict schema validation → execute → template reply
      medium → Qwen3-0.6B single-shot, JSON-schema-constrained tool-call → validate → execute / clarify
      low    → clarify / reject (never execute)
  → (multi-turn) complete a pending clarification from the next reply
```

## The three specs

| Spec | What | Key result (gold test split, n=184) |
|---|---|---|
| **1 — Deterministic router** | retrieval + hybrid scoring + calibrated gate + rule param-extraction + strict validation + eval harness | recall@1 0.82 / @3 0.91; OOD & incorrect execution ≈0; P95 **72 ms**; LLM-ceiling e2e **0.845** |
| **2 — LLM fallback + classifier + multi-turn** | Qwen3-0.6B via **xgrammar**-constrained decoding; supervised classifier (Arm D); bounded multi-turn; `__ood__` prototypes + reject option | schema-valid **0.995**, param-match **0.63**, e2e **0.46**, multi-turn follow-up **1.0**; but executing the medium band via LLM leaks OOD (0.32) |
| **3 — Accuracy & safety hardening** | **learned execution-confidence gate** (LR over cheap routing features) → tunable safety/coverage frontier | safe point (τ=0.7, no LLM): OOD **0.107** (3×↓), incorrect **0.067** (5×↓), coverage 0.51, ~275 ms; balanced point hits avg-LLM-calls **0.447 (≤0.5)** |

Full analysis and the safety/coverage frontier are in **[`docs/superpowers/RESULTS.md`](docs/superpowers/RESULTS.md)**; each spec's design and TDD plan live under `docs/superpowers/specs/` and `docs/superpowers/plans/`.

## Layout

```
t2f/
  normalize · segment · embed · retrieve · score · gate · params/ · validate · respond · pipeline
  llm/        # LLMClient interface + xgrammar-constrained Qwen3-0.6B + FakeLLMClient (Spec 2)
  classify/   # char-ngram + embedding LR classifiers, candidate source, training (Spec 2)
  dialog.py   # bounded multi-turn follow-up resolver (Spec 2)
  safety/     # execution-confidence features + model + calibration (Spec 3)
  tools/      # hard-negative mining (Spec 3)
data/
  catalog/    # 92 function cards across 10 domains (YAML)
  eval/       # hand-verified gold.jsonl (312) + generated silver.jsonl + followups.jsonl
  ood/        # 96 out-of-domain / chitchat negative prototypes
eval/         # all PRD metrics, pluggable arms (C, baseline, C+LLM, D), runner
docs/superpowers/  # specs, plans, RESULTS.md
```

## Setup & test

Core deps: `numpy pyyaml psutil pytest`. Real models add `transformers torch` (embedder + LLM),
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
python3 -m t2f.classify.train --embedding                                            # Spec 2: train classifiers
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

## Deferred: SA8797 on-device port

The design ports to Qualcomm SA8797 via GGUF/llama.cpp on the Hexagon NPU (GBNF replaces xgrammar for
constrained decoding) plus on-device latency/memory/crash benchmarking. It is **deferred** pending the
target hardware + Qualcomm toolchain; `GgufEmbedder` / `GgufLLMClient` stubs mark the seam.

---

*Built with [Claude Code](https://claude.com/claude-code) via iterative brainstorm → spec → TDD plan →
subagent-driven execution → adversarial review cycles. Evaluation numbers are measured, not estimated;
gaps vs. targets are documented with levers rather than hidden.*
