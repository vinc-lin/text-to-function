# Text-to-Function Router (Spec 1)

Deterministic, non-LLM Chinese vehicle-control router + eval harness. Maps a colloquial
Chinese in-car utterance to a validated `tool_call` (or a clarification) with zero LLM calls,
using multi-prototype embedding retrieval + a calibrated confidence gate. See
`docs/superpowers/specs/2026-07-20-text-to-function-routing-design.md` for the design and
`docs/superpowers/RESULTS.md` for evaluation results.

## Layout
- `t2f/` — pipeline stages (normalize → segment → embed → retrieve → score → gate → params → validate → respond)
- `data/catalog/` — 92 function cards across 10 domains
- `data/eval/` — hand-verified `gold.jsonl` (312 rows) + generated `silver.jsonl`
- `eval/` — metrics, arms (Arm C hybrid + baseline), runner

## Setup
Core deps (numpy, pyyaml, psutil, pytest) are required. This dev box already has them on the
system interpreter, so no install is needed to run the core suite. The real embedder uses
`transformers` + `torch` (already present here); on a fresh box: `pip install transformers torch`.

## Test
```
python3 -m pytest -q             # core (no network / no model); '-m "not model"' applied via pyproject
python3 -m pytest -m model -q    # model-backed test (loads Qwen3-Embedding-0.6B; needs GPU/network)
```

## Eval
```
# Fake embedder (fast, no model) — harness sanity check:
python3 -m eval.run_eval --arm C --dataset data/eval/gold.jsonl --fake --permissive

# Real Qwen3-Embedding-0.6B, calibrate the gate on the dev split, report on test:
python3 -m eval.run_eval --arm C        --dataset data/eval/gold.jsonl --calibrate
python3 -m eval.run_eval --arm baseline --dataset data/eval/gold.jsonl --calibrate
```
The real embedder runs via `t2f.embed.TransformersEmbedder` (last-token pooling, GPU if available).
Note: on this box a broken `torchvision` install breaks `sentence-transformers`; the transformers
backend neutralizes it (`sys.modules["torchvision"] = None`) and is the default `--backend`.

## Scope
Spec 1 is the deterministic fast path. The Qwen3-0.6B single-shot LLM fallback + supervised
classifier (Arm D) + multi-turn clarification are Spec 2; the SA8797 / GGUF-llama.cpp on-device
port is Spec 3.
