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

## Spec 2 — LLM fallback + classifier + multi-turn
The medium band is resolved by Qwen3-0.6B via `transformers` + **xgrammar** (JSON-schema-constrained,
single-shot), behind `t2f.llm.LLMClient`. A supervised classifier (`t2f/classify/`) augments retrieval
candidates (Arm D), and `t2f/dialog.py` completes clarifications over turns. Out-of-domain safety uses
`__ood__` negative prototypes (`data/ood/prototypes.txt`) + an OOD-aware gate + an LLM reject option.

```
python3 -m pip install --user scikit-learn joblib xgrammar   # one-time
python3 -m t2f.classify.train --embedding                    # train classifiers -> models/
python3 -m eval.run_eval --arm C_llm --dataset data/eval/gold.jsonl --calibrate
python3 -m eval.run_eval --arm D     --dataset data/eval/gold.jsonl --calibrate
```
See `docs/superpowers/specs/2026-07-21-spec2-*.md` and the Spec 2 section of `RESULTS.md`.

## Scope
Spec 1 (deterministic fast path) and Spec 2 (LLM fallback + classifier + multi-turn) are complete.
The SA8797 / GGUF-llama.cpp on-device port (GBNF replaces xgrammar) is Spec 3.
