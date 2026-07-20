# Text-to-Function Router (Spec 1)

Deterministic, non-LLM Chinese vehicle-control router + eval harness.

## Setup
# Core deps (numpy, pyyaml, psutil, pytest) are required. This dev box already has them on
# the system interpreter. For the real embedder also install: pip install "sentence-transformers".

## Test
python3 -m pytest -q             # core (no network); marker '-m "not model"' is applied via pyproject
python3 -m pytest -m model -q    # model-backed tests (needs sentence-transformers + network)

## Eval
python -m eval.run_eval --arm C --dataset data/eval/gold.jsonl
