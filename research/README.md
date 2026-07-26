# research/ — measured, not shipped

Code that produced numbers in [`docs/superpowers/RESULTS.md`](../docs/superpowers/RESULTS.md) but is
**not on the runtime path** and is **not packaged** (see `pyproject.toml`: only `t2f*` and `eval*` are
included).

It lives here rather than being deleted because deleting code should not delete evidence. The
measurements stay valid; this is the code behind them.

## `research/safety/` — the Spec-3 learned execution-confidence gate

Logistic regression over 11 cheap routing features predicting `P(top-1 correct)`, plus the gate that
bands on it and the offline threshold calibrator.

**Why it moved out of `t2f/`:**

1. **No eval arm ever constructed it.** All four builders in `eval/arms.py` hardcode the plain
   threshold `ConfidenceGate`. The frontier reported in RESULTS.md came from an ad-hoc run whose
   driver was never committed.
2. **It was costing the runtime anyway.** `t2f/gate.py` imported `confidence_features` at *module*
   level, so this whole path — including `ParameterExtractor` — loaded on every gate import while
   `confidence_features()` was only ever called from `ConfidenceModelGate.decide`, which nothing
   outside tests constructs.
3. **The plain gate now measures better.** Spec 4's deterministic arm C reports OOD false-execution
   0.000, incorrect-execution 0.031 and P95 73 ms — better than the learned gate's best published
   operating point (0.107 / 0.067 / ~275 ms).

Point 3 is the one that matters: this is not promising work parked for later, it is work the simpler
approach overtook. Reviving it needs a reason beyond "it exists".

**What still works:** the tests (`tests/test_confidence_*.py`, `tests/test_integration_spec3.py`) run
against it in place. `research/` imports from `t2f`; nothing in `t2f` imports from `research`.
