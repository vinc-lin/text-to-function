# research/ — measured, not shipped

Code that produced numbers in [`docs/superpowers/RESULTS.md`](../docs/superpowers/RESULTS.md) but is
**not on the runtime path** and is **not packaged** (see `pyproject.toml`: only `t2f*` and `eval*` are
included).

It lives here rather than being deleted because deleting code should not delete evidence. The
measurements stay valid; this is the code behind them.

## `research/safety/` — deleted 2026-08-04

The Spec-3 learned execution-confidence gate: logistic regression over 11 cheap routing features
predicting `P(top-1 correct)`, the gate that banded on it, and the offline threshold calibrator.
**The measurements it produced are unaffected** — Spec 3's safety/coverage frontier stays in
[`RESULTS.md`](../docs/superpowers/RESULTS.md), which is the point of that file. What is gone is the
code, its 11 tests and the trained model.

Deleting it goes against this directory's own rule, so here is why it was the exception:

1. **Nothing ever constructed it.** All four builders in `eval/arms.py` hardcode the plain threshold
   `ConfidenceGate`. The frontier in RESULTS.md came from an ad-hoc run whose driver was never
   committed — so it had already stopped being reproducible, and keeping the code did not make it so.
2. **The simpler approach overtook it.** Spec 4's deterministic arm C reports OOD false-execution
   0.000, incorrect-execution 0.031 and P95 73 ms — better than the learned gate's best published
   operating point (0.107 / 0.067 / ~275 ms), which was measured against Spec 2. It was not promising
   work parked for later; it was work that lost.
3. **Its presence made a claim.** A `models/confidence.joblib` on disk and 11 green tests read as a
   mechanism that runs. It did not. That is the failure mode this repo spends most of its effort on
   — a thing that looks wired and is not — and keeping it meant keeping that impression alive.

Reviving it means re-deriving it, and would need a reason beyond "it existed once". `git log` has the
code; RESULTS.md has the numbers and the argument.

## `research/classify/` — the Spec-2 supervised candidate source

Still here, still measured, still not shipped: it is what Arm D builds
(`eval/arms.py::build_arm_d`), which is a real eval arm, unlike the above. `research/` imports from
`t2f`; nothing in `t2f` imports from `research`.
