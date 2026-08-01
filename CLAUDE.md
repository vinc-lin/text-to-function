# Central Model — working notes

Chinese in-car text-to-function router. Retrieval-first, LLM-optional: a small LLM is a
fallback, never the primary router. A proactive Scene Engine sits beside it.

## Commands

```bash
python3 -m pytest -q          # ~35s, 871 tests. Model tests deselected by default.
python3 -m pytest -m model -q # 5 tests, loads Qwen3 models, needs GPU
python3 -m cli                # hand-testing session against a simulated car
python3 -m ui                 # the same session in a browser, :8770
```

**Always run from the repo root.** The catalog path is relative — a `cd` anywhere, including
inside a heredoc, gives `CatalogError: no cards found under data/catalog`.

## Before claiming behaviour did not change

```bash
python3 -m eval.run_eval --arm C --dataset data/eval/gold.jsonl --fake --permissive
python3 -m eval.run_scene_eval --arm S
```

Both must come back **byte-identical** for any change not meant to move behaviour.

Two traps. `eval_report_*.json` is gitignored, so `git status` never shows them moving — diff
them explicitly. And arm C's `p50/p95_latency_ms` are wall-clock jitter: two runs on an
identical tree differ more than most real changes do, so a moved latency line is not evidence
of anything.

`--fake` uses a hashed-ngram embedder with no semantics. It misroutes badly, and exists to
check plumbing after a code change — never to judge behaviour.

## Layering — do not invert

```
t2f    → (nothing)              the router core; its measured numbers are the evidence base
scene  → t2f
sim    → t2f
intake → scene, sim             the packaged composition root — note it does NOT import t2f;
                                it is handed a pipeline, which is why it can be the root
                                without depending on the thing it routes to
cli, ui → everything            dev tools, deliberately NOT packaged
```

`t2f/` importing `scene` or `sim` would invert the one dependency edge that has stayed clean
since the beginning. `pyproject.toml` ships `t2f eval sim scene intake`; `cli`, `ui` and
`research` are excluded on purpose.

## Two invariants everything defends

**`executor.execute` is the only path to the car.** Driver commands, scene consent and UI
buttons all route through it, so validation, preconditions, physical limits and the operation
log cannot be bypassed. `ui/actions.py` keeps `ACTIONS` (Central Model actions) and `CONTROLS`
(simulator controls) as disjoint tables, with a test — adding a route without adding an entry
is how a second path to the car gets built by accident.

**Silence is the safe default.** A stale signal, a missing model, a proposal that fails
validation, an exception mid-evaluation — all degrade to saying nothing, with the reason
recorded so the silence can explain itself.

## Gotchas

- **A broken `torchvision` install breaks `sentence-transformers`.** `t2f/embed.py:74`
  neutralizes it (`sys.modules.setdefault("torchvision", None)`). Use `TransformersEmbedder`,
  not the sentence-transformers backend.
- **The UI server is single-threaded on purpose.** SQLite thread affinity — and a threaded
  server would *lie rather than break*: `ui/state.py` wraps each pane defensively, so it would
  serve a snapshot with an empty car while everything else rendered fine.
- **Contract sweeps have silently shrunk to fit twice.** Both times a new condition type or
  rule made properties skip themselves or pass vacuously while staying green. After adding
  either, verify the sweep by mutation rather than trusting a green run.
- **The session clock is wall time, not monotonic**, because the car stamps `time.time()` and
  `--db` persists it. Mixing the two once produced an age of minus 1.78 billion seconds, which
  read as *fresh* — staleness failed open and looked wired.
- **`docs/OVERVIEW.zh.md` is hand-written by the repo owner.** Correct facts it states; do not
  restructure it.
- Dated reports (`docs/TEST_REPORT.md`, `docs/superpowers/RESULTS.md`) get **appended** update
  sections. Their historical figures stay as written — they are records, not snapshots.

## Docs conventions

**One home per number.** `docs/TEST_REPORT.md` owns current measured figures; `README.md`
summarises and links; `docs/superpowers/RESULTS.md` keeps dated per-spec records. All four
living docs once restated the same metrics and drifted into being wrong in four different
ways — adding a metric table to the README is how that starts again.

**Design goes in `docs/superpowers/specs/`.** There is no plans directory; 12.5k lines of
one-use implementation scaffolding was deleted on 2026-08-01. Dated reports get appended
update sections rather than rewrites — they are records, not snapshots.

## Where to read next

`README.md` for the whole picture · `docs/TRYING_IT.md` to drive it by hand ·
`docs/TEST_REPORT.md` for current measured numbers · `docs/superpowers/specs/` for per-feature
design and the reasoning behind each decision.
