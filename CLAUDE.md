# Central Model — working notes

Chinese in-car text-to-function router. Retrieval-first, LLM-optional: a small LLM is a
fallback, never the primary router. A proactive Scene Engine sits beside it.

## Commands

```bash
python3 -m pytest -q          # ~45s, 1127 tests. Model tests deselected by default.
python3 -m pytest -m model -q # ~45s, 74 tests, needs GPU. Mostly the e2e suite re-run on the
                              # real embedder — end-to-end routing is witnessed there, nowhere else.
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
- **An assertion inside an `xfail` body is unguarded** — the same shape as the sweep above, in a
  different mechanism, and it has now come up twice. `xfail(strict=True)` reports the same result
  whichever line trips, so a safety check sitting after a failing one is absorbed and the report
  reads green. `tests/e2e/test_s4b_failure_cause.py:73-79` found it first; s7 and s9 follow that
  pattern. The "nothing was dispatched, the car did not move" half goes in its own unconditional
  test, never beside an assertion that is expected to fail.
- **`sim/schema.sql` cannot change a table that already exists.** Every statement is
  `CREATE TABLE IF NOT EXISTS`, so a column added there is a silent no-op on every `--db` file
  written before it — fresh databases pass the whole suite while every persisted car lacks it.
  Shape changes go in `sim/migrate.py`, versioned, which also refuses a database from the future
  rather than misreading rows it does not understand.
- **Newest row first, liveness second — never both in the `WHERE`.** `Store.newest_perception`
  returns the newest row for a key expired or not, and `Observation.is_live` decides in Python.
  Filtering in SQL looks like the same query and is not: with the newest row expired it returns
  an OLDER live one, resurrecting a belief `SceneContext.get` reports as gone, and rules start
  deciding on it. One test pins this; it is worth knowing before you edit the test to match.
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
`docs/TEST_REPORT.md` for current measured numbers · `docs/superpowers/specs/README.md` indexes
the fourteen design docs and says which describe code that exists — two do not.
