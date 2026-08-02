# The Store — SQLite as the system's I/O boundary

**Date:** 2026-08-02
**Goal:** one database that holds everything arriving at the Central Model and everything it
decides, does and says — with producers writing input rows from wherever they run, and the
system returning its reply synchronously while recording it.

> **Sequencing.** This is built and **measured** before the vehicle path adopts it. Phase 3 —
> flipping the shipped runtime onto per-frame SQLite writes — is gated on the numbers in §9 and
> is not part of this work.

---

## 1. What is wrong today

The record of a run is scattered across three places and two of them do not survive the process:

| | where it lives | survives? |
|---|---|---|
| vehicle signals | `signal` table, overwritten in place | current value only |
| perception | `SceneContext`, an in-memory dict | no |
| voice | nowhere — a return value; the UI keeps a display buffer | no |
| what was executed | `operation_log` | yes |
| what was **said** | a return value | no |
| **why** — bands, verdicts, what suppressed what | `_last_reports`, cleared on the next observation | no |

So there is no artifact you could replay, audit, or generate from. You cannot ask the system
why it did something an hour ago, because it did not write that down.

A second problem follows from the first: **integrating a real producer means writing Python.**
On a vehicle the VLM runs on a different accelerator, probably in C++, on its own schedule. It
has to reach `ingest()` through a binding today.

## 2. The shape: inputs by interface, outputs by record

```
vision process ──writes rows──┐
CAN reader     ──writes rows──┼──▶ observation_raw ──▶ process_pending() ──▶ returns a reply
ASR            ──writes rows──┘         (pending)              │
                                                               └─ and writes every decision,
                                                                  action and sentence as rows
```

**Inputs are an interface.** Producers write rows and are done — separate processes, any
language, no binding, no coupling to our runtime.

**Outputs are a record.** `route()` still returns a `RouteResult` and the scene engine still
returns a `SceneOutcome`; both *also* write their rows. The reply stays synchronous, so the
50–85 ms figure this project leads with survives, and every existing caller and test keeps
working.

**Why not outputs by interface too:** the driver is waiting to be spoken to. Turning "return a
sentence" into "write a row and something notices" adds a poll interval to the number the
project is built around, and needs a worker loop that collides with SQLite's thread affinity —
already the reason the UI server is single-threaded.

## 3. Schema

### Input

```sql
CREATE TABLE observation_raw (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    at           REAL NOT NULL,
    source       TEXT NOT NULL,     -- declared in intake/sources.py
    payload      TEXT NOT NULL,     -- the caption, the transcript, the frame
    processed_at REAL,              -- NULL = pending
    error        TEXT,              -- set when processing raised; the row is still marked done
    expires_at   REAL               -- NULL = keep. Retention reads only this.
);
CREATE INDEX raw_pending ON observation_raw(processed_at, at, id);

CREATE TABLE perception (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_id     INTEGER REFERENCES observation_raw(id),
    at         REAL NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    confidence REAL NOT NULL,
    ttl        REAL NOT NULL,
    source     TEXT NOT NULL
);
CREATE INDEX perception_newest ON perception(key, at DESC);

CREATE TABLE utterance (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_id INTEGER REFERENCES observation_raw(id),
    at     REAL NOT NULL,
    text   TEXT                      -- nullable: retention clears it, the row survives
);
```

`signal` is unchanged — current vehicle state, keyed `(entity, attribute)`, overwritten in
place. Its history is in `observation_raw`, because every frame that arrived is there.

**`perception` is append-only while `signal` is overwritten, and that asymmetry is deliberate:**
a belief expires and Scene Context needs *newest per key*; a signal holds until something
commands it.

### Output

```sql
CREATE TABLE turn (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_id  INTEGER REFERENCES observation_raw(id),   -- what triggered it
    at      REAL NOT NULL,
    kind    TEXT NOT NULL,           -- route | scene | consent
    reply   TEXT NOT NULL DEFAULT '' -- what the driver heard; '' is silence, and is a fact
);

CREATE TABLE decision (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id       INTEGER NOT NULL REFERENCES turn(id),
    subject       TEXT NOT NULL,     -- the clause text, or the rule id
    verdict       TEXT NOT NULL,     -- band for routing; Verdict for a rule
    chosen        TEXT,              -- function name, or scene id
    reason        TEXT NOT NULL DEFAULT '',
    suppressed_by TEXT NOT NULL DEFAULT ''
);
```

**One `decision` table serves routing and rules** because they are the same shape: a subject, a
verdict, what was chosen, and why. Two tables would drift; the alternative — a JSON blob — is
unqueryable, and being able to ask *why* is the point.

`operation_log` gains `turn_id`, which is what connects "what we decided" to "what the car did".
That is a change to an existing table and is what the migration path is first used for.

### Meta

```sql
CREATE TABLE schema_version (version INTEGER NOT NULL);
```

`init_schema()` is `CREATE TABLE IF NOT EXISTS` and nothing else today. A store that persists
across runs and holds the only account of what happened cannot change shape without a migration
path, so one exists from the start rather than being retrofitted after the first painful change.

## 4. `process_pending`

```python
store.process_pending(now) -> list[Outcome]
```

Takes rows with `processed_at IS NULL` in `(at, id)` order and runs each through the existing
`intake` dispatch. **Explicit, never a timer** — the same discipline `intake.pump(now)` already
follows, called from the loop that owns the clock.

`processed_at` is a column rather than a separate queue table, so "pending" is a queryable state
that cannot disagree with the row it describes.

**An input that raises is marked processed with `error` set.** It must not block the queue
forever and must not vanish: a dropped input is silence, and this project treats unexplained
silence as the failure mode worth the most effort to prevent.

## 5. Scene Context becomes a query

Same class, same name, same interface — `WorldView` binds `perception.get` and
`perception.live`, and every rule sits above that. Only the backing changes.

```sql
SELECT value, confidence, source, at, ttl FROM perception
WHERE key = ? ORDER BY at DESC LIMIT 1
```

Liveness is then tested **in Python by `Observation.is_live`**, not in the `WHERE` clause.

This is the sharpest correctness detail in the whole design. `WHERE at + ttl >= now ORDER BY at
DESC` would return an older *live* row when the newest has expired — resurrecting a belief that
today reads as gone, and silently changing rule outcomes. **Newest first, liveness second**,
matching `SceneContext.get` exactly, with one definition of "live" shared across both stores
rather than one in SQL and one in Python that can drift at the boundary.

## 6. Retention and the privacy switch

`expires_at` lives on `observation_raw` **only**. One sweep covers captions, transcripts and
frames, and the parsed layer survives it — so a run stays replayable at the belief level after
the words are gone.

`--no-raw-capture` stores the parse and never the payload.

**Persisting voice on a vehicle is a data-protection decision, not only an engineering one.**
Raw transcripts are the contents of what people say in a private space, retained on a device;
for a Chinese in-vehicle product that sits under PIPL, and equivalent regimes apply elsewhere.
Designed in now this is three schema decisions. Retrofitted later it is a migration and a
disclosure.

## 7. The environment simulator

Generates plausible data for all three sources **into this schema** — which is the payoff of
having one: a mock is rows, not a second corpus format.

It writes `observation_raw` and lets `process_pending` do the rest, so simulated and real inputs
travel exactly the same path. A generator that wrote parsed rows directly would be testing a
path nothing else uses.

**Generated data encodes our beliefs, and at volume it stops looking like it does.** Twenty
hand-written rows carry an obvious caveat; ten thousand generated ones read as evidence. Every
metric measured over generated data is labelled as agreement-with-our-own-model, the same way
`data/eval/scenes.jsonl` is labelled today.

## 8. What is not changing

- `t2f/` — untouched.
- `Pipeline.route()`, `SceneEngine.observe()`, `WorldView`, every rule — unchanged signatures.
- The vehicle path — still in-memory perception, until §9 says otherwise.

## 9. What phase 2 measures

| | why it matters |
|---|---|
| write cost per perception frame | today a dict write; on the vehicle path it becomes a disk write at frame rate |
| rule evaluation: N queries vs N dict reads | runs per rule per event, the hot path of the scene engine |
| end-to-end against the 50–85 ms baseline | the number this project leads with, held across four phases |
| write volume at 10 Hz | flash wear on an automotive SoC is a real constraint, not a theoretical one |

**A bad number stops the work rather than being optimised around.** That is the point of
measuring before adopting.

## 10. Proof obligations

- `python3 -m eval.run_eval --arm C --dataset data/eval/gold.jsonl --fake --permissive` —
  **byte-identical**. Any routing metric that moves means the change reached `t2f/`.
- `python3 -m eval.run_scene_eval --arm S` — **byte-identical**. The rules must decide exactly
  what they decided before; only the plumbing beneath them changed.

Both reports are gitignored, so they must be diffed explicitly. Arm C's two latency lines are
wall-clock jitter and move between runs on an identical tree.

## 11. Deferred, and named as deferred

- **Predictive simulation.** Dropped from scope entirely.
- **Outputs as an interface.** Rejected in §2 for a stated reason, not postponed vaguely.
- **The vehicle path.** Gated on §9.
- **Caption parsing.** Still deferred; the schema is already shaped for it — raw holds the
  caption, parsed holds the interpretation, and the parse is the Stage 1 → Stage 2 step.

## 12. Risks

1. **This may be built and not adopted.** If per-frame writes cost too much, we will have a
   working implementation and a decision not to ship it. That is the accepted shape of
   measure-first, and it is far cheaper than learning it after the runtime depends on it.
2. **`--db` changes meaning.** A persisted car becomes a file containing what people said.
   Retention handles it; the change in what that flag means is stated rather than discovered.
3. **`decision` could become a JSON dumping ground.** It has five typed columns on purpose. The
   moment something does not fit, the answer is a column or a table, not a blob — a store you
   cannot query is a log file with extra steps.
4. **The completeness claim needs a test.** "The database is the record" is a property, not a
   promise: a test must assert that every path through the system leaves a row, or the gaps
   show up as silence.
