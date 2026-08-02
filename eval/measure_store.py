"""What the store costs — the four numbers §9 of the design gates phase 3 on.

    write cost per perception frame · rule evaluation, N queries vs N dict reads ·
    end-to-end against the 50–85 ms baseline · write volume at 10 Hz

**This file is written to run in two trees.** The "before" figures are not estimates and not a
reimplementation of the dict that used to back `SceneContext`: they come from running this same
script inside a worktree checked out at the last commit before the store existed, where
`SceneContext()` still takes no argument. Every measurement here goes through a public API that
exists in both trees — `SceneContext.update/get/live`, `Intake.ingest`, `Session.build` — so the
delta between the two runs is the store and nothing else. Sections with no meaning in the old
tree say so and skip themselves rather than printing a zero.

    git worktree add --detach <dir> <pre-store-commit>
    cp eval/measure_store.py <dir>/eval/
    cd <dir> && python3 -m eval.measure_store --only frame,reads,e2e

**Every figure is a median over repetitions, and the count is printed beside it.** Nothing here
is a single sample: this box is a WSL2 VM on a Windows host with 24 hyperthreads it does not
own, and single samples of a millisecond-scale disk write on it are worth nothing.

**The filesystem is measured, not assumed.** The repository sits on a `/mnt/x` DrvFs mount,
which is a 9p-family passthrough to NTFS and is roughly an order of magnitude slower to fsync
than the ext4 disk backing `/tmp`. A file-database number is meaningless without saying which
one it was taken on, so every file measurement is repeated on both and labelled with the
filesystem type read out of `/proc/mounts`.

**And none of this is SA8797.** There is no automotive SoC anywhere near this repository. What
these numbers establish is the SHAPE of the cost — how it scales with rows, how much of it is
fsync, whether it is linear in commits or in bytes — which is what transfers. The absolute
milliseconds do not.
"""
from __future__ import annotations
import argparse
import gc
import inspect
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

from scene.context import Observation, SceneContext

# The store does not exist in the pre-store tree, and this script has to run there. Absent
# means "measure what can be measured and say the rest is n/a", never a crash.
try:
    from intake.store import Store
except ImportError:                                        # pragma: no cover - the old tree
    Store = None

STORE_BACKED = "store" in inspect.signature(SceneContext.__init__).parameters
TREE = "store" if STORE_BACKED else "pre-store"


# --- timing -----------------------------------------------------------------------------
def per_op(make, work, close=None, *, batch: int, trials: int) -> float:
    """Median seconds per operation, over `trials` batches of `batch` operations.

    A fresh state per trial, because most of what is measured here writes to a table and an
    insert into a table with 200,000 rows in it is not the operation anybody runs. The batch
    exists because `time.perf_counter()` around a 2 µs dict write measures the clock.

    GC is left ON. It runs in production too, and disabling it for a batch of 2000 inserts
    only moves the collection to the moment the timer stops.
    """
    samples = []
    for _ in range(trials):
        state = make()
        try:
            gc.collect()
            t0 = time.perf_counter()
            for i in range(batch):
                work(state, i)
            samples.append((time.perf_counter() - t0) / batch)
        finally:
            if close is not None:
                close(state)
    return statistics.median(samples)


def each_op(state, work, n: int) -> list[float]:
    """Every individual duration, for the percentiles end-to-end is reported in."""
    out = []
    for i in range(n):
        t0 = time.perf_counter()
        work(state, i)
        out.append(time.perf_counter() - t0)
    return out


def pct(xs: list[float], p: float) -> float:
    ordered = sorted(xs)
    if not ordered:
        return float("nan")
    k = (len(ordered) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def ms(seconds: float) -> str:
    return f"{seconds * 1000:.3f}"


def us(seconds: float) -> str:
    return f"{seconds * 1e6:.1f}"


# --- where the bytes land ------------------------------------------------------------------
def fs_type(path) -> str:
    """The filesystem under a path, from /proc/mounts. Printed beside every file figure.

    Read rather than assumed because the whole point of measuring on two of them is that they
    differ by an order of magnitude, and a table that does not say which one it used is a table
    nobody can check.
    """
    target = str(Path(path).resolve())
    best_mount, best_type = "", "unknown"
    try:
        lines = Path("/proc/mounts").read_text().splitlines()
    except OSError:                                        # pragma: no cover - not Linux
        return best_type
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        mount, kind = parts[1], parts[2]
        if (target == mount or target.startswith(mount.rstrip("/") + "/")) \
                and len(mount) >= len(best_mount):
            best_mount, best_type = mount, kind
    return best_type


def disk_dirs(repo_root: Path) -> list[tuple[str, Path]]:
    """(label, directory) for every filesystem worth measuring a file database on.

    Two: the mount the repository is on, which is what `--db ./car.db` actually uses, and the
    system temp directory, which on this box is a real Linux disk. The gap between them is
    larger than any change this task could make, which is the reason both are here.
    """
    out = []
    for label, base in (("repo", repo_root), ("tmp", Path(tempfile.gettempdir()))):
        try:
            d = Path(tempfile.mkdtemp(prefix=".measure-store-", dir=str(base)))
        except OSError:                                    # pragma: no cover
            continue
        out.append((f"{label} ({fs_type(d)})", d))
    return out


def db_bytes(conn) -> int:
    """Size of the database as SQLite itself accounts for it.

    `page_count * page_size` rather than `st_size`, so the figure is the same for `:memory:`
    and for a file, and so a WAL that has not been checkpointed yet is not read as free space.
    Cross-checked against the file's real size once, in the volume section.
    """
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    return int(page_count) * int(page_size)


class CommitProxy:
    """A connection with its commits counted, and optionally suppressed.

    Why a proxy and not a patched `SqliteVehicle`: the lever being measured is "the car stops
    committing on every `set_signal`", and the honest way to measure that is to run the SHIPPED
    statement with the commit removed, not a copy of the SQL with a line missing. A copy in
    `eval/` would be a second version of the write path that could quietly stop matching the
    one it claims to be measuring.

    `__enter__`/`__exit__` are spelled out because `with conn:` does not go through
    `__getattr__`, and `write_many` uses exactly that form — so a proxy that only wrapped
    `commit()` would count `set_signal` and `log` and silently miss every actuating write.
    A clean exit from that block IS a commit and is counted as one. Suppressing means the
    implicit transaction stays open and whatever commits next flushes it — the proposed change.
    """

    def __init__(self, conn, *, suppress: bool):
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_suppress", suppress)
        object.__setattr__(self, "commits", 0)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_conn"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_conn"), name, value)

    def commit(self):
        object.__setattr__(self, "commits", object.__getattribute__(self, "commits") + 1)
        if not object.__getattribute__(self, "_suppress"):
            object.__getattribute__(self, "_conn").commit()

    def __enter__(self):
        return object.__getattribute__(self, "_conn").__enter__()

    def __exit__(self, *exc):
        if exc[0] is None:
            object.__setattr__(self, "commits", object.__getattribute__(self, "commits") + 1)
            if object.__getattribute__(self, "_suppress"):
                return False                               # leave the transaction open
        return object.__getattribute__(self, "_conn").__exit__(*exc)


# --- compositions ----------------------------------------------------------------------
def build_session(db=":memory:", *, fake=True, llm=False):
    """A whole system, assembled the way the CLI assembles one.

    Through `Session.build` rather than by hand so the thing being timed is the composition
    that actually ships, and so this script keeps working in the pre-store tree where the
    constructor has one keyword fewer.
    """
    from cli.session import Session
    kwargs = dict(fake=fake, llm=llm, db=db)
    return Session.build(**kwargs)


def fresh_context(db=":memory:"):
    """A `SceneContext` and whatever backs it, in either tree.

    Returns (context, conn) — `conn` is None when the context is a dict, which is the one place
    this script has to branch on the tree rather than on a measurement.
    """
    if not STORE_BACKED:
        return SceneContext(), None
    from sim.vehicle import SqliteVehicle
    car = SqliteVehicle(db)
    car.init_schema()
    return SceneContext(Store(car.conn)), car.conn


def obs(i: int, *, keys: int = 2, at: float = 1000.0, ttl: float = 300.0) -> Observation:
    return Observation(key=f"inside.k{i % keys}", value=f"v{i}", confidence=0.9,
                       source="cabin_cam", at=at + i * 0.1, ttl=ttl)


def percept_source_of(key: str) -> str:
    """Which camera a key of this namespace comes from.

    Spelled here rather than imported from `cli.session.percept_source`, which does not exist
    in the pre-store tree — and a source that does not match the key's namespace is a
    `ValueError` at the envelope, not a slow path.
    """
    return "front_cam" if key.startswith("outside.") else "cabin_cam"


# --- 1. write cost per perception frame ---------------------------------------------------
def measure_frame(args, out: dict) -> None:
    """A dict write before this work, against a row now.

    Two levels, because they answer two different questions and only one of them is the number
    phase 3 is gated on:

    * `SceneContext.update` alone — the belief write, the thing that was literally a dict
      assignment. This is the cleanest before/after and the smallest.
    * `Intake.ingest(Percept)` — what a perception frame costs the SYSTEM: the raw row, the
      belief, the turn, a decision per rule, the mark-processed, one commit, and the rule
      evaluation in between. On the vehicle path this is the per-frame cost, not the one above.

    The commit is separated out on purpose. `update` is called from inside the engine and does
    not commit; the commit happens once per input at the door. Timing them apart is what says
    whether the cost is the insert or the fsync, and the answer decides whether batching or
    WAL is the lever worth pulling.
    """
    rows = []
    if not STORE_BACKED:
        median = per_op(lambda: fresh_context()[0],
                        lambda st, i: st.update(obs(i)),
                        batch=args.batch, trials=args.trials)
        rows.append(("SceneContext.update", "dict, in memory", median, args.batch, args.trials))
    else:
        def mk_mem():
            ctx, conn = fresh_context()
            return ctx, conn

        rows.append(("SceneContext.update", "row, :memory:, no commit",
                     per_op(mk_mem, lambda st, i: st[0].update(obs(i)),
                            batch=args.batch, trials=args.trials),
                     args.batch, args.trials))
        rows.append(("SceneContext.update", "row, :memory:, commit per frame",
                     per_op(mk_mem, lambda st, i: (st[0].update(obs(i)), st[1].commit()),
                            batch=args.batch, trials=args.trials),
                     args.batch, args.trials))
        for label, d in args.dirs:
            def mk_file(d=d):
                path = Path(tempfile.mkdtemp(dir=str(d))) / "car.db"
                ctx, conn = fresh_context(str(path))
                return ctx, conn, path

            def close_file(st):
                st[1].close()
                shutil.rmtree(st[2].parent, ignore_errors=True)

            rows.append(("SceneContext.update", f"row, file on {label}, no commit",
                         per_op(mk_file, lambda st, i: st[0].update(obs(i)), close_file,
                                batch=args.batch, trials=args.trials),
                         args.batch, args.trials))
            rows.append(("SceneContext.update", f"row, file on {label}, commit per frame",
                         per_op(mk_file, lambda st, i: (st[0].update(obs(i)), st[1].commit()),
                                close_file,
                                batch=args.disk_batch, trials=args.disk_trials),
                         args.disk_batch, args.disk_trials))

    # The whole frame, through the door that ships. A fresh session per trial, not one reused:
    # a matched rule goes on cooldown, every frame after it falls through to `_fallback`, and
    # `_fallback` calls `live()` — so a reused session would be measuring a perception table
    # growing under a full scan rather than the cost of a frame.
    from intake.envelope import Input, Percept
    frame_batch = max(20, args.disk_batch)

    def ingest_percept(st, i):
        st.intake.ingest(Input(source="cabin_cam", at=1000.0 + i * 0.1,
                               payload=Percept("inside.rear_occupant", "child", 0.9, 300.0)))

    def close_sess(st):
        st[0].car.close()
        if st[1] is not None:
            shutil.rmtree(st[1].parent, ignore_errors=True)

    rows.append(("Intake.ingest(Percept)", f"{TREE} tree, :memory:",
                 per_op(lambda: (build_session(":memory:"), None),
                        lambda st, i: ingest_percept(st[0], i), close_sess,
                        batch=frame_batch, trials=args.trials),
                 frame_batch, args.trials))
    if STORE_BACKED:
        for label, d in args.dirs:
            def mk_sess(d=d):
                path = Path(tempfile.mkdtemp(dir=str(d))) / "car.db"
                return build_session(str(path)), path

            rows.append(("Intake.ingest(Percept)", f"store tree, file on {label}",
                         per_op(mk_sess, lambda st, i: ingest_percept(st[0], i), close_sess,
                                batch=args.disk_batch, trials=args.disk_trials),
                         args.disk_batch, args.disk_trials))

    out["frame"] = [{"op": o, "backing": b, "median_ms": round(m * 1000, 4),
                     "batch": k, "trials": t} for o, b, m, k, t in rows]
    print(f"\n### 1. Write cost per perception frame — {TREE} tree\n")
    print("| operation | backing | median | batch × trials |")
    print("|---|---|---:|---|")
    for o, b, m, k, t in rows:
        print(f"| `{o}` | {b} | **{ms(m)} ms** | {k} × {t} |")


# --- 2. rule evaluation: N queries vs N dict reads ------------------------------------------
def measure_reads(args, out: dict) -> None:
    """The scene engine's hot path: what one event costs in perception reads.

    **`N` is counted, not assumed, and it is not what the rule set looks like it is.**
    `evaluate_explained` checks signal conditions FIRST, so a rule whose signal already settles
    the question never reads perception at all: on a parked car `animal_ahead` rejects on
    `speed_kph` and only one of the two rules issues a query. `live()` is not on the shipped
    hot path either — `_fallback` returns before touching it when no scene model is attached,
    which is arm S, the shipped arm. Both of those are the kind of thing an estimate gets
    wrong in the direction that flatters the design, so they are instrumented.

    Table size is the axis that matters. `get` rides `perception_newest` and barely moves;
    `live` walks the table for `DISTINCT key` and does not. The 36,000-row column is one hour
    at 10 Hz with no compaction — exactly the state phase 3 would reach on an hour-long drive
    if `compact_perception` did not exist.
    """
    counts = {"get": 0, "live": 0}
    real_get, real_live = SceneContext.get, SceneContext.live

    def counting_get(self, key, now):
        counts["get"] += 1
        return real_get(self, key, now)

    def counting_live(self, now):
        counts["live"] += 1
        return real_live(self, now)

    class NullSceneLLM:
        """Attached only to reach the fallback's reads. What it decides is not measured here."""

        def decide(self, snapshot, rules, speech):
            return {}

    from intake.envelope import Input, Percept, SignalWrite
    # Patched BEFORE the session is built, because `WorldView.__init__` captures
    # `perception.get` and `perception.live` as bound methods — deliberately, so the hub cannot
    # reach `update`. Patching the class afterwards counts nothing, silently, which is exactly
    # the trap that makes an instrumented count worth more than an assumed one.
    SceneContext.get, SceneContext.live = counting_get, counting_live
    per_event = {}
    try:
        def event(sess, key, value, at):
            counts["get"] = counts["live"] = 0
            sess.intake.ingest(Input(source=percept_source_of(key), at=at,
                                     payload=Percept(key, value, 0.9, 300.0)))
            return dict(counts)

        sess = build_session(":memory:")
        per_event["arm S, car parked (as seeded)"] = event(
            sess, "inside.rear_occupant", "child", 1000.0)
        # A moving car is what a vehicle actually is for most of a drive, and it is the state
        # in which BOTH rules get past their signal condition and read perception.
        sess.intake.ingest(Input(source="can0", at=1001.0,
                                 payload=SignalWrite("vehicle.all", "speed_kph", 45.0)))
        per_event["arm S, car moving at 45 kph"] = event(
            sess, "inside.nothing_a_rule_reads", "x", 1002.0)
        sess.attach_scene_llm(NullSceneLLM())
        per_event["arm S_llm, moving, nothing matched"] = event(
            sess, "inside.nothing_a_rule_reads", "y", 1003.0)
    finally:
        SceneContext.get, SceneContext.live = real_get, real_live

    print(f"\n### 2. Rule evaluation — {TREE} tree\n")
    print("Perception reads per event, counted through a real `SceneEngine.observe`:\n")
    print("| state | `get` | `live` |")
    print("|---|---:|---:|")
    for state, c in per_event.items():
        print(f"| {state} | {c['get']} | {c['live']} |")
    print()

    # The worst case of the counts above: every shipped rule reaching its Observed condition.
    n_reads = max(c["get"] for c in per_event.values())
    sizes = [0, 100, args.rows]
    rows = []
    for size in sizes:
        def mk(size=size):
            ctx, conn = fresh_context()
            for i in range(size):
                ctx.update(obs(i))
            if conn is not None:
                conn.commit()
            return ctx

        rows.append((f"get(key, now) × 1", size,
                     per_op(mk, lambda st, i: st.get("inside.k0", 1000.0),
                            batch=args.batch, trials=args.trials)))
        rows.append((f"get(key, now) × {n_reads} (one event, both rules)", size,
                     per_op(mk, lambda st, i: [st.get(f"inside.k{j % 2}", 1000.0)
                                               for j in range(n_reads)],
                            batch=max(200, args.batch // 4), trials=args.trials)))
        rows.append(("live(now)", size,
                     per_op(mk, lambda st, i: st.live(1000.0),
                            batch=200 if size >= 1000 else args.batch, trials=args.trials)))

    if STORE_BACKED:
        # The same table after the retention pass the pump already runs. This is the number
        # that decides whether an hour-long drive is survivable, and it is not a faster query.
        def mk_compacted():
            ctx, conn = fresh_context()
            for i in range(args.rows):
                ctx.update(obs(i, ttl=2.0))
            conn.commit()
            Store(conn).compact_perception(1000.0 + args.rows * 0.1)
            return ctx

        rows.append(("live(now), after compact_perception", args.rows,
                     per_op(mk_compacted, lambda st, i: st.live(1000.0 + args.rows * 0.1),
                            batch=200, trials=args.trials)))

    out["reads"] = {"per_event": per_event,
                    "timings": [{"op": o, "rows": n, "median_us": round(m * 1e6, 2)}
                                for o, n, m in rows]}
    print("| operation | perception rows | median |")
    print("|---|---:|---:|")
    for o, n, m in rows:
        print(f"| `{o}` | {n:,} | **{us(m)} µs** |")


# --- 3. end-to-end against the 50–85 ms baseline --------------------------------------------
def measure_e2e(args, out: dict) -> None:
    """The number this project leads with, with the store in the path.

    The real embedder, `llm=False` — arm C, the recommended build and the one the 50–85 ms
    figure belongs to. `pipeline.route()` alone is the baseline component: it is pure `t2f`,
    it touches no store in either tree, and it is here so the store's share can be read off
    the difference rather than inferred from a report written on another day.
    """
    from eval.dataset import load_dataset
    from intake.envelope import Input, Utterance

    rows = [r for r in load_dataset(args.dataset) if r.get("utterance")]
    rows = [r for r in rows if r.get("split") != "dev"][:args.utterances]
    texts = [r["utterance"] for r in rows]
    if not texts:
        print("\n### 3. End-to-end — no rows in the dataset, skipped\n")
        return

    print(f"\n### 3. End-to-end — {TREE} tree, {'FAKE' if args.fake else 'real'} embedder, "
          f"arm C (no LLM), n={len(texts)}\n")
    results = []
    warm = min(10, len(texts))

    sess = build_session(":memory:", fake=args.fake, llm=False)
    for t in texts[:warm]:
        sess.pipeline.route(t)                              # warm the caches, not the disk
    results.append(("pipeline.route (no store in the path)", ":memory:",
                    each_op(sess, lambda st, i: st.pipeline.route(texts[i]), len(texts))))

    def ingest(st, i):
        st.intake.ingest(Input(source="mic", at=1000.0 + i, payload=Utterance(texts[i])))

    sess2 = build_session(":memory:", fake=args.fake, llm=False)
    for i in range(warm):
        ingest(sess2, i)
    results.append((f"Intake.ingest(Utterance) — {TREE} tree", ":memory:",
                    each_op(sess2, ingest, len(texts))))

    if STORE_BACKED:
        # The third variant is the two levers applied together, and it is here rather than in
        # §5 because this is the table anyone checking "does 50–85 ms survive" will read.
        for tuning in ("as shipped", "WAL + synchronous=NORMAL, car commits deferred"):
            for label, d in args.dirs:
                path = Path(tempfile.mkdtemp(dir=str(d))) / "car.db"
                s = build_session(str(path), fake=args.fake, llm=False)
                if tuning != "as shipped":
                    s.car.conn.execute("PRAGMA journal_mode=WAL")
                    s.car.conn.execute("PRAGMA synchronous=NORMAL")
                    raw = s.car.conn
                    s.car.conn = CommitProxy(raw, suppress=True)
                    s.intake.store.conn = raw     # the door still commits, once per input
                for i in range(warm):
                    ingest(s, i)
                results.append((f"Intake.ingest(Utterance) — {tuning}", f"file on {label}",
                                each_op(s, ingest, len(texts))))
                s.car.conn = getattr(s.car.conn, "_conn", s.car.conn)
                s.car.conn.commit()
                s.car.close()
                shutil.rmtree(path.parent, ignore_errors=True)

    # The first row again, on the same warm session, after everything else has run. With the
    # real embedder the GPU is still ramping when the first variant is measured, which showed
    # up as `Intake.ingest` looking FASTER than the `route` inside it. A repeat is how that
    # gets caught instead of being reported as the store making routing cheaper.
    results.append(("pipeline.route — re-measured last, same session", ":memory:",
                    each_op(sess, lambda st, i: st.pipeline.route(texts[i]), len(texts))))

    out["e2e"] = [{"op": o, "db": db, "p50_ms": round(pct(v, 50) * 1000, 3),
                   "p95_ms": round(pct(v, 95) * 1000, 3), "n": len(v)} for o, db, v in results]
    print("| path | database | p50 | p95 | n |")
    print("|---|---|---:|---:|---:|")
    for o, db, v in results:
        print(f"| {o} | {db} | **{ms(pct(v, 50))} ms** | {ms(pct(v, 95))} ms | {len(v)} |")


# --- 4. write volume at 10 Hz ---------------------------------------------------------------
def measure_volume(args, out: dict) -> None:
    """Bytes on the flash, per frame and per hour.

    Two producers, because they write different amounts and both run at frame rate on a
    vehicle: a CAN reader sending a `SignalWrite` (a raw row plus an in-place `signal` update)
    and a camera sending a `Percept` (a raw row, a belief, a turn, a decision per rule).

    Measured with and without the retention pass the pump already runs, because they are
    different questions: without it the file is append-only and bytes/hour is growth; with it,
    each table is bounded by its own policy — and the per-table breakdown below is there
    because they are NOT all bounded, and which ones are is the whole flash-wear answer.

    Simulated clock, never a sleep. `apply_retention` measures its interval on the caller's
    clock, so frames 0.1 s apart get exactly the passes an hour of real driving would get.

    The per-table shares are taken by deleting one table at a time and VACUUMing, so each
    figure is that table's pages plus its indices as SQLite actually accounts for them —
    exact, rather than the row width somebody added up by hand.
    """
    from intake.envelope import Input, Percept, SignalWrite
    if not STORE_BACKED:
        print("\n### 4. Write volume — n/a in the pre-store tree (nothing is written down)\n")
        return

    # The fastest disk available: bytes are the answer here and page counts do not depend on
    # the filesystem, so there is no reason to spend an extra ten minutes on the 9p mount.
    label, d = sorted(args.dirs, key=lambda p: "9p" in p[0])[0]
    workloads = [
        ("SignalWrite (can0)", lambda i, now: Input(source="can0", at=now,
                                                    payload=SignalWrite("vehicle.all", "speed_kph",
                                                                        45.0 + (i % 10) * 0.1))),
        ("Percept (cabin_cam)", lambda i, now: Input(source="cabin_cam", at=now,
                                                     payload=Percept("inside.rear_occupant",
                                                                     "child", 0.9, 2.0))),
    ]
    TABLES = ("decision", "turn", "utterance", "perception", "observation_raw")
    rows = []
    for name, make_input in workloads:
        for retention in (False, True):
            path = Path(tempfile.mkdtemp(dir=str(d))) / "car.db"
            sess = build_session(str(path))
            conn = sess.car.conn
            base = db_bytes(conn)
            for i in range(args.frames):
                now = 1000.0 + i * 0.1
                if retention:
                    sess.intake.pump(now)
                sess.intake.ingest(make_input(i, now))
            conn.commit()
            grown = db_bytes(conn) - base
            on_disk = path.stat().st_size
            counted = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
            # What the file SETTLES at, rather than what it grew to. The content window is an
            # hour and the signal window is a day, so a run shorter than either never sees a
            # raw row expire and would report unbounded growth for a layer that is bounded.
            # Jumping the clock and pumping is the same code path an hour of driving takes —
            # this module owns no timer, so the caller's clock is the only clock there is, and
            # the two horizons are what separate "the words" from "a number about the machine".
            settled = {}
            if retention:
                for horizon, seconds in (("+2 h", 7200.0), ("+2 days", 2 * 86400.0)):
                    sess.intake.pump(now + seconds)
                    conn.commit()
                    settled[horizon] = {
                        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                        for t in TABLES}
                counted = settled["+2 days"]
            # Children before parents: `decision.turn_id` is a plain reference and would refuse.
            shares, last = {}, db_bytes(conn)
            conn.execute("VACUUM")
            last = db_bytes(conn)
            for t in TABLES:
                conn.execute(f"DELETE FROM {t}")
                conn.commit()
                conn.execute("VACUUM")
                now_bytes = db_bytes(conn)
                shares[t] = last - now_bytes
                last = now_bytes
            sess.car.close()
            shutil.rmtree(path.parent, ignore_errors=True)
            rows.append((name, retention, grown / args.frames, on_disk, counted, shares,
                         settled))

    hour = 36000
    out["volume"] = [{"workload": n, "retention": r, "frames": args.frames,
                      "bytes_per_frame": round(bpf, 1),
                      "bytes_per_hour_at_10hz": int(bpf * hour), "file_bytes": disk,
                      "rows": c, "table_bytes": s, "settled": st}
                     for n, r, bpf, disk, c, s, st in rows]
    print(f"\n### 4. Write volume at 10 Hz — file on {label}, {args.frames:,} frames "
          f"({args.frames / 10 / 60:.0f} simulated minutes)\n")
    print("| producer | retention | bytes/frame | extrapolated bytes/hour | file after run |")
    print("|---|---|---:|---:|---:|")
    for n, r, bpf, disk, c, s, st in rows:
        print(f"| {n} | {'on' if r else 'off'} | {bpf:,.0f} | "
              f"**{bpf * hour / 1e6:,.1f} MB** | {disk / 1e6:,.2f} MB |")
    print("\nWhere those bytes sit, and what is left after the clock is moved past both "
          "retention windows. Rows and KiB are the state each configuration ends in — the "
          "run's end with retention off, the settled state with it on:\n")
    print("| producer | retention | " + " | ".join(TABLES) + " |")
    print("|---|---|" + "---:|" * len(TABLES))
    for n, r, bpf, disk, c, s, st in rows:
        cells = " | ".join(f"{c[t]:,} rows, {s[t] / 1024:,.0f} KiB" for t in TABLES)
        print(f"| {n} | {'on' if r else 'off'} | {cells} |")
    print("\nRaw rows surviving each horizon — the content window against the signal one:\n")
    print("| producer | at the end of the run | +2 h | +2 days |")
    print("|---|---:|---:|---:|")
    for n, r, bpf, disk, c, s, st in rows:
        if not st:
            continue
        raw_end = next(x[4]["observation_raw"] for x in rows if x[0] == n and not x[1])
        print(f"| {n} | {raw_end:,} | {st['+2 h']['observation_raw']:,} | "
              f"{st['+2 days']['observation_raw']:,} |")


# --- the two levers ---------------------------------------------------------------------
def measure_levers(args, out: dict) -> None:
    """The car's own commits, and WAL. Both measured before either is recommended.

    **The car's commits.** `SqliteVehicle.set_signal`, `write_many` and `log` each commit,
    outside `Store`, so the one-commit-per-input boundary Task 4 established does not actually
    hold for any input that touches the car. Deferring them means the car's writes stay in the
    open transaction until the door commits at the end of the input — the same boundary the
    store already uses, so nothing becomes durable later than the input it belongs to.

    Measured with TWO proxies over the one connection: the car's suppresses, the store's does
    not. One proxy for both would suppress the door's commit as well, which is not the proposed
    change — it is "never commit", and it measured 0.02 ms because nothing reached the disk at
    all. That mistake is worth naming, because it is fast and it looks like a result.

    **WAL.** `journal_mode=WAL` with `synchronous=NORMAL` is the standard answer and it changes
    what a power cut costs: on FULL a returned commit is on the platter, while on WAL+NORMAL a
    commit is durable against a process crash but the last transactions can be lost to an OS
    crash or a power cut. On a car that is a decision about the record, so it is measured here
    and argued in prose — never switched on as a tuning knob.
    """
    if not STORE_BACKED:
        print("\n### 5. Levers — n/a in the pre-store tree\n")
        return
    from intake.envelope import Input, Percept, SignalWrite

    print(f"\n### 5. The two levers\n")
    print("**Commits per input, counted.** Where the fsyncs actually are:\n")
    print("| input | commits, as shipped | of those, the car's | left if the car defers |")
    print("|---|---:|---:|---:|")
    def signal_in(s, now):
        s.intake.ingest(Input(source="can0", at=now,
                              payload=SignalWrite("vehicle.all", "speed_kph", 45.0)))

    def percept_in(s, now):
        s.intake.ingest(Input(source="cabin_cam", at=now,
                              payload=Percept("inside.nobody", "x", 0.9, 300.0)))

    def ask(s, now):
        # The consent path: a scene question, then the driver's 好. It is the only input kind
        # that reaches `executor.execute` — and therefore `write_many` and `log` — without a
        # real embedder deciding what the words meant.
        s.intake.ingest(Input(source="cabin_cam", at=now,
                              payload=Percept("inside.rear_occupant", "child", 0.9, 300.0)))

    census = []
    for kind, setup, drive in (
        ("SignalWrite (can0), the 10 Hz input", None, signal_in),
        ("Percept (cabin_cam), no action", None, percept_in),
        ("Utterance answered as consent, which actuates", ask, lambda s, now: s.handle("好")),
    ):
        sess = build_session(":memory:")
        now = sess._now()
        if setup is not None:
            setup(sess, now)
        # Consume the retention interval before counting. `Session.handle` pumps, the first
        # pump of a session runs a retention pass, and its two commits belong to a
        # once-a-minute policy rather than to this input — counting them here would blame the
        # input for them. On the session's own clock, so the pending question stays answerable.
        sess.pump(now)
        car_proxy = CommitProxy(sess.car.conn, suppress=False)
        store_proxy = CommitProxy(sess.car.conn, suppress=False)
        sess.car.conn = car_proxy
        sess.intake.store.conn = store_proxy
        drive(sess, now)
        census.append((kind, car_proxy.commits + store_proxy.commits, car_proxy.commits))
        sess.car.conn = car_proxy._conn
        sess.car.close()
    for kind, total, cars in census:
        print(f"| {kind} | {total} | {cars} | {total - cars} |")
    out["commit_census"] = [{"input": k, "commits": t, "car": c} for k, t, c in census]

    pragmas = [("default (journal=delete, synchronous=FULL)", []),
               ("WAL + synchronous=NORMAL", ["PRAGMA journal_mode=WAL",
                                             "PRAGMA synchronous=NORMAL"]),
               ("WAL + synchronous=FULL", ["PRAGMA journal_mode=WAL",
                                           "PRAGMA synchronous=FULL"])]
    rows = []
    for label, d in args.dirs:
        for mode, statements in pragmas:
            for suppress in (False, True):
                path = Path(tempfile.mkdtemp(dir=str(d))) / "car.db"
                sess = build_session(str(path))
                car = sess.car
                raw = car.conn
                # Applied to the session's OWN connection, and that is not fussiness:
                # `journal_mode` is a property of the file and survives a reopen, but
                # `synchronous` is per-connection and is lost the moment a probe connection
                # closes. Setting it anywhere else would report a durability change the
                # measured run did not actually have.
                applied = mode
                for s in statements:
                    got = raw.execute(s).fetchone()
                    if s.endswith("WAL") and got and got[0] != "wal":
                        applied = f"{mode} — REFUSED, stayed {got[0]}"
                car_proxy = CommitProxy(raw, suppress=suppress)
                store_proxy = CommitProxy(raw, suppress=False)
                car.conn = car_proxy
                sess.intake.store.conn = store_proxy

                def one(st, i):
                    st.intake.ingest(Input(source="can0", at=1000.0 + i * 0.1,
                                           payload=SignalWrite("vehicle.all", "speed_kph",
                                                               45.0 + (i % 7))))

                for i in range(5):
                    one(sess, i)
                before = store_proxy.commits + (0 if suppress else car_proxy.commits)
                median = per_op(lambda: sess, one,
                                batch=args.disk_batch, trials=args.disk_trials)
                after = store_proxy.commits + (0 if suppress else car_proxy.commits)
                fsyncs = (after - before) / (args.disk_batch * args.disk_trials)
                car.conn = raw
                raw.commit()
                raw.close()
                shutil.rmtree(path.parent, ignore_errors=True)
                rows.append((label, applied,
                             "deferred to the door" if suppress else "as shipped",
                             median, fsyncs))

    out["levers"] = [{"disk": l, "journal": j, "car_commits": c,
                      "median_ms": round(m * 1000, 4), "real_commits_per_input": round(p, 2)}
                     for l, j, c, m, p in rows]
    print("\n**`Intake.ingest(SignalWrite)`, the 10 Hz input**, "
          f"median of {args.disk_batch} × {args.disk_trials}:\n")
    print("| disk | journal mode | car's commits | median per input | fsyncing commits/input |")
    print("|---|---|---|---:|---:|")
    for l, j, c, m, p in rows:
        print(f"| {l} | {j} | {c} | **{ms(m)} ms** | {p:.2f} |")


SECTIONS = {"frame": measure_frame, "reads": measure_reads, "e2e": measure_e2e,
            "volume": measure_volume, "levers": measure_levers}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="frame,reads,e2e,volume,levers",
                    help="comma-separated: " + ",".join(SECTIONS))
    ap.add_argument("--batch", type=int, default=2000, help="operations per in-memory trial")
    ap.add_argument("--trials", type=int, default=9, help="trials to take the median over")
    ap.add_argument("--disk-batch", type=int, default=100, help="operations per file-DB trial")
    ap.add_argument("--disk-trials", type=int, default=7)
    ap.add_argument("--rows", type=int, default=36000,
                    help="the large perception table: one hour at 10 Hz")
    ap.add_argument("--frames", type=int, default=6000, help="frames in the volume section")
    ap.add_argument("--utterances", type=int, default=60)
    ap.add_argument("--dataset", default="data/eval/gold.jsonl")
    ap.add_argument("--fake", action="store_true",
                    help="hashed-ngram embedder — plumbing only, never a behaviour number")
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / "data" / "catalog").exists():
        print("run from the repo root: the catalog path is relative", file=sys.stderr)
        return 2
    args.dirs = disk_dirs(repo_root)
    out = {"tree": TREE, "python": sys.version.split()[0],
           "sqlite": __import__("sqlite3").sqlite_version,
           "disks": [{"label": l, "path": str(p)} for l, p in args.dirs]}

    print(f"# What the store costs — {TREE} tree")
    print(f"\nPython {out['python']} · SQLite {out['sqlite']} · "
          f"{os.cpu_count()} logical CPUs · file databases on "
          + ", ".join(l for l, _ in args.dirs))
    print("\n**Not an SA8797.** An x86 dev box under WSL2; absolute milliseconds do not "
          "transfer, the shape of the cost does.")
    try:
        for name in [s.strip() for s in args.only.split(",") if s.strip()]:
            if name not in SECTIONS:
                print(f"unknown section {name!r}", file=sys.stderr)
                return 2
            SECTIONS[name](args, out)
    finally:
        for _, d in args.dirs:
            shutil.rmtree(d, ignore_errors=True)

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
