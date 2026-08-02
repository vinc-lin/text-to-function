"""Every row in and out, and the one query that must not be optimised."""
import pytest

from intake.store import CONTENT_RETENTION, SIGNAL_RETENTION, Store
from scene.context import SceneContext
from sim.vehicle import SqliteVehicle


@pytest.fixture
def car():
    car = SqliteVehicle(":memory:")
    car.init_schema()
    return car


@pytest.fixture
def store(car):
    return Store(car.conn)


# --- the raw layer -------------------------------------------------------------------------

def test_a_raw_row_round_trips(store):
    rid = store.put_raw("cabin_cam", 100.0, "后排坐着一个小孩")
    row = store.pending()[0]
    assert row["id"] == rid and row["source"] == "cabin_cam"
    assert row["payload"] == "后排坐着一个小孩" and row["processed_at"] is None


def test_pending_is_ordered_by_time_then_id(store):
    """A producer on its own clock can insert out of order, and the queue replays the world in
    the order it happened. `id` breaks ties so a replay of one store cannot reach two
    outcomes."""
    late = store.put_raw("can0", 200.0, "b")
    early = store.put_raw("can0", 100.0, "a")
    same_a = store.put_raw("mic", 150.0, "c")
    same_b = store.put_raw("mic", 150.0, "d")
    assert [r["id"] for r in store.pending()] == [early, same_a, same_b, late]


def test_a_processed_row_leaves_the_queue(store):
    rid = store.put_raw("mic", 100.0, "开车窗")
    store.mark_processed(rid, at=101.0)
    assert store.pending() == []


def test_a_failed_row_is_still_processed(store):
    """It must neither block the queue forever nor vanish. A dropped input is silence, and
    silence is the failure this system spends the most effort preventing."""
    rid = store.put_raw("mic", 100.0, "开车窗")
    store.mark_processed(rid, at=101.0, error="RuntimeError: boom")
    assert store.pending() == []
    row = store.conn.execute("SELECT * FROM observation_raw WHERE id = ?", (rid,)).fetchone()
    assert row["error"] == "RuntimeError: boom" and row["processed_at"] == 101.0


def test_pending_respects_a_limit(store):
    for i in range(5):
        store.put_raw("can0", float(i), str(i))
    assert len(store.pending(limit=2)) == 2


# --- the liveness rule ---------------------------------------------------------------------

def test_the_newest_row_wins_even_when_it_has_expired(store):
    """The single most important test in this file.

    Filtering liveness in SQL (`WHERE at + ttl >= ?`) would return the OLDER live row here,
    resurrecting a belief SceneContext.get reports as gone. Newest first; liveness is the
    caller's question, answered once, in Python, by Observation.is_live.
    """
    store.put_perception(None, at=100.0, key="k", value="old", confidence=0.9,
                         ttl=1000.0, source="cabin_cam")          # still live at now=200
    store.put_perception(None, at=150.0, key="k", value="new", confidence=0.9,
                         ttl=1.0, source="cabin_cam")             # expired by now=200
    assert store.newest_perception("k")["value"] == "new"
    assert store.live_perception_keys()[0]["value"] == "new"


def test_out_of_order_arrival_does_not_beat_a_newer_belief(store):
    """A delayed frame must not overwrite a fresher one — the rule SceneContext.update
    already holds, now enforced by the query rather than by insertion order."""
    store.put_perception(None, at=150.0, key="k", value="new", confidence=0.9, ttl=300.0,
                         source="cabin_cam")
    store.put_perception(None, at=100.0, key="k", value="stale", confidence=0.9, ttl=300.0,
                         source="cabin_cam")
    assert store.newest_perception("k")["value"] == "new"


def test_an_unknown_key_reads_as_absent(store):
    assert store.newest_perception("nope") is None


def test_live_keys_returns_one_row_per_key(store):
    for at, value in ((100.0, "a"), (150.0, "b")):
        store.put_perception(None, at=at, key="k1", value=value, confidence=0.9, ttl=300.0,
                             source="cabin_cam")
    store.put_perception(None, at=100.0, key="k2", value="c", confidence=0.9, ttl=300.0,
                         source="cabin_cam")
    rows = {r["key"]: r["value"] for r in store.live_perception_keys()}
    assert rows == {"k1": "b", "k2": "c"}


# --- typed values --------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["child", True, False, 3, 45.5, None])
def test_a_perception_value_keeps_its_type(store, value):
    """JSON-encoded, matching how SqliteVehicle stores a signal. A bare TEXT column hands
    every value back as a string, so False returns as 'false' and a rule comparing it to
    False silently stops matching."""
    store.put_perception(None, at=100.0, key="k", value=value, confidence=0.9, ttl=300.0,
                         source="cabin_cam")
    got = store.newest_perception("k")["value"]
    assert got == value and type(got) is type(value)


# --- the output layer ----------------------------------------------------------------------

def test_a_turn_is_opened_then_closed(store):
    """Two steps so a turn that raises mid-way still leaves a row. Writing only at the end
    records exactly the turns nobody needs to investigate."""
    tid = store.open_turn(None, at=100.0, kind="route")
    opened = store.conn.execute("SELECT * FROM turn WHERE id = ?", (tid,)).fetchone()
    assert opened["reply"] == "" and opened["kind"] == "route"
    store.close_turn(tid, "已为您打开车窗。")
    closed = store.conn.execute("SELECT * FROM turn WHERE id = ?", (tid,)).fetchone()
    assert closed["reply"] == "已为您打开车窗。"


def test_silence_is_recorded_as_a_fact(store):
    """'' is what the driver heard, and it is a fact rather than a missing value."""
    tid = store.open_turn(None, at=100.0, kind="scene")
    store.close_turn(tid, "")
    assert store.conn.execute("SELECT reply FROM turn WHERE id=?", (tid,)).fetchone()[0] == ""


def test_one_decision_table_serves_routing_and_rules(store):
    """Same shape either way: a subject, a verdict, what was chosen, and why."""
    tid = store.open_turn(None, at=100.0, kind="route")
    store.put_decision(tid, subject="开车窗", verdict="high", chosen="open_window")
    store.put_decision(tid, subject="animal_ahead", verdict="reject", chosen=None,
                       reason="vehicle.all/speed_kph is stale (40.0s > 2.0s)")
    rows = store.conn.execute("SELECT * FROM decision WHERE turn_id=? ORDER BY id",
                              (tid,)).fetchall()
    assert [r["verdict"] for r in rows] == ["high", "reject"]
    assert rows[1]["chosen"] is None and "stale" in rows[1]["reason"]


def test_a_decision_needs_a_turn_that_exists(store):
    """Foreign keys are ON in this connection. A decision with no turn is an orphan record of
    a reason for something that never happened."""
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        store.put_decision(9999, subject="x", verdict="high")


# --- retention -----------------------------------------------------------------------------

def test_a_swept_raw_row_leaves_its_parsed_rows_behind(store):
    """The whole point of putting expiry on the raw layer only: a run stays replayable at the
    belief level after the words are gone."""
    rid = store.put_raw("mic", 100.0, "开车窗", expires_at=200.0)
    store.put_utterance(rid, 100.0, "开车窗")
    store.put_perception(rid, 100.0, "inside.rear_occupant", "child", 0.9, 300.0, "cabin_cam")
    assert store.sweep(now=201.0) == 1
    assert store.conn.execute("SELECT COUNT(*) FROM observation_raw").fetchone()[0] == 0
    assert store.conn.execute("SELECT COUNT(*) FROM utterance").fetchone()[0] == 1
    assert store.newest_perception("inside.rear_occupant")["value"] == "child"
    # The rows survive and their link does not: the raw row they pointed at is gone, and a
    # NULL is the honest thing to say about a parent that no longer exists.
    for table in ("utterance", "perception"):
        assert store.conn.execute(f"SELECT raw_id FROM {table}").fetchone()[0] is None


def test_a_swept_raw_row_takes_the_turn_it_produced_with_it(store):
    """The bug version 3 of the schema exists for. `turn.raw_id` was a plain reference, so any
    raw row that reached a handler was PINNED — and `ingest` opens a turn for every utterance
    and every percept. The sweep did not under-delete, it raised."""
    rid = store.put_raw("mic", 100.0, "开车窗", expires_at=200.0)
    tid = store.open_turn(rid, 100.0, "route")
    store.put_decision(tid, "开车窗", "high", "open_window")
    store.close_turn(tid, "已为您打开车窗。")
    assert store.sweep(now=201.0) == 1
    row = store.conn.execute("SELECT raw_id, reply FROM turn").fetchone()
    assert row["raw_id"] is None and row["reply"] == "已为您打开车窗。"


def test_the_sweep_takes_every_copy_of_the_words(store):
    """Three columns hold verbatim speech, and retention that cleared one of them would be
    theatre: the payload, the transcript `_route` writes a second time, and the clause a route
    decision names. What survives is what was DECIDED — band, function, reply."""
    rid = store.put_raw("mic", 100.0, "开车窗", expires_at=200.0)
    store.put_utterance(rid, 100.0, "开车窗")
    tid = store.open_turn(rid, 100.0, "route")
    store.put_decision(tid, "开车窗", "high", "open_window", reason="")
    store.close_turn(tid, "已为您打开车窗。")
    store.sweep(now=201.0)
    assert store.conn.execute("SELECT text FROM utterance").fetchone()[0] is None
    row = store.conn.execute("SELECT subject, verdict, chosen FROM decision").fetchone()
    assert row["subject"] == ""
    assert (row["verdict"], row["chosen"]) == ("high", "open_window")


def test_the_sweep_takes_the_word_a_driver_consented_with(store):
    """A consent turn writes no `utterance` row, so once the payload goes its subject is the
    LAST verbatim copy of what the driver said. Missed until `/store` printed `decided 好` an
    hour after the sweep had supposedly taken the words: the privacy switch already reaches
    this column through `spoken`, and retention has to reach everything the switch does."""
    rid = store.put_raw("mic", 100.0, '{"text": "好"}', expires_at=200.0)
    tid = store.open_turn(rid, 100.0, "consent")
    store.put_decision(tid, "好", "yes", "set_window_child_lock")
    store.close_turn(tid, "已为您打开车窗儿童锁。")
    store.sweep(now=201.0)
    row = store.conn.execute("SELECT subject, verdict, chosen FROM decision").fetchone()
    assert row["subject"] == ""
    assert (row["verdict"], row["chosen"]) == ("yes", "set_window_child_lock"), \
        "that consent was given, and to what, is not speech and stays"


def test_the_sweep_leaves_a_rule_id_alone(store):
    """A scene decision's subject is a rule id. It is ours, it is not anybody's speech, and
    blanking it would delete the only thing that says which rule spoke."""
    rid = store.put_raw("cabin_cam", 100.0, '{"key": "x"}', expires_at=200.0)
    tid = store.open_turn(rid, 100.0, "scene")
    store.put_decision(tid, "child_alone", "match", "child_alone", reason="rear_occupant=child")
    store.close_turn(tid, "")
    store.sweep(now=201.0)
    row = store.conn.execute("SELECT subject, reason FROM decision").fetchone()
    assert row["subject"] == "child_alone" and row["reason"] == "rear_occupant=child"


def test_an_unprocessed_row_past_its_window_still_goes(store):
    """Deliberate. Sweeping only PROCESSED rows would keep an undrained queue on the disk
    forever, which is the case retention exists for — and an input an hour late is not one
    worth running, because acting on it commands the car about a world that has moved on."""
    store.put_raw("mic", 100.0, "开车窗", expires_at=200.0)
    assert len(store.pending()) == 1
    assert store.sweep(now=201.0) == 1
    assert store.pending() == []


# --- the retention windows -------------------------------------------------------------------

def test_voice_and_vision_get_the_content_window(store):
    for source in ("mic", "cabin_cam", "front_cam"):
        rid = store.put_raw(source, 100.0, "x")
        row = store.conn.execute("SELECT expires_at FROM observation_raw WHERE id=?",
                                 (rid,)).fetchone()
        assert row["expires_at"] == 100.0 + CONTENT_RETENTION


def test_a_signal_is_kept_longer_than_a_sentence(store):
    """Not content, and — unlike a percept or an utterance — it has no parsed layer to fall
    back on: `signal` is overwritten in place, so `observation_raw` is the only history of what
    the vehicle was doing."""
    rid = store.put_raw("can0", 100.0, "45.0")
    row = store.conn.execute("SELECT expires_at FROM observation_raw WHERE id=?",
                             (rid,)).fetchone()
    assert row["expires_at"] == 100.0 + SIGNAL_RETENTION
    assert SIGNAL_RETENTION > CONTENT_RETENTION


def test_an_undeclared_source_is_treated_as_content(store):
    """A producer wrote rows for a source nobody declared. Nobody has argued it is safe to
    keep, so it gets the shorter window — the presumption has to run that way round."""
    rid = store.put_raw("some_new_camera", 100.0, "x")
    row = store.conn.execute("SELECT expires_at FROM observation_raw WHERE id=?",
                             (rid,)).fetchone()
    assert row["expires_at"] == 100.0 + CONTENT_RETENTION


def test_the_window_is_configurable(car):
    store = Store(car.conn, retention={"mic": 5.0})
    store.put_raw("mic", 100.0, "开车窗")
    assert store.sweep(now=104.0) == 0
    assert store.sweep(now=106.0) == 1


def test_an_explicit_expiry_beats_the_policy(store):
    rid = store.put_raw("mic", 100.0, "x", expires_at=101.0)
    row = store.conn.execute("SELECT expires_at FROM observation_raw WHERE id=?",
                             (rid,)).fetchone()
    assert row["expires_at"] == 101.0


def test_a_retention_pass_runs_at_most_once_an_interval(store):
    store.put_raw("mic", 100.0, "a", expires_at=100.0)
    assert store.apply_retention(1000.0, interval=60.0) == (1, 0)
    store.put_raw("mic", 100.0, "b", expires_at=100.0)
    assert store.apply_retention(1030.0, interval=60.0) == (0, 0), "inside the interval"
    assert store.apply_retention(1061.0, interval=60.0) == (1, 0)


def test_a_clock_that_jumps_backwards_still_gets_a_pass(store):
    """`/clock -3600` is the session lying about the time, not an instruction to stop applying
    a policy. A plain `now - last < interval` would suppress every pass until it caught up."""
    store.apply_retention(1000.0, interval=60.0)
    store.put_raw("mic", 100.0, "a", expires_at=100.0)
    assert store.apply_retention(400.0, interval=60.0) == (1, 0)


# --- the privacy switch ------------------------------------------------------------------------

def test_the_switch_keeps_the_row_and_drops_the_words(car):
    """A row with an empty payload, NOT no row. The raw row is what says something arrived at
    all — its source, its instant, the turn it produced — and dropping it would make an input
    that was heard indistinguishable from one that never happened. The record stays complete
    and only the content goes."""
    store = Store(car.conn, raw_capture=False)
    rid = store.put_raw("mic", 100.0, "开车窗")
    row = store.conn.execute("SELECT * FROM observation_raw WHERE id=?", (rid,)).fetchone()
    assert row["payload"] == "" and row["source"] == "mic" and row["at"] == 100.0


def test_the_switch_reaches_every_copy_of_the_words(car):
    """Three columns, one switch. Most utterances are a single clause, so a flag that blanked
    the payload and left `decision.subject` would drop a copy of the sentence and keep the
    sentence — wired, and changing nothing."""
    store = Store(car.conn, raw_capture=False)
    rid = store.put_raw("mic", 100.0, "开车窗")
    store.put_utterance(rid, 100.0, "开车窗")
    tid = store.open_turn(rid, 100.0, "route")
    store.put_decision(tid, store.spoken("开车窗") or "", "high", "open_window")
    store.close_turn(tid, "已为您打开车窗。")
    assert store.conn.execute("SELECT text FROM utterance").fetchone()[0] is None
    assert store.conn.execute("SELECT subject FROM decision").fetchone()[0] == ""
    # What survives is everything ABOUT the words rather than the words.
    assert store.conn.execute("SELECT chosen FROM decision").fetchone()[0] == "open_window"
    assert store.conn.execute("SELECT reply FROM turn").fetchone()[0] == "已为您打开车窗。"


def test_the_switch_leaves_the_parse_alone(car):
    """`--no-raw-capture` stores the parse and never the payload. A belief IS the parse."""
    store = Store(car.conn, raw_capture=False)
    store.put_perception(None, 100.0, "inside.rear_occupant", "child", 0.9, 300.0, "cabin_cam")
    assert store.newest_perception("inside.rear_occupant")["value"] == "child"


def test_capture_on_is_the_default(store):
    rid = store.put_raw("mic", 100.0, "开车窗")
    store.put_utterance(rid, 100.0, "开车窗")
    assert store.conn.execute("SELECT payload FROM observation_raw").fetchone()[0] == "开车窗"
    assert store.conn.execute("SELECT text FROM utterance").fetchone()[0] == "开车窗"


# --- compaction ----------------------------------------------------------------------------
#
# The claim being tested is one sentence: a row that is not the newest (at, id) for its key can
# never be returned again, by any caller, at any `now`. Everything below is that claim under
# the conditions where it could fail.

def _beliefs(store, keys, nows):
    """Every answer every read can give, across several `now` values. The snapshot compaction
    must not change."""
    ctx = SceneContext(store)
    return {(k, n): (store.newest_perception(k), ctx.get(k, n),
                     {kk: v for kk, v in sorted(ctx.live(n).items())})
            for k in keys for n in nows}


def test_compaction_changes_no_read(store):
    """The important one. Compaction that alters a read is data loss wearing a performance
    costume."""
    import random
    rng = random.Random(20260802)
    keys = [f"inside.k{i}" for i in range(6)]
    for _ in range(600):
        key = rng.choice(keys)
        # Out-of-order arrivals, duplicate timestamps and a spread of ttls, because those are
        # the three things that make "the newest row" a question rather than "the last row".
        store.put_perception(None, at=float(rng.randrange(0, 400)), key=key,
                             value=rng.choice(["child", "adult", True, False, 3, None]),
                             confidence=round(rng.random(), 3),
                             ttl=float(rng.choice([1, 5, 30, 300])),
                             source=rng.choice(["cabin_cam", "front_cam"]))
    nows = [0.0, 50.0, 200.0, 399.0, 400.0, 700.0, 1e6]
    before = _beliefs(store, keys, nows)
    total = store.conn.execute("SELECT COUNT(*) FROM perception").fetchone()[0]

    # Compacted at several instants, including ones EARLIER than reads that follow: `now` is
    # not a filter on which row is returned, so no compaction time may change any read time.
    for now in (100.0, 400.0, 1e6):
        store.compact_perception(now)
    assert _beliefs(store, keys, nows) == before
    assert store.conn.execute("SELECT COUNT(*) FROM perception").fetchone()[0] < total


def test_compaction_reduces_the_row_count(store):
    for i in range(500):
        store.put_perception(None, at=float(i), key="inside.k", value=i, confidence=0.9,
                             ttl=10.0, source="cabin_cam")
    assert store.compact_perception(now=1000.0) == 499
    assert store.conn.execute("SELECT COUNT(*) FROM perception").fetchone()[0] == 1
    assert store.newest_perception("inside.k")["value"] == 499


def test_compaction_keeps_the_newest_even_when_it_has_expired(store):
    """The §5 rule, restated as a deletion. Taking the dead newest row would make the older
    live one the answer — resurrecting a belief `SceneContext.get` reports as gone."""
    store.put_perception(None, at=100.0, key="k", value="old", confidence=0.9, ttl=1000.0,
                         source="cabin_cam")
    store.put_perception(None, at=150.0, key="k", value="new", confidence=0.9, ttl=1.0,
                         source="cabin_cam")
    store.compact_perception(now=1e6)
    assert store.newest_perception("k")["value"] == "new"
    assert SceneContext(store).get("k", 1e6) is None


def test_compaction_keeps_a_superseded_row_that_is_still_live(store):
    """Not needed for safety — nothing can read it — and kept anyway: a belief from four
    seconds ago is what you read when a rule fired and you want to know what perception was
    doing around it."""
    store.put_perception(None, at=100.0, key="k", value="a", confidence=0.9, ttl=300.0,
                         source="cabin_cam")
    store.put_perception(None, at=150.0, key="k", value="b", confidence=0.9, ttl=300.0,
                         source="cabin_cam")
    assert store.compact_perception(now=200.0) == 0


def test_compaction_is_inclusive_at_the_boundary_like_is_live(store):
    """`Observation.is_live` is `now <= at + ttl`, so at exactly `at + ttl` the row is still
    live. This must not be the one place that disagrees."""
    store.put_perception(None, at=100.0, key="k", value="a", confidence=0.9, ttl=10.0,
                         source="cabin_cam")
    store.put_perception(None, at=150.0, key="k", value="b", confidence=0.9, ttl=10.0,
                         source="cabin_cam")
    assert store.compact_perception(now=110.0) == 0, "at + ttl exactly: still live"
    assert store.compact_perception(now=110.001) == 1


def test_compaction_survives_out_of_order_arrival(store):
    """`MAX(id) GROUP BY key` is the obvious predicate and it is wrong here: the late frame has
    the largest id and is NOT the newest row. That version deletes the belief every rule is
    reading and leaves a stale one in its place."""
    store.put_perception(None, at=150.0, key="k", value="new", confidence=0.9, ttl=1.0,
                         source="cabin_cam")
    store.put_perception(None, at=100.0, key="k", value="late", confidence=0.9, ttl=1.0,
                         source="cabin_cam")
    assert store.compact_perception(now=1e6) == 1
    assert store.newest_perception("k")["value"] == "new"


def test_compaction_breaks_a_tie_the_way_the_read_does(store):
    """Same `at`, so `id DESC` decides — in both places, or the survivor is not the row the
    read would have returned."""
    for value in ("first", "second"):
        store.put_perception(None, at=100.0, key="k", value=value, confidence=0.9, ttl=1.0,
                             source="cabin_cam")
    assert store.compact_perception(now=1e6) == 1
    assert store.newest_perception("k")["value"] == "second"


def test_compaction_never_loses_a_key(store):
    """`live()` walks `SELECT DISTINCT key`, so a key whose every row went would vanish from
    the world rather than read as expired."""
    for i, key in enumerate(("k1", "k2", "k3")):
        for j in range(4):
            store.put_perception(None, at=float(i * 10 + j), key=key, value=j, confidence=0.9,
                                 ttl=1.0, source="cabin_cam")
    store.compact_perception(now=1e6)
    assert {r["key"] for r in store.live_perception_keys()} == {"k1", "k2", "k3"}


def test_compacting_twice_is_a_no_op(store):
    for i in range(10):
        store.put_perception(None, at=float(i), key="k", value=i, confidence=0.9, ttl=1.0,
                             source="cabin_cam")
    assert store.compact_perception(now=1e6) == 9
    assert store.compact_perception(now=1e6) == 0


def test_a_row_with_no_expiry_is_never_swept(car):
    """A NULL `expires_at` now takes SAYING so — a source mapped to None. `expires_at=None` at
    the call site means "apply the policy", because retention that has to be asked for is
    retention nobody asks for."""
    store = Store(car.conn, retention={"can0": None})
    store.put_raw("can0", 100.0, "45.0")
    assert store.conn.execute("SELECT expires_at FROM observation_raw").fetchone()[0] is None
    assert store.sweep(now=1e9) == 0


def test_sweeping_twice_is_a_no_op(store):
    store.put_raw("mic", 100.0, "x", expires_at=200.0)
    assert store.sweep(now=201.0) == 1
    assert store.sweep(now=201.0) == 0


def test_clearing_perception_leaves_the_raw_record(store):
    """A reset forgets what perception believed. It does not rewrite what arrived."""
    rid = store.put_raw("cabin_cam", 100.0, "child")
    store.put_perception(rid, 100.0, "inside.rear_occupant", "child", 0.9, 300.0, "cabin_cam")
    store.clear_perception()
    assert store.newest_perception("inside.rear_occupant") is None
    assert store.conn.execute("SELECT COUNT(*) FROM observation_raw").fetchone()[0] == 1


# --- the scan --------------------------------------------------------------------------------

def test_live_keys_and_newest_agree_by_construction(store):
    """`live_perception_keys` is `newest_perception` in a loop, so the newest-per-key rule has
    one implementation. Two queries that had to agree would eventually not."""
    store.put_perception(None, at=100.0, key="k", value="old", confidence=0.9, ttl=1000.0,
                         source="cabin_cam")
    store.put_perception(None, at=150.0, key="k", value="new", confidence=0.9, ttl=1.0,
                         source="cabin_cam")
    assert [r["value"] for r in store.live_perception_keys()] == \
           [store.newest_perception("k")["value"]] == ["new"]


# --- bounding the audit trail ----------------------------------------------------------------

def test_a_swept_turn_takes_its_decisions_with_it(store):
    """A reason with no turn is an orphan record of something that did not happen."""
    old = store.open_turn(None, at=100.0, kind="route")
    store.put_decision(old, subject="开车窗", verdict="high", chosen="open_window")
    keep = store.open_turn(None, at=900.0, kind="route")
    store.put_decision(keep, subject="关车窗", verdict="high", chosen="open_window")
    store.commit()

    assert store.sweep_turns(before=500.0) == 1
    rows = store.conn.execute("SELECT turn_id FROM decision").fetchall()
    assert [r[0] for r in rows] == [keep]


def test_a_swept_turn_does_not_take_what_the_car_did(store):
    """The asymmetry that matters. An operation must outlive the explanation of why it
    happened: a gap in the reasoning is a cost, a gap in what the vehicle did is the one thing
    this store exists never to have."""
    tid = store.open_turn(None, at=100.0, kind="route")
    store.conn.execute(
        "INSERT INTO operation_log (function, parameters, outcome, error, detail, at, turn_id)"
        " VALUES ('open_window', '{}', 'executed', NULL, '', 100.0, ?)", (tid,))
    store.commit()

    store.sweep_turns(before=500.0)
    row = store.conn.execute("SELECT function, outcome, turn_id FROM operation_log").fetchone()
    assert row["function"] == "open_window" and row["outcome"] == "executed"
    assert row["turn_id"] is None, "the link goes; the record of the act does not"


def test_sweeping_turns_twice_is_a_no_op(store):
    store.open_turn(None, at=100.0, kind="scene")
    store.commit()
    assert store.sweep_turns(before=500.0) == 1
    assert store.sweep_turns(before=500.0) == 0


def test_a_recent_turn_is_never_swept(store):
    tid = store.open_turn(None, at=900.0, kind="scene")
    store.commit()
    assert store.sweep_turns(before=500.0) == 0
    assert store.conn.execute("SELECT COUNT(*) FROM turn WHERE id=?", (tid,)).fetchone()[0] == 1


def test_retention_bounds_the_audit_trail_too(store):
    """Wired into the pass that already runs, not left as a method somebody must remember.
    Task 8 found this was the only thing in the store nothing bounded."""
    from intake.store import TURN_RETENTION
    now = 10 * TURN_RETENTION
    store.open_turn(None, at=now - TURN_RETENTION - 1.0, kind="route")   # past the window
    keep = store.open_turn(None, at=now - 60.0, kind="route")            # inside it
    store.commit()
    store.apply_retention(now)
    assert [r[0] for r in store.conn.execute("SELECT id FROM turn")] == [keep]


# --- reading it back -------------------------------------------------------------------------

def test_recent_turns_is_newest_first(store):
    """`(at, id)` DESC, mirroring `pending`'s `(at, id)`. A producer on its own clock writes
    out of order, so the largest id is not always the newest turn."""
    late = store.open_turn(None, at=300.0, kind="route")
    early = store.open_turn(None, at=100.0, kind="scene")
    middle = store.open_turn(None, at=200.0, kind="consent")
    assert [t["id"] for t in store.recent_turns()] == [late, middle, early]


def test_recent_turns_respects_the_limit(store):
    for i in range(5):
        store.open_turn(None, at=float(i), kind="route")
    assert len(store.recent_turns(limit=2)) == 2


def test_a_turn_arrives_with_every_decision_it_produced(store):
    tid = store.open_turn(None, at=100.0, kind="scene")
    store.put_decision(tid, "animal_ahead", "reject", reason="not above 5.0")
    store.put_decision(tid, "rear_child_window_lock", "match", chosen="rear_child_window_lock",
                       suppressed_by="outranked by animal_ahead")
    store.close_turn(tid, "前方有动物，请注意。")

    turn = store.recent_turns()[0]
    assert turn["reply"] == "前方有动物，请注意。"
    assert [(d["subject"], d["verdict"]) for d in turn["decisions"]] == [
        ("animal_ahead", "reject"), ("rear_child_window_lock", "match")]
    assert turn["decisions"][1]["suppressed_by"] == "outranked by animal_ahead"


def test_the_words_come_from_the_utterance_row_when_there_is_one(store):
    """Both columns hold the sentence for a voice turn — once bare, once inside a JSON object
    — and the bare one is what a person reads."""
    raw = store.put_raw("mic", 100.0, '{"text": "开车窗"}')
    store.put_utterance(raw, 100.0, "开车窗")
    store.open_turn(raw, 100.0, "route")
    assert store.recent_turns()[0]["heard"] == "开车窗"


def test_a_scene_turn_falls_through_to_the_payload(store):
    """No utterance row: nobody said anything. The payload IS the column, printed as written."""
    raw = store.put_raw("cabin_cam", 100.0, '{"key": "inside.rear_occupant"}')
    store.open_turn(raw, 100.0, "scene")
    assert store.recent_turns()[0]["heard"] == '{"key": "inside.rear_occupant"}'


def test_a_turn_survives_the_words_that_caused_it(store):
    """The LEFT JOIN, and the reason for it. Retention deletes the raw row and the schema's
    ON DELETE SET NULL empties `raw_id`; an inner join would hide exactly the turns old enough
    to be worth looking up."""
    raw = store.put_raw("mic", 100.0, '{"text": "开车窗"}', expires_at=200.0)
    store.put_utterance(raw, 100.0, "开车窗")
    tid = store.open_turn(raw, 100.0, "route")
    store.put_decision(tid, "开车窗", "high", chosen="open_window")
    store.close_turn(tid, "已为您打开当前区域车窗。")
    store.commit()
    store.sweep(now=300.0)

    turn = store.recent_turns()[0]
    assert turn["id"] == tid and turn["heard"] is None, "the words are gone"
    assert turn["reply"] == "已为您打开当前区域车窗。"
    assert [d["verdict"] for d in turn["decisions"]] == ["high"], "the reasoning is not"


def test_words_that_were_never_written_read_the_same_as_words_that_were_deleted(store):
    """`--no-raw-capture` leaves '' where the sweep leaves NULL. Two spellings of one fact —
    not recorded — and a display must not have to tell them apart to say so."""
    quiet = Store(store.conn, raw_capture=False)
    raw = quiet.put_raw("mic", 100.0, '{"text": "开车窗"}')
    quiet.put_utterance(raw, 100.0, "开车窗")
    quiet.open_turn(raw, 100.0, "route")
    assert quiet.recent_turns()[0]["heard"] is None


def test_a_raw_row_that_raised_carries_its_error_onto_the_turn(store):
    """The turn itself just looks quiet. The error is on the row that arrived, and this is the
    one read that puts the two beside each other."""
    raw = store.put_raw("mic", 100.0, '{"text": "开车窗"}')
    store.open_turn(raw, 100.0, "route")
    store.mark_processed(raw, 101.0, error="RuntimeError: boom")
    assert store.recent_turns()[0]["error"] == "RuntimeError: boom"


def test_an_empty_store_reads_back_empty(store):
    assert store.recent_turns() == []
