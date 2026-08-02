"""Every row in and out, and the one query that must not be optimised."""
import pytest

from intake.store import Store
from sim.vehicle import SqliteVehicle


@pytest.fixture
def store():
    car = SqliteVehicle(":memory:")
    car.init_schema()
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


def test_a_row_with_no_expiry_is_never_swept(store):
    store.put_raw("can0", 100.0, "45.0")
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
