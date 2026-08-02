"""A store that can change shape without losing what it already holds.

Both failures worth testing here are silent ones. A change that never reaches a file already on
disk leaves a car that starts up fine and is missing a column nothing checks for. A file written
by a newer build and read anyway produces rows that are wrong rather than absent -- and this
database is becoming the only record of what the car did, so a wrong row is worse than no
database at all.
"""
import json
import sqlite3

import re

import pytest

from sim.migrate import (CURRENT_VERSION, SchemaError, SchemaTooNew, migrate,
                         read_version)
from sim.vehicle import SqliteVehicle

# sim/schema.sql exactly as it stood before `turn_id` -- every persisted `--db` file on disk
# today has this shape, and this is the only place that shape still exists.
#
# **FROZEN. Do not regenerate it from sim/schema.sql.** The difference between the two IS the
# thing under test; making them one string leaves every test below green and testing nothing.
_OLD_SCHEMA = """
CREATE TABLE IF NOT EXISTS signal (
    entity     TEXT NOT NULL,
    attribute  TEXT NOT NULL,
    value      TEXT NOT NULL,
    unit       TEXT,
    min_value  REAL,
    max_value  REAL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (entity, attribute)
);
CREATE TABLE IF NOT EXISTS operation_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    function   TEXT NOT NULL,
    parameters TEXT NOT NULL,
    outcome    TEXT NOT NULL,
    error      TEXT,
    detail     TEXT,
    at         REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS device (
    entity    TEXT PRIMARY KEY,
    available INTEGER NOT NULL DEFAULT 1,
    reason    TEXT
);
CREATE TABLE IF NOT EXISTS precondition (
    function        TEXT NOT NULL,
    requires_entity TEXT NOT NULL,
    requires_attr   TEXT NOT NULL,
    equals          TEXT NOT NULL,
    detail          TEXT NOT NULL
);
"""

_NEW_TABLES = {"observation_raw", "perception", "utterance", "turn", "decision",
               "schema_version"}


def _tables(conn) -> set:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(conn, table) -> list:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def _fingerprint(conn) -> tuple:
    """Enough of a store's shape and bookkeeping that a second migration could not hide in it.

    The DDL of every object, plus the ledger's row COUNT -- a step that re-ran and appended a
    second version row would leave the DDL identical and this different.
    """
    ddl = tuple(sorted(r[0] for r in conn.execute(
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL")))
    return ddl, conn.execute("SELECT COUNT(*), MAX(version) FROM schema_version").fetchone()


@pytest.fixture
def old_car(tmp_path):
    """A real file, with rows in it, in the shape this change has to survive.

    On disk rather than in memory because that is the case that actually exists: a `--db
    car.sqlite` a user already has, opened tomorrow by a build that knows more tables.
    """
    path = tmp_path / "car.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(_OLD_SCHEMA)
    conn.execute("INSERT INTO signal VALUES (?,?,?,?,?,?,?)",
                 ("window.driver", "window_position", json.dumps(50), "percent", 0.0, 100.0, 1000.0))
    conn.execute("INSERT INTO signal VALUES (?,?,?,?,?,?,?)",
                 ("climate.all", "ac_power", json.dumps(True), None, None, None, 1000.0))
    conn.execute("INSERT INTO operation_log (function, parameters, outcome, error, detail, at)"
                 " VALUES (?,?,?,?,?,?)",
                 ("open_window", json.dumps({"is_open": True}), "executed", None, "", 1001.0))
    conn.execute("INSERT INTO operation_log (function, parameters, outcome, error, detail, at)"
                 " VALUES (?,?,?,?,?,?)",
                 ("set_temperature", json.dumps({"temperature": 25}), "refused",
                  "precondition_failed", "空调尚未开启", 1002.0))
    conn.execute("INSERT INTO device VALUES (?,?,?)", ("window.rear", 0, "执行器无响应"))
    conn.commit()
    conn.close()
    return str(path)


# --- version 0 is a real thing, and the fixture has to be one ------------------------------

def test_the_old_schema_fixture_really_is_version_0(old_car):
    """Guards the guard. If this fixture ever drifts into the current shape, every migration
    test below still passes while migrating nothing."""
    conn = sqlite3.connect(old_car)
    assert read_version(conn) == 0
    assert "turn_id" not in _columns(conn, "operation_log")
    assert _tables(conn) & _NEW_TABLES == set()


def test_a_fresh_database_lands_at_the_current_version():
    car = SqliteVehicle(":memory:")
    car.init_schema()
    assert read_version(car.conn) == CURRENT_VERSION
    assert _NEW_TABLES <= _tables(car.conn)
    assert "turn_id" in _columns(car.conn, "operation_log")


# --- a file from before this change ---------------------------------------------------------

def test_a_version_0_database_migrates_without_losing_a_row(old_car):
    car = SqliteVehicle(old_car)
    car.init_schema()
    assert read_version(car.conn) == CURRENT_VERSION
    assert car.get_signal("window.driver", "window_position") == 50
    assert car.get_signal("climate.all", "ac_power") is True
    assert car.limits_of("window.driver", "window_position") == (0.0, 100.0)
    assert car.is_available("window.rear") == (False, "执行器无响应")
    assert [(r["function"], r["outcome"]) for r in car.recent_operations()] == [
        ("set_temperature", "refused"), ("open_window", "executed")]      # newest first


def test_rows_written_before_turns_existed_carry_no_turn(old_car):
    """NULL is the honest answer: those operations have no turn to point at. A default of 0 or
    a synthesised row would put a fact in the record that never happened."""
    car = SqliteVehicle(old_car)
    car.init_schema()
    assert [r["turn_id"] for r in car.recent_operations()] == [None, None]


def test_the_new_tables_arrive_on_an_old_file(old_car):
    car = SqliteVehicle(old_car)
    car.init_schema()
    assert _NEW_TABLES <= _tables(car.conn)


def test_a_migrated_car_can_still_write_its_own_log(old_car):
    """The sharp one. `turn_id` is `REFERENCES turn(id)`, and with foreign keys on SQLite
    resolves a parent table when the INSERT is PREPARED -- so an `operation_log` carrying that
    column while `turn` is missing rejects every write, NULL `turn_id` or not. A migration that
    ran the ALTER without the CREATE pass would leave a car that opens cleanly and fails at the
    first command the driver gives."""
    car = SqliteVehicle(old_car)
    car.init_schema()
    car.log("close_window", {"is_open": False}, "executed", None, "")
    assert car.recent_operations()[0]["function"] == "close_window"


def test_a_migrated_store_and_a_fresh_one_have_the_same_shape(old_car):
    """The failure the two-mechanism split invites: schema.sql and migrate.py both describe
    `operation_log`, and nothing forces them to describe the same one. A fresh car would get
    the column with its foreign key and a migrated car would get it without, or vice versa,
    and every test that only ever opens one kind of database would stay green."""
    migrated = SqliteVehicle(old_car)
    migrated.init_schema()
    fresh = SqliteVehicle(":memory:")
    fresh.init_schema()

    def shape(conn):
        # Comments stripped before comparing. Shape is what SQLite enforces, and the two files
        # legitimately explain themselves differently -- schema.sql annotates every column,
        # migrate.py's rebuild carries its argument in the step's docstring instead. Comparing
        # prose would make this guard fail on an edit that changed nothing and, worse, invite
        # someone to keep it green by copying comments around.
        def bare(sql: str) -> str:
            return " ".join(re.sub(r"--[^\n]*", "", sql).split())
        return {r[0]: bare(r[1]) for r in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL")}

    a, b = shape(migrated.conn), shape(fresh.conn)
    assert set(a) == set(b)
    # `operation_log`'s two definitions are written in different files, so compare what SQLite
    # ended up believing rather than the text: the ALTER appends the column, schema.sql spells
    # it inline, and only the resolved columns are comparable.
    assert {k: v for k, v in a.items() if k != "operation_log"} == \
           {k: v for k, v in b.items() if k != "operation_log"}
    assert _columns(migrated.conn, "operation_log") == _columns(fresh.conn, "operation_log")
    assert [tuple(r) for r in migrated.conn.execute("PRAGMA foreign_key_list(operation_log)")] == \
           [tuple(r) for r in fresh.conn.execute("PRAGMA foreign_key_list(operation_log)")]


def test_a_second_step_runs_in_order_and_leaves_one_ledger_row(monkeypatch, old_car):
    """There is one step today, so the loop's own behaviour -- order, and a ledger that states
    a version rather than accumulating a history -- is otherwise untested until the change that
    depends on it. A stub second step exercises it now, while it is cheap to be wrong about."""
    from sim import migrate as m

    ran = []

    def _to_2(conn):
        ran.append(2)
        conn.execute("ALTER TABLE turn ADD COLUMN note TEXT")

    monkeypatch.setattr(m, "_STEPS", [(1, m._to_1), (2, _to_2)])
    monkeypatch.setattr(m, "CURRENT_VERSION", 2)

    car = SqliteVehicle(old_car)
    car.init_schema()
    assert ran == [2]                                     # _to_1 ran too, or this would raise
    assert m.read_version(car.conn) == 2
    assert "note" in _columns(car.conn, "turn")
    assert car.conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1

    assert m.migrate(car.conn) == 2                       # and a third pass adds nothing
    assert ran == [2]
    assert car.conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1


def test_migrate_alone_refuses_a_database_the_schema_has_not_reached(old_car):
    """Which is why `init_schema` creates before it migrates, and why that order is not a
    style choice. Named here so a later reordering fails loudly instead of at runtime."""
    conn = sqlite3.connect(old_car)
    conn.execute("PRAGMA foreign_keys = ON")
    with pytest.raises(SchemaError):
        migrate(conn)


# --- doing it twice ---------------------------------------------------------------------

def test_migrating_twice_is_a_no_op(old_car):
    car = SqliteVehicle(old_car)
    car.init_schema()
    once = _fingerprint(car.conn)
    assert migrate(car.conn) == CURRENT_VERSION
    assert _fingerprint(car.conn) == once
    car.close()

    reopened = SqliteVehicle(old_car)                     # and again in a fresh process
    reopened.init_schema()
    assert _fingerprint(reopened.conn) == once
    assert [(r["function"], r["outcome"]) for r in reopened.recent_operations()] == [
        ("set_temperature", "refused"), ("open_window", "executed")]


# --- a file from after this change ------------------------------------------------------

def test_an_unknown_future_version_refuses_rather_than_guessing(tmp_path):
    """A store written by a newer build holds columns whose meaning this one does not have.
    Reading them anyway does not fail -- it produces plausible wrong answers about what the car
    did, which is the failure this whole number exists to prevent."""
    path = str(tmp_path / "car.sqlite")
    car = SqliteVehicle(path)
    car.init_schema()
    car.conn.execute("DELETE FROM schema_version")
    car.conn.execute("INSERT INTO schema_version (version) VALUES (?)", (CURRENT_VERSION + 7,))
    car.conn.commit()
    car.close()

    later = SqliteVehicle(path)
    with pytest.raises(SchemaTooNew) as excinfo:
        later.init_schema()
    assert str(CURRENT_VERSION + 7) in str(excinfo.value)


def test_a_refused_store_is_not_written_to(tmp_path):
    """Refusing has to mean the file is as it was found. `CREATE TABLE IF NOT EXISTS` would
    otherwise recreate -- empty -- any table the newer build had dropped or renamed, which is
    a write to a store we are in the middle of declaring unreadable."""
    path = str(tmp_path / "car.sqlite")
    car = SqliteVehicle(path)
    car.init_schema()
    car.conn.execute("DROP TABLE device")                 # stand-in for "a shape we don't know"
    car.conn.execute("DELETE FROM schema_version")
    car.conn.execute("INSERT INTO schema_version (version) VALUES (?)", (CURRENT_VERSION + 7,))
    car.conn.commit()
    car.close()

    later = SqliteVehicle(path)
    with pytest.raises(SchemaTooNew):
        later.init_schema()
    assert "device" not in _tables(later.conn)


def test_a_newer_row_beside_an_older_one_still_reads_as_newer(tmp_path):
    """`schema_version` declares no uniqueness, so the ledger can hold more than one row. The
    ambiguity has to resolve toward refusing: an unexpected row from a newer build must not be
    masked by the one this build wrote."""
    path = str(tmp_path / "car.sqlite")
    car = SqliteVehicle(path)
    car.init_schema()
    car.conn.execute("INSERT INTO schema_version (version) VALUES (?)", (CURRENT_VERSION + 7,))
    car.conn.commit()
    assert read_version(car.conn) == CURRENT_VERSION + 7
    car.close()

    with pytest.raises(SchemaTooNew):
        SqliteVehicle(path).init_schema()


# --- version 2: retention was impossible before it -----------------------------------------

def _v1_store(tmp_path):
    """A store at version 1, with the plain foreign keys that version shipped."""
    import sqlite3
    path = str(tmp_path / "v1.sqlite")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE observation_raw (id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL,
            source TEXT NOT NULL, payload TEXT NOT NULL, processed_at REAL, error TEXT,
            expires_at REAL);
        CREATE TABLE perception (id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_id INTEGER REFERENCES observation_raw(id), at REAL NOT NULL, key TEXT NOT NULL,
            value TEXT NOT NULL, confidence REAL NOT NULL, ttl REAL NOT NULL,
            source TEXT NOT NULL);
        CREATE TABLE utterance (id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_id INTEGER REFERENCES observation_raw(id), at REAL NOT NULL, text TEXT);
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version VALUES (1);
    """)
    conn.commit()
    return path, conn


def test_version_1_could_not_delete_a_raw_row_at_all(tmp_path):
    """The defect step 2 exists for, demonstrated rather than asserted in prose.

    A plain reference means the parsed layer PINS the raw row, so retention -- the whole point
    of splitting them -- raises instead of running."""
    import sqlite3
    path, conn = _v1_store(tmp_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("INSERT INTO observation_raw (at, source, payload) VALUES (1.0,'mic','x')")
    conn.execute("INSERT INTO utterance (raw_id, at, text) VALUES (1, 1.0, 'x')")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM observation_raw WHERE id = 1")


def test_migrating_to_2_makes_retention_possible_without_losing_the_belief(tmp_path):
    """After the step: the words go, the parsed rows stay, and their raw_id reads NULL --
    which is exactly what 'the record of what was heard is gone, what we concluded remains'
    should look like in a row."""
    from sim.vehicle import SqliteVehicle
    path, conn = _v1_store(tmp_path)
    conn.execute("INSERT INTO observation_raw (at, source, payload, expires_at) "
                 "VALUES (1.0,'mic','开车窗', 5.0)")
    conn.execute("INSERT INTO utterance (raw_id, at, text) VALUES (1, 1.0, '开车窗')")
    conn.execute("INSERT INTO perception (raw_id, at, key, value, confidence, ttl, source) "
                 "VALUES (1, 1.0, 'k', '\"child\"', 0.9, 300.0, 'cabin_cam')")
    conn.commit()
    conn.close()

    car = SqliteVehicle(path)
    car.init_schema()
    car.conn.execute("DELETE FROM observation_raw WHERE id = 1")
    car.conn.commit()

    assert car.conn.execute("SELECT COUNT(*) FROM utterance").fetchone()[0] == 1
    assert car.conn.execute("SELECT COUNT(*) FROM perception").fetchone()[0] == 1
    assert car.conn.execute("SELECT raw_id FROM utterance").fetchone()[0] is None
    assert car.conn.execute("SELECT raw_id FROM perception").fetchone()[0] is None


def test_the_rebuild_preserves_every_parsed_row(tmp_path):
    """A table rebuild copies rows by hand, which is exactly where a migration loses data."""
    from sim.vehicle import SqliteVehicle
    path, conn = _v1_store(tmp_path)
    for i in range(5):
        conn.execute("INSERT INTO perception (raw_id, at, key, value, confidence, ttl, source) "
                     "VALUES (NULL, ?, ?, '\"v\"', 0.5, 300.0, 'cabin_cam')", (float(i), f"k{i}"))
    conn.commit(); conn.close()

    car = SqliteVehicle(path)
    car.init_schema()
    rows = car.conn.execute("SELECT id, at, key, confidence FROM perception ORDER BY id").fetchall()
    assert [r["key"] for r in rows] == [f"k{i}" for i in range(5)]
    assert [r["id"] for r in rows] == [1, 2, 3, 4, 5], "ids must survive a rebuild"


def test_migrating_to_2_twice_is_a_no_op(tmp_path):
    from sim.vehicle import SqliteVehicle
    path, conn = _v1_store(tmp_path)
    conn.close()
    SqliteVehicle(path).init_schema()
    car = SqliteVehicle(path)
    car.init_schema()
    # CURRENT_VERSION, not the literal 2: a v1 store opened twice must land wherever this build
    # is, and the point of the test is that the SECOND open changes nothing, not the number.
    assert (car.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
            == CURRENT_VERSION)


# --- version 3: the half of version 2 that was missed --------------------------------------

def test_version_2_could_not_sweep_a_raw_row_that_produced_a_turn(tmp_path):
    """The defect step 3 exists for. Step 2 gave `perception` and `utterance` ON DELETE SET
    NULL and left `turn` with a plain reference -- and `ingest` opens a turn for every utterance
    and every percept, so the only raw row version 2 could delete was one that reached no
    handler at all. Retention was not partial; on real data it raised."""
    import sqlite3
    path = str(tmp_path / "v2.sqlite")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE observation_raw (id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL,
            source TEXT NOT NULL, payload TEXT NOT NULL, processed_at REAL, error TEXT,
            expires_at REAL);
        CREATE TABLE turn (id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_id INTEGER REFERENCES observation_raw(id), at REAL NOT NULL, kind TEXT NOT NULL,
            reply TEXT NOT NULL DEFAULT '');
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version VALUES (2);
    """)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("INSERT INTO observation_raw (at, source, payload) VALUES (1.0,'mic','开车窗')")
    conn.execute("INSERT INTO turn (raw_id, at, kind) VALUES (1, 1.0, 'route')")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM observation_raw WHERE id = 1")


def test_migrating_to_3_lets_the_turn_outlive_the_words(tmp_path):
    """After the step the raw row goes and the turn stays, with `raw_id` NULL -- what it decided
    survives what it heard, which is the whole point of the raw/parsed split."""
    from sim.vehicle import SqliteVehicle
    path, conn = _v1_store(tmp_path)
    conn.close()
    car = SqliteVehicle(path)
    car.init_schema()
    car.conn.execute("INSERT INTO observation_raw (at, source, payload) VALUES (1.0,'mic','开窗')")
    car.conn.execute("INSERT INTO turn (raw_id, at, kind, reply) VALUES (1, 1.0, 'route', 'ok')")
    car.conn.execute("INSERT INTO decision (turn_id, subject, verdict) VALUES (1, '开窗', 'high')")
    car.conn.commit()

    car.conn.execute("DELETE FROM observation_raw WHERE id = 1")
    car.conn.commit()

    assert car.conn.execute("SELECT raw_id, reply FROM turn").fetchone()["raw_id"] is None
    assert car.conn.execute("SELECT reply FROM turn").fetchone()["reply"] == "ok"
    assert car.conn.execute("SELECT COUNT(*) FROM decision").fetchone()[0] == 1


def test_the_turn_rebuild_keeps_what_decision_points_at(tmp_path):
    """`ALTER TABLE RENAME` rewrites other tables' REFERENCES clauses when foreign keys are on,
    so a careless rebuild leaves `decision.turn_id` and `operation_log.turn_id` pointing at a
    `turn_old` this step then drops. The symptom would be a decision that can no longer be
    written at all."""
    from sim.vehicle import SqliteVehicle
    path, conn = _v1_store(tmp_path)
    conn.close()
    car = SqliteVehicle(path)
    car.init_schema()
    car.conn.execute("INSERT INTO turn (at, kind) VALUES (1.0, 'route')")
    car.conn.execute("INSERT INTO decision (turn_id, subject, verdict) VALUES (1, 'x', 'high')")
    car.conn.execute("INSERT INTO operation_log (function, parameters, outcome, at, turn_id) "
                     "VALUES ('open_window', '{}', 'ok', 1.0, 1)")
    car.conn.commit()
    assert car.conn.execute("SELECT COUNT(*) FROM decision").fetchone()[0] == 1
    for table in ("decision", "operation_log"):
        parents = [r["table"] for r in car.conn.execute(f"PRAGMA foreign_key_list({table})")]
        assert "turn_old" not in parents and "turn" in parents
