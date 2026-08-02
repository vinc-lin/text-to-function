"""Changing the shape of a store that holds the only account of what happened.

`init_schema()` was `CREATE TABLE IF NOT EXISTS` and nothing else, which is enough for exactly
as long as tables only ever get ADDED. `operation_log` gaining `turn_id` is the first change to
a table that already has rows in every persisted `--db` file, and `IF NOT EXISTS` is *silent*
about it: it sees the table, does nothing, reports success, and the column never appears. The
car comes up looking fine and is missing the one link between what it did and why. So the two
mechanisms are split by the kind of change they can make, and neither may claim the other's:

    sim/schema.sql   creates tables that are absent; never alters one that exists
    sim/migrate.py   alters tables that exist; never creates one, except its own ledger

**The order in `init_schema` is load-bearing, not tidiness.** The CREATE pass runs first
because step 1's new column is `REFERENCES turn(id)`, and with `PRAGMA foreign_keys = ON`
SQLite resolves a foreign key's parent table when a statement is PREPARED. An `operation_log`
carrying that column while `turn` is missing cannot be inserted into at all -- `no such table:
main.turn`, NULL `turn_id` or not. The failure would land not here but at the first executed
command, on a car that started up cleanly. `_to_1` therefore refuses rather than build that.

**Version 0 is a database with no `schema_version` row.** That is every car file on disk today
-- and also every fresh one, because the CREATE pass makes the table and leaves it empty. One
number, two very different databases, which is why every step has to be a no-op when the CREATE
pass has already produced the shape it was going to build.

Idempotence is not a nicety here, it is the crash story: the version is written after each step
rather than once at the end, so an interrupted migration resumes at the next step, and a crash
between a step and its version write just runs that step again to find nothing left to do.
"""
from __future__ import annotations

import sqlite3
from typing import Callable

CURRENT_VERSION = 4


class SchemaError(RuntimeError):
    """This store cannot be brought to the shape this build expects."""


class SchemaTooNew(SchemaError):
    """The store was written by a newer build than this one.

    Refusing is the entire value of the version number. An older binary that pressed on would
    read columns whose meaning it does not have and write rows in a shape the newer one does
    not expect -- turning the only record of what the car did into a plausible fiction, with
    nothing downstream able to tell. A store you cannot read is recoverable; a store you have
    misread is not.
    """


# --- reading the version -------------------------------------------------------------------

def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    # PRAGMA takes no parameters; every table name reaching here is a literal in this module.
    return any(r[1] == column for r in conn.execute(f"PRAGMA table_info({table})"))


def read_version(conn: sqlite3.Connection) -> int:
    """The version this store is at. 0 means "from before any of this existed".

    Two shapes both read as 0 and both are ordinary: no `schema_version` table at all (a car
    file written before this module), and the table present but empty (a fresh database, whose
    CREATE pass made it seconds ago). Treating the empty table as an error would make every
    first open a failure.

    `MAX(version)` rather than "the row", because that is the direction this ambiguity has to
    resolve. The table declares no uniqueness, so if a newer build has left a row this one did
    not expect, the store must read as NEWER and be refused -- never as older and be written to.

    Column access is by index, not by name: this runs on whatever connection it is handed, and
    only `SqliteVehicle` sets `row_factory`.
    """
    if not _has_table(conn, "schema_version"):
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return 0 if row is None or row[0] is None else int(row[0])


def refuse_if_newer(conn: sqlite3.Connection) -> int:
    """Read the version, refusing anything this build does not know how to read. Returns it.

    Called before the CREATE pass as well as from `migrate`, because refusing has to mean *I
    touched nothing*. `CREATE TABLE IF NOT EXISTS` against a store from a newer build would
    quietly recreate, empty, any table that build had dropped or renamed -- writing to a file
    in the middle of declaring it unreadable. The check is a read and costs one query, so it
    can happen at the only point where it is still true that nothing has been written.
    """
    version = read_version(conn)
    if version > CURRENT_VERSION:
        raise SchemaTooNew(
            f"store is at schema version {version}, this build knows {CURRENT_VERSION}: "
            "it was written by a newer version of this software, and reading it here would "
            "misinterpret rows rather than fail")
    return version


# --- the steps ------------------------------------------------------------------------------

def _to_1(conn: sqlite3.Connection) -> None:
    """`operation_log` gains `turn_id`: what the car did, tied to why it did it.

    Guarded on both sides, for two different reasons.

    The column may already be there -- on a fresh database the CREATE pass wrote the current
    definition moments ago, and this step still runs, because an empty `schema_version` reads
    as 0 whether the file is old or brand new. Without the check every fresh car would fail to
    open with `duplicate column name`.

    And `turn` must be there, because the column references it: adding it to a database the
    CREATE pass has not reached produces a car that starts fine and cannot write its own log
    (module docstring). Raising here names the ordering; the alternative failure names a
    missing table, three layers away, on the first command the driver gives.
    """
    if not _has_table(conn, "turn"):
        raise SchemaError(
            "migrate() runs after the schema is created: operation_log.turn_id references "
            "turn(id), and adding it while turn is missing leaves a table nothing can insert "
            "into. Call SqliteVehicle.init_schema(), which does both in order.")
    if _has_column(conn, "operation_log", "turn_id"):
        return
    conn.execute("ALTER TABLE operation_log ADD COLUMN turn_id INTEGER REFERENCES turn(id)")


# (target version, how to get there). Append only, and never renumber: the numbers are written
# into files we do not have.
def _to_2(conn: sqlite3.Connection) -> None:
    """`perception.raw_id` and `utterance.raw_id` gain ON DELETE SET NULL.

    Version 1 declared them as plain references, which quietly forbids the thing the raw/parsed
    split exists for: retention deletes a raw row, the foreign key refuses, and the sweep
    raises. CASCADE is the other obvious answer and is worse -- it takes the belief with the
    words, so a swept store forgets what it concluded as well as what it heard.

    SQLite cannot alter a constraint, so the tables are rebuilt and copied. Foreign keys are
    disabled for the swap because the copy briefly has two tables referencing one parent, and
    re-enabled after: `PRAGMA foreign_keys` is a no-op inside a transaction, which is why this
    commits first rather than wrapping the whole thing.
    """
    if _fk_is_set_null(conn, "perception") and _fk_is_set_null(conn, "utterance"):
        return                       # the CREATE pass already built the current definition
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        for table, columns in (
            ("perception", "id, raw_id, at, key, value, confidence, ttl, source"),
            ("utterance",  "id, raw_id, at, text"),
        ):
            if _fk_is_set_null(conn, table):
                continue
            conn.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
            conn.executescript(_REBUILT[table])
            conn.execute(f"INSERT INTO {table} ({columns}) SELECT {columns} FROM {table}_old")
            conn.execute(f"DROP TABLE {table}_old")
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def _to_3(conn: sqlite3.Connection) -> None:
    """`turn.raw_id` gains ON DELETE SET NULL -- the half of step 2 that was missed.

    Step 2 fixed `perception` and `utterance` and stopped there, and the retention tests of the
    day only ever swept a raw row that had reached no handler. Every real one has: `ingest`
    opens a turn for an utterance and for a percept alike, so `turn` pinned the raw row and
    `sweep` raised `FOREIGN KEY constraint failed` on the first voice input it was ever pointed
    at. Retention was not slow or partial, it was impossible.

    SET NULL rather than CASCADE for the reason step 2 gives: CASCADE would take the turn -- and
    through `decision.turn_id`, every reason it recorded -- along with the words. The turn is
    the belief level the raw/parsed split exists to keep.

    Written as its own step rather than folded into `_to_2`, even though it is the same repair.
    A step is a historical record: `_to_2` has already run against files this build will be
    handed, and editing it would mean a store that reports version 2 and does not have what
    version 2 now claims. Append only, and never renumber.
    """
    if not _has_table(conn, "turn"):
        # Unreachable through `init_schema`, which creates before it migrates, and stated
        # anyway: skipping the step and writing version 3 would leave a file claiming a shape
        # it does not have, which is the one failure the version number exists to prevent.
        raise SchemaError(
            "migrate() runs after the schema is created: turn is missing, so this step has "
            "nothing to alter. Call SqliteVehicle.init_schema(), which does both in order.")
    if _fk_is_set_null(conn, "turn"):
        return                       # the CREATE pass already built the current definition
    conn.commit()
    # Foreign keys OFF for the swap, and this is doing two jobs. It stops the copy tripping the
    # constraint, and -- the part specific to THIS table -- it stops `ALTER TABLE RENAME` from
    # helpfully rewriting `decision.turn_id` and `operation_log.turn_id` to point at
    # `turn_old`, which SQLite does when foreign keys are enabled. Those two would then
    # reference a table this step is about to drop.
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        columns = "id, raw_id, at, kind, reply"
        conn.execute("ALTER TABLE turn RENAME TO turn_old")
        conn.executescript(_REBUILT["turn"])
        conn.execute(f"INSERT INTO turn ({columns}) SELECT {columns} FROM turn_old")
        conn.execute("DROP TABLE turn_old")
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def _fk_is_set_null(conn: sqlite3.Connection, table: str) -> bool:
    """Whether this table's raw_id already nulls itself when the raw row goes."""
    try:
        rows = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
    except sqlite3.OperationalError:
        return False                 # table absent; the CREATE pass has not run yet
    return any(r["from"] == "raw_id" and (r["on_delete"] or "").upper() == "SET NULL"
               for r in rows)


_REBUILT = {
    "perception": """
        CREATE TABLE perception (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_id     INTEGER REFERENCES observation_raw(id) ON DELETE SET NULL,
            at         REAL NOT NULL,
            key        TEXT NOT NULL,
            value      TEXT NOT NULL,
            confidence REAL NOT NULL,
            ttl        REAL NOT NULL,
            source     TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS perception_newest ON perception(key, at DESC);
    """,
    "utterance": """
        CREATE TABLE utterance (
            id     INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_id INTEGER REFERENCES observation_raw(id) ON DELETE SET NULL,
            at     REAL NOT NULL,
            text   TEXT
        );
    """,
    "turn": """
        CREATE TABLE turn (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_id  INTEGER REFERENCES observation_raw(id) ON DELETE SET NULL,
            at      REAL NOT NULL,
            kind    TEXT NOT NULL,
            reply   TEXT NOT NULL DEFAULT ''
        );
    """,
}


def _to_4(conn: sqlite3.Connection) -> None:
    """`turn` becomes deletable: decisions cascade with it, operations keep their row.

    Task 8 measured the reason. Nothing ever deleted a turn or a decision, so the audit trail
    grew at 188 B per input forever -- 6.8 MB an hour at 10 Hz, 6.8 GB per thousand hours --
    and most of those rows record a frame in which nothing happened. That was the one number
    in the whole measurement that argued against the vehicle path, and it is a retention gap
    rather than a latency one.

    The two directions differ on purpose. A `decision` CASCADEs: it is a reason for a turn, and
    a reason with no turn is an orphan record of something that did not happen. An
    `operation_log` row SET NULLs: the car really did that, and an operation must outlive the
    explanation of why it happened. Losing the link leaves a gap in the reasoning; losing the
    row would leave a gap in what the vehicle did, which is the one thing this store exists to
    never do.
    """
    if _fk_action(conn, "decision", "turn_id") == "CASCADE" and \
            _fk_action(conn, "operation_log", "turn_id") == "SET NULL":
        return
    conn.commit()
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        _rebuild(conn, "decision",
                 "id, turn_id, subject, verdict, chosen, reason, suppressed_by",
                 """CREATE TABLE decision (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        turn_id       INTEGER NOT NULL REFERENCES turn(id) ON DELETE CASCADE,
                        subject       TEXT NOT NULL,
                        verdict       TEXT NOT NULL,
                        chosen        TEXT,
                        reason        TEXT NOT NULL DEFAULT '',
                        suppressed_by TEXT NOT NULL DEFAULT ''
                    );""")
        _rebuild(conn, "operation_log",
                 "id, function, parameters, outcome, error, detail, at, turn_id",
                 """CREATE TABLE operation_log (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        function   TEXT NOT NULL,
                        parameters TEXT NOT NULL,
                        outcome    TEXT NOT NULL,
                        error      TEXT,
                        detail     TEXT,
                        at         REAL NOT NULL,
                        turn_id    INTEGER REFERENCES turn(id) ON DELETE SET NULL
                    );""")
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def _fk_action(conn: sqlite3.Connection, table: str, column: str) -> str:
    try:
        rows = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
    except sqlite3.OperationalError:
        return ""
    return next((r["on_delete"] or "" for r in rows if r["from"] == column), "")


def _rebuild(conn: sqlite3.Connection, table: str, columns: str, create: str) -> None:
    """Rename, recreate, copy, drop. SQLite cannot alter a constraint any other way."""
    conn.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
    conn.executescript(create)
    conn.execute(f"INSERT INTO {table} ({columns}) SELECT {columns} FROM {table}_old")
    conn.execute(f"DROP TABLE {table}_old")


_STEPS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, _to_1),
    (2, _to_2),
    (3, _to_3),
    (4, _to_4),
]


def migrate(conn: sqlite3.Connection) -> int:
    """Bring `conn` up to `CURRENT_VERSION`, applying each step in order. Returns the version.

    The ledger is created here if absent, which is the one table this module owns: it cannot
    depend on schema.sql having run to discover whether schema.sql has run.
    """
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    version = refuse_if_newer(conn)
    for target, step in _STEPS:
        if target <= version:
            continue
        step(conn)
        # One row, replaced, so the ledger states a version rather than accumulating a history
        # nothing reads. Committed here rather than after the loop -- see the module docstring
        # on why a half-migrated store has to be a resumable state and not a corrupt one.
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (target,))
        conn.commit()
        version = target
    return version
