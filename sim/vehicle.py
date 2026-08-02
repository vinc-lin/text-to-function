"""The simulated car. The DB is not a log with state attached — it IS the vehicle.

Rows are SIGNALS, (entity, attribute), not functions, so open_window and
set_window_position address the same physical window instead of holding two
contradictory beliefs about it.
"""
from __future__ import annotations
import json, sqlite3, time
from pathlib import Path
from typing import Any, Optional

from sim.migrate import migrate, refuse_if_newer

SCHEMA = Path(__file__).parent / "schema.sql"


class SqliteVehicle:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def init_schema(self) -> None:
        """Create what is absent, then change what has changed shape. Both, in that order.

        The order is required rather than tidy, and the refusal comes first because it is a
        read: a store from a newer build must not be written to by the CREATE pass on its way
        to being refused. sim/migrate.py holds the whole argument for both.
        """
        refuse_if_newer(self.conn)
        self.conn.executescript(SCHEMA.read_text(encoding="utf-8"))
        self.conn.commit()
        migrate(self.conn)

    # --- signals ---------------------------------------------------------------
    def set_signal(self, entity: str, attribute: str, value: Any,
                   unit: Optional[str] = None, limits: tuple | None = None,
                   at: Optional[float] = None) -> None:
        """`at` is the clock the stamp is made on, defaulting to this machine's.

        Optional, and defaulted, so every existing caller keeps stamping exactly as it did.
        It exists because `signal_age` compares this stamp against whatever clock the READER
        holds, and the two must be the same one. The publisher in `intake/ingest.py` re-stamps
        on the session's offset clock: without this parameter it would write `time.time()`
        while every reader asked at `time.time() + offset`, so one `/clock +5` would make every
        pumped signal read as five seconds old the instant it was published -- stale
        immediately, for a reason that has nothing to do with the bus.

        Not validated against wall time on purpose. A caller on a deliberately shifted clock is
        the point, and a "that stamp looks wrong" guard here would have to encode which lies
        about the time are legitimate.
        """
        lo, hi = (limits or (None, None))
        self.conn.execute(
            """INSERT INTO signal (entity, attribute, value, unit, min_value, max_value, updated_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(entity, attribute) DO UPDATE SET
                 value=excluded.value, updated_at=excluded.updated_at""",
            (entity, attribute, json.dumps(value), unit, lo, hi,
             time.time() if at is None else at))
        self.conn.commit()

    def get_signal(self, entity: str, attribute: str) -> Any:
        row = self.conn.execute(
            "SELECT value FROM signal WHERE entity=? AND attribute=?", (entity, attribute)).fetchone()
        return json.loads(row["value"]) if row else None

    def signal_age(self, entity: str, attribute: str, now: float) -> Optional[float]:
        """Seconds since this signal was last written; `None` if the car does not hold it.

        `updated_at` has been stamped on every write since this schema existed and read by
        nothing, so a speed frozen ten minutes ago was indistinguishable from a live one to
        every rule -- a dead bus read exactly like a stationary car. This is the first reader.

        `None`, never an exception and never a large number standing in for "unknown". "I have
        no such signal" and "I have one and it is ancient" are different facts that need
        different words: a sentinel like 1e9 makes the first look like the second, and a caller
        deciding whether to warn about a stale bus would warn about a signal this car has never
        heard of. `signal_status` in the hub is built on exactly that distinction.

        `now` must be on the same clock as the stamp -- i.e. `time.time()`, which is what
        `set_signal` uses. A caller on a different time base (a monotonic clock, a session
        offset) gets an age that is meaningless rather than merely wrong, and because a
        negative or absurd age reads as LIVE, the symptom is staleness that silently never
        fires. Not clamped to zero for that reason: an age from the wrong clock should look
        obviously wrong to whoever prints it, not be quietly rounded into plausibility.
        """
        row = self.conn.execute(
            "SELECT updated_at FROM signal WHERE entity=? AND attribute=?",
            (entity, attribute)).fetchone()
        return None if row is None else now - row["updated_at"]

    def limits_of(self, entity: str, attribute: str) -> tuple:
        row = self.conn.execute(
            "SELECT min_value, max_value FROM signal WHERE entity=? AND attribute=?",
            (entity, attribute)).fetchone()
        return (row["min_value"], row["max_value"]) if row else (None, None)

    def write_many(self, writes: list[tuple]) -> None:
        """All signals for one operation commit together, or none of them do.

        No `at` here, unlike `set_signal`, and the asymmetry is checked rather than assumed:
        the only caller is `SqliteExecutor.execute`, which holds no clock, and every signal it
        writes is an ACTUATED one -- a window position, a temperature -- which declares no
        `max_age` and is therefore never read for freshness at all (tests/sim/test_seed.py
        enforces that no function writes a sensed signal). A parameter nothing passes and
        nothing reads is one that can only ever be wrong, so it is left off until something
        needs it.
        """
        try:
            with self.conn:                       # implicit transaction
                for entity, attribute, value in writes:
                    if not entity or not attribute:
                        raise ValueError(f"bad signal address: {entity!r}.{attribute!r}")
                    self.conn.execute(
                        """INSERT INTO signal (entity, attribute, value, updated_at)
                           VALUES (?,?,?,?)
                           ON CONFLICT(entity, attribute) DO UPDATE SET
                             value=excluded.value, updated_at=excluded.updated_at""",
                        (entity, attribute, json.dumps(value), time.time()))
        except ValueError:
            raise

    # --- devices ---------------------------------------------------------------
    def set_device(self, entity: str, available: bool, reason: str | None = None) -> None:
        self.conn.execute(
            """INSERT INTO device (entity, available, reason) VALUES (?,?,?)
               ON CONFLICT(entity) DO UPDATE SET available=excluded.available, reason=excluded.reason""",
            (entity, 1 if available else 0, reason))
        self.conn.commit()

    def is_available(self, entity: str) -> tuple[bool, Optional[str]]:
        row = self.conn.execute(
            "SELECT available, reason FROM device WHERE entity=?", (entity,)).fetchone()
        if row is None:
            return True, None                     # unknown device is assumed present
        return bool(row["available"]), row["reason"]

    # --- preconditions ---------------------------------------------------------
    def add_precondition(self, function: str, entity: str, attribute: str,
                         equals: Any, detail: str) -> None:
        self.conn.execute(
            "INSERT INTO precondition VALUES (?,?,?,?,?)",
            (function, entity, attribute, json.dumps(equals), detail))
        self.conn.commit()

    def preconditions_for(self, function: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM precondition WHERE function=?", (function,)).fetchall()
        return [{"entity": r["requires_entity"], "attribute": r["requires_attr"],
                 "equals": json.loads(r["equals"]), "detail": r["detail"]} for r in rows]

    # --- log -------------------------------------------------------------------
    def log(self, function: str, parameters: dict, outcome: str,
            error: str | None, detail: str) -> None:
        self.conn.execute(
            "INSERT INTO operation_log (function, parameters, outcome, error, detail, at)"
            " VALUES (?,?,?,?,?,?)",
            (function, json.dumps(parameters, ensure_ascii=False), outcome, error, detail, time.time()))
        self.conn.commit()

    def recent_operations(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM operation_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()
