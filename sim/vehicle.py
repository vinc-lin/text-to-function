"""The simulated car. The DB is not a log with state attached — it IS the vehicle.

Rows are SIGNALS, (entity, attribute), not functions, so open_window and
set_window_position address the same physical window instead of holding two
contradictory beliefs about it.
"""
from __future__ import annotations
import json, sqlite3, time
from pathlib import Path
from typing import Any, Optional

SCHEMA = Path(__file__).parent / "schema.sql"


class SqliteVehicle:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA.read_text(encoding="utf-8"))
        self.conn.commit()

    # --- signals ---------------------------------------------------------------
    def set_signal(self, entity: str, attribute: str, value: Any,
                   unit: Optional[str] = None, limits: tuple | None = None) -> None:
        lo, hi = (limits or (None, None))
        self.conn.execute(
            """INSERT INTO signal (entity, attribute, value, unit, min_value, max_value, updated_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(entity, attribute) DO UPDATE SET
                 value=excluded.value, updated_at=excluded.updated_at""",
            (entity, attribute, json.dumps(value), unit, lo, hi, time.time()))
        self.conn.commit()

    def get_signal(self, entity: str, attribute: str) -> Any:
        row = self.conn.execute(
            "SELECT value FROM signal WHERE entity=? AND attribute=?", (entity, attribute)).fetchone()
        return json.loads(row["value"]) if row else None

    def limits_of(self, entity: str, attribute: str) -> tuple:
        row = self.conn.execute(
            "SELECT min_value, max_value FROM signal WHERE entity=? AND attribute=?",
            (entity, attribute)).fetchone()
        return (row["min_value"], row["max_value"]) if row else (None, None)

    def write_many(self, writes: list[tuple]) -> None:
        """All signals for one operation commit together, or none of them do."""
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
