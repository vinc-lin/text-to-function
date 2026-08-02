"""Every row in and out. The only module that reads or writes the I/O tables.

`Store(conn)` takes the connection `SqliteVehicle` already owns — the same shape
`SqliteExecutor(car, cards)` uses — because one database with two connections would be two
opinions about a transaction. It owns `observation_raw`, `perception`, `utterance`, `turn` and
`decision`; `signal`, `device` and `precondition` stay with the car.

`operation_log` is the one shared table, and only one column of it: `turn_id`. The car owns
what it did; a turn owns why, and there is nowhere else that fact could live. See
`operations_watermark`.

**Liveness is not decided here.** `newest_perception` returns the newest row for a key whether
or not it has expired, and the caller applies `Observation.is_live`. Filtering in SQL would
return an older *live* row when the newest has expired — resurrecting a belief that
`SceneContext.get` reports as gone — and it would put a second definition of "live" in a second
language, where it can drift from the first at the boundary. See the design's §5, and
`test_the_newest_row_wins_even_when_it_has_expired`, which fails if anyone moves the filter
into the query.
"""
from __future__ import annotations
import json
from typing import Any, Optional


class Store:
    def __init__(self, conn):
        self.conn = conn

    def commit(self) -> None:
        """Make everything written since the last commit durable.

        The writers here deliberately do NOT commit. SQLite fsyncs on commit, and a single
        voice input touches six of them — raw, utterance, open_turn, decision, close_turn,
        mark_processed — which measured 17.4 ms on a file database against 2.86 ms for one
        commit. Cost was linear in the number of store CALLS, not in bytes written.

        Nothing is lost by deferring: within one connection an uncommitted write is already
        visible to every read, and this store has exactly one connection by construction. What
        changes is only when the data reaches the disk, and the right boundary for that is one
        input — an input is either recorded or it is not, and half a turn in the record is a
        worse artifact than none.
        """
        self.conn.commit()

    # --- the raw layer ------------------------------------------------------------------
    def put_raw(self, source: str, at: float, payload: str,
                expires_at: Optional[float] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO observation_raw (at, source, payload, expires_at) VALUES (?,?,?,?)",
            (at, source, payload, expires_at))
        return cur.lastrowid

    def pending(self, limit: Optional[int] = None) -> list:
        """Unprocessed rows, oldest first.

        Ordered by `(at, id)` rather than by `id` alone: a producer writing on its own clock
        can insert out of order, and the queue is meant to replay the world in the order it
        happened rather than the order it arrived. `id` breaks ties so the order is total —
        two frames stamped the same instant must still process deterministically, or a replay
        of the same store could reach a different outcome.
        """
        sql = ("SELECT * FROM observation_raw WHERE processed_at IS NULL "
               "ORDER BY at, id")
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return self.conn.execute(sql).fetchall()

    def mark_processed(self, raw_id: int, at: float, error: Optional[str] = None) -> None:
        """Done, whether or not it worked.

        An input that raised is marked processed WITH its error. It must not block the queue
        forever, and it must not vanish: a dropped input is silence, and this system treats
        unexplained silence as the failure worth the most effort to prevent.
        """
        self.conn.execute(
            "UPDATE observation_raw SET processed_at = ?, error = ? WHERE id = ?",
            (at, error, raw_id))

    def sweep(self, now: float) -> int:
        """Delete raw rows past their expiry. Returns how many went.

        The raw layer only. `perception` and `utterance` survive, so a run stays replayable at
        the belief level after the words are gone — which is what makes the privacy switch a
        column rather than a redesign.
        """
        cur = self.conn.execute(
            "DELETE FROM observation_raw WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,))
        self.conn.commit()
        return cur.rowcount

    # --- the parsed layer ---------------------------------------------------------------
    def put_perception(self, raw_id: Optional[int], at: float, key: str, value: Any,
                       confidence: float, ttl: float, source: str) -> int:
        # JSON-encoded, matching how SqliteVehicle stores a signal value. A perception value
        # has the same problem a signal value does: it can be a string, a boolean or a number,
        # and a bare TEXT column hands every one of them back as a string — so `False` returns
        # as `'false'` and a rule comparing it to `False` silently stops matching.
        cur = self.conn.execute(
            "INSERT INTO perception (raw_id, at, key, value, confidence, ttl, source) "
            "VALUES (?,?,?,?,?,?,?)",
            (raw_id, at, key, json.dumps(value), confidence, ttl, source))
        return cur.lastrowid

    def put_utterance(self, raw_id: Optional[int], at: float, text: Optional[str]) -> int:
        cur = self.conn.execute(
            "INSERT INTO utterance (raw_id, at, text) VALUES (?,?,?)", (raw_id, at, text))
        return cur.lastrowid

    # --- the output layer ---------------------------------------------------------------
    def open_turn(self, raw_id: Optional[int], at: float, kind: str) -> int:
        """A turn is opened before the work and closed after it.

        Two steps rather than one write at the end, so a turn that raises mid-way still leaves
        a row. The alternative records only the turns that finished, which is precisely the
        set you do not need to investigate.
        """
        cur = self.conn.execute(
            "INSERT INTO turn (raw_id, at, kind) VALUES (?,?,?)", (raw_id, at, kind))
        return cur.lastrowid

    def operations_watermark(self) -> int:
        """The last `operation_log` id before a turn starts. 0 when the car has done nothing.

        Half of how `operation_log.turn_id` gets filled: take this when a turn opens, hand it
        back to `close_turn`, and every operation logged in between is attributed to that turn.
        `operation_log` belongs to the car — this module writes exactly one column of it, the
        one that says *why*, which is a fact about a turn and about nothing else.

        **The alternative was telling the executor which turn it is in, and it is worse.** That
        needs every caller of `execute` to set and unset an ambient turn id correctly, and a
        caller that forgets does not leave a NULL — it leaves the PREVIOUS turn's id on rows
        that belong to this one. A wrong answer in the table you consult to find out what
        happened is worse than no answer. A watermark needs no cooperation from anything that
        actuates: it catches the router, the scene engine's consent, and whatever is wired up
        next, because all three go through `executor.execute` and all three log.
        """
        row = self.conn.execute("SELECT MAX(id) AS last FROM operation_log").fetchone()
        return row["last"] or 0

    def close_turn(self, turn_id: int, reply: str,
                   since_operation: Optional[int] = None) -> None:
        """What the driver heard, and — if a watermark was taken — what the car did about it.

        `since_operation=None` means "attribute nothing", so a caller that does not care about
        operations keeps its two-argument call and gets NULLs. Bounded by the watermark rather
        than by `turn_id IS NULL` alone: a `--db` file written before turns existed is full of
        untagged rows, and the first turn of the next session would otherwise claim all of them.
        """
        self.conn.execute("UPDATE turn SET reply = ? WHERE id = ?", (reply or "", turn_id))
        if since_operation is not None:
            self.conn.execute(
                "UPDATE operation_log SET turn_id = ? WHERE id > ? AND turn_id IS NULL",
                (turn_id, since_operation))

    def put_decision(self, turn_id: int, subject: str, verdict: str,
                     chosen: Optional[str] = None, reason: str = "",
                     suppressed_by: str = "") -> int:
        """One table for routing bands and rule verdicts alike.

        They are the same shape — a subject, a verdict, what was chosen, and why — so two
        tables would be two things to keep in step. A JSON blob would be neither queryable nor
        checkable, and being able to ask the database *why* is the point of writing it down.
        """
        cur = self.conn.execute(
            "INSERT INTO decision (turn_id, subject, verdict, chosen, reason, suppressed_by) "
            "VALUES (?,?,?,?,?,?)",
            (turn_id, subject, verdict, chosen, reason or "", suppressed_by or ""))
        return cur.lastrowid

    # --- reads for Scene Context --------------------------------------------------------
    def newest_perception(self, key: str) -> Optional[dict]:
        """The newest row for this key, expired or not. Liveness is the caller's question."""
        row = self.conn.execute(
            "SELECT * FROM perception WHERE key = ? ORDER BY at DESC, id DESC LIMIT 1",
            (key,)).fetchone()
        return self._decode(row)

    def live_perception_keys(self) -> list:
        """The newest row per key, expired or not — same contract as `newest_perception`.

        One indexed seek per distinct key, rather than a join over the whole table. Measured
        at 36,000 rows (one hour at 10 Hz, two keys): the join costs 2.16 ms, a bounded join
        1.47 ms, this 1.26 ms — and this one is `newest_perception` in a loop, so the
        newest-per-key rule has exactly one implementation instead of two that must agree.

        **All three are O(rows), and that is the real point.** A cleverer query does not fix
        it, because the cost is the `DISTINCT key` scan over an append-only table. The answer
        is retention on `perception`, not on the raw layer alone — see the note in
        `sim/schema.sql` and Task 5.
        """
        keys = [r[0] for r in self.conn.execute("SELECT DISTINCT key FROM perception")]
        return [self.newest_perception(k) for k in keys]

    def clear_perception(self) -> None:
        """Forget every belief. The reset path — the car underneath was replaced."""
        self.conn.execute("DELETE FROM perception")
        self.conn.commit()

    @staticmethod
    def _decode(row) -> Optional[dict]:
        if row is None:
            return None
        d = dict(row)
        d["value"] = json.loads(d["value"])
        return d
