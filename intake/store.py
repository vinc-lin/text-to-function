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

**Two things here delete, and they answer to different questions.** `sweep` is retention: how
long we are allowed to keep what a microphone and a camera picked up in a private space.
`compact_perception` is arithmetic: rows no read can ever return, removed so `live()` stops
being O(table). They are kept separate because a change to one must not be able to quietly
become a change to the other — a retention window shortened for compliance would otherwise
also be a change to what the rules can see.
"""
from __future__ import annotations
import json
from typing import Any, Optional

from .envelope import SignalWrite
from .sources import SOURCES

# --- how long the raw layer is kept ---------------------------------------------------------
#
# Two windows, because two different things are being kept and only one of them is content.
#
# CONTENT is what a microphone or a camera picked up: the words someone said in their own car,
# and who a camera saw sitting where. On a Chinese in-vehicle product that is personal
# information under PIPL, and the parsed layer survives the sweep — the belief stays in
# `perception`, the turn and its decisions stay — so deleting the raw row costs the words and
# the frame and nothing else. One hour is about one drive: long enough to answer "why did it
# just do that" while the driver is still in the seat, short enough that yesterday's
# conversation is not on the flash.
#
# SIGNAL is a number about the machine — speed, gear, a door state. It is not content, and it
# has no parsed layer to fall back on: `signal` is overwritten in place, so `observation_raw`
# is the ONLY history of what the vehicle was doing (see sim/schema.sql). Sweeping it on the
# content window would throw away the trace that explains a decision whose reason was a signal,
# and would delete something no privacy argument asks us to delete. A day.
CONTENT_RETENTION = 3600.0
SIGNAL_RETENTION = 86400.0

# How often a pumped loop actually runs a retention pass. Both statements are unindexed scans,
# and at 10 Hz one per frame would cost more than the writes they clean up. A minute of the
# caller's clock, not a timer — see `apply_retention`.
RETENTION_INTERVAL = 60.0


def default_retention() -> dict:
    """source -> seconds to keep its raw payload, or None to keep it forever.

    Derived from the registry rather than typed out beside it, so a source added to
    `intake/sources.py` cannot be born with no retention policy at all. What it is derived FROM
    is the payload type, because that is what says whether the row is content: a `SignalWrite`
    is a number about the car and everything else is something a sensor picked up off a person.

    An unknown source — one a producer wrote rows for without declaring — falls to the
    content window in `put_raw`, not to the signal one. A source nobody declared is a source
    nobody has argued is safe to keep, and the presumption has to run that way round.
    """
    return {name: (SIGNAL_RETENTION if src.accepts is SignalWrite else CONTENT_RETENTION)
            for name, src in SOURCES.items()}


class Store:
    def __init__(self, conn, *, retention: Optional[dict] = None, raw_capture: bool = True):
        """`retention` maps a source to how long its raw payload is kept; `raw_capture=False`
        keeps none of it.

        **The switch is not the same lever as the window, and conflating them would lose the
        distinction that matters.** Retention says how long the words may be kept. The switch
        says they are never written down at all — `--no-raw-capture`, for a vehicle whose
        operator will not persist voice, where "we delete it after an hour" is not the answer
        being asked for.

        `retention=None` means the declared default, and a source mapped to `None` inside it
        means keep forever. Both spellings are needed: one is "I did not choose", the other is
        "I chose not to expire this".
        """
        self.conn = conn
        self.retention = default_retention() if retention is None else dict(retention)
        self.raw_capture = bool(raw_capture)
        # When the last retention pass ran, on the caller's clock. None until one has.
        self._retained_at: Optional[float] = None

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

    # --- what may be written down ---------------------------------------------------------
    def spoken(self, text: Optional[str]) -> Optional[str]:
        """A verbatim fragment of what a person said, as this store is allowed to keep it.

        Every column that holds the driver's own words goes through here, and there are three:
        `observation_raw.payload`, `utterance.text`, and `decision.subject` on a route turn,
        which holds the clause the router decided about. That third one is why this is a
        method rather than an `if` inside `put_raw`. Most utterances are a single clause,
        so a switch that blanked the payload and left the subject would delete a copy of the
        sentence and keep the sentence: the flag would read as wired and change nothing.

        What survives the switch is everything that is ABOUT the words rather than the words:
        the raw row itself, its source and its timestamp, the turn, the band, the function
        chosen, the reason, and the reply the driver heard. That is enough to ask why the car
        did what it did. It is not enough to reconstruct the conversation, and that is the
        point of the flag.
        """
        return text if self.raw_capture else None

    def expiry_for(self, source: str, at: float) -> Optional[float]:
        """When a raw row from this source stops being ours to keep. None = never.

        Content by default: a source with no declared window is one nobody has argued is safe
        to keep, so it gets the shorter of the two. See `default_retention`.
        """
        window = self.retention.get(source, CONTENT_RETENTION)
        return None if window is None else at + window

    # --- the raw layer ------------------------------------------------------------------
    def put_raw(self, source: str, at: float, payload: str,
                expires_at: Optional[float] = None) -> int:
        """One row for what arrived, stamped with when it stops being ours to keep.

        `expires_at=None` means "apply the policy", NOT "keep forever". Retention that has to
        be asked for is retention nobody asks for: every caller in this repo passed nothing,
        every row was immortal, and the column existed while the behaviour did not. A caller
        that genuinely wants a row kept says so by mapping its source to `None` in `retention`,
        which is a statement rather than an omission.
        """
        cur = self.conn.execute(
            "INSERT INTO observation_raw (at, source, payload, expires_at) VALUES (?,?,?,?)",
            (at, source, self.spoken(payload) or "",
             self.expiry_for(source, at) if expires_at is None else expires_at))
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

    _EXPIRED = "SELECT id FROM observation_raw WHERE expires_at IS NOT NULL AND expires_at <= ?"

    def sweep(self, now: float) -> int:
        """Delete raw rows past their expiry, and the words they left elsewhere. Returns how
        many raw rows went.

        The parsed ROWS survive — `perception` keeps the belief, `turn` keeps what was decided
        and `decision` keeps why — so a run stays replayable at the belief level after the
        words are gone. What does not survive is any verbatim copy of them, and there are two
        outside `observation_raw`:

        `utterance.text` is the same sentence the payload holds, written a second time by
        `_route`. Deleting one and keeping the other would be theatre; the schema already
        declares the column nullable for exactly this, and a row with a NULL text still says
        somebody spoke at that instant.

        `decision.subject` on a ROUTE turn is the clause the router decided about, which is a
        slice of the sentence. Scene turns are left alone: their subjects are rule ids, which
        are ours and not anybody's speech.

        Both are scrubbed BEFORE the delete, while `raw_id` still points somewhere: the
        foreign keys are ON DELETE SET NULL, so the link is gone a statement later.

        **A pending row past its window is deleted unprocessed, and that is deliberate.** The
        alternative — sweeping only processed rows — keeps an undrained queue on the disk
        forever, which is the case retention exists for. And an input an hour late is not one
        worth running: acting on it would command the car about a world that has moved on,
        which is worse than dropping it. The row's absence is the record; a queue that far
        behind has already failed at something bigger than this.
        """
        # Only route turns: `kind` is the one thing that distinguishes a subject that is speech
        # from a subject that is a rule id, and the store must not have to parse the string to
        # find out which it is holding.
        self.conn.execute(
            "UPDATE decision SET subject = '' WHERE turn_id IN "
            f"(SELECT id FROM turn WHERE kind = 'route' AND raw_id IN ({self._EXPIRED}))",
            (now,))
        self.conn.execute(
            f"UPDATE utterance SET text = NULL WHERE raw_id IN ({self._EXPIRED})", (now,))
        cur = self.conn.execute(
            "DELETE FROM observation_raw WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,))
        self.conn.commit()
        return cur.rowcount

    def apply_retention(self, now: float,
                        interval: float = RETENTION_INTERVAL) -> tuple[int, int]:
        """One retention pass — sweep the raw layer, compact perception — at most once per
        `interval`. Returns (raw rows deleted, perception rows deleted).

        **Explicit, never a timer**, the discipline `pump` and `process_pending` already
        follow: this module owns no thread and no clock, and SQLite here has thread affinity.
        The interval is measured on whatever clock the caller hands in, so a session running
        `/clock +3600` gets a pass, and a test stating `now` never has to sleep.

        Called from `Intake.pump` and `Intake.process_pending` rather than from `cli/` or
        `ui/`, because neither of those is packaged: retention driven from a dev tool is
        retention that does not exist in the deployment it was written for. Forgetting it is
        silent in the worst direction — nothing raises, the store simply keeps everything.

        A clock that jumps BACKWARDS runs a pass rather than suppressing one until it catches
        up: `/clock -3600` is the session lying about the time, not an instruction to stop
        applying a policy.
        """
        if self._retained_at is not None and 0 <= now - self._retained_at < interval:
            return (0, 0)
        self._retained_at = now
        return (self.sweep(now), self.compact_perception(now))

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
        # Through `spoken`, so the switch reaches the second copy of the sentence as well as
        # the first. The row is still written: that somebody spoke, when, and which turn it
        # produced are facts about the run, and dropping the row would lose them along with
        # the words. A NULL text is the honest shape of "we heard something and did not keep
        # what it was".
        cur = self.conn.execute(
            "INSERT INTO utterance (raw_id, at, text) VALUES (?,?,?)",
            (raw_id, at, self.spoken(text)))
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
        is retention on `perception`, not on the raw layer alone — `compact_perception`, which
        takes this from 1.019 ms to 0.019 ms at those 36,000 rows with a 2 s ttl.
        """
        keys = [r[0] for r in self.conn.execute("SELECT DISTINCT key FROM perception")]
        return [self.newest_perception(k) for k in keys]

    def compact_perception(self, now: float) -> int:
        """Delete perception rows no read can return. Returns how many went.

        **The predicate, and why it is safe.** Every read of this table returns, for a key, the
        single row that maximises `(at, id)` — `newest_perception` says so in one query and
        `live_perception_keys` is that in a loop. `now` never selects a row: it only decides
        whether the newest one is still live, and a dead newest returns None rather than
        falling back to an older live row (design §5, and the test that pins it). So:

            a row that is not the newest (at, id) for its key can never be returned again,
            for any key, by any caller, at any `now` — past, present or future.

        That is the safe set. This deletes a subset of it:

            NOT the newest for its key   AND   at + ttl < now

        The second condition buys nothing for correctness and is kept for a person: a
        superseded belief from four seconds ago is what you read when a rule fired and you want
        to know what perception was doing around it, and it costs nothing to keep until it is
        dead anyway. It also makes the window self-scaling — the compaction horizon is the
        belief's OWN declared lifetime, so a 10 Hz source with a 2 s ttl compacts down to
        twenty rows while a slow source with a 300 s ttl keeps its history.

        `< now` and not `<= now`, matching `Observation.is_live`, which is inclusive at the
        boundary: at exactly `at + ttl` the row is still live and this must not be the one
        place that disagrees.

        `ROW_NUMBER() OVER (ORDER BY at DESC, id DESC)` and NOT `MAX(id) GROUP BY key`, which is
        the obvious version and is wrong: rows arrive out of order — `SceneContext.update`
        writes a late frame without comparing timestamps precisely because the read orders by
        `at` — so the largest id for a key is not always the newest row. That version would
        delete the belief every rule is reading and leave a stale one in its place, and only
        under delayed input, which is when it matters most.

        **Why this exists at all.** Retention on the raw layer was the whole of the design's §6
        and it is not enough. `perception` is append-only, so at 10 Hz it is 36,000 rows an
        hour, and `live()` measured 1.26 ms there against 8 µs for a single-key `get`. Three
        query shapes were tried and all three are O(rows): the cost is scanning the table for
        `DISTINCT key`, so no query fixes it and only fewer rows do.
        """
        cur = self.conn.execute(
            "DELETE FROM perception WHERE id IN ("
            "  SELECT id FROM ("
            "    SELECT id, at + ttl AS dead_at,"
            "           ROW_NUMBER() OVER (PARTITION BY key ORDER BY at DESC, id DESC) AS place"
            "    FROM perception)"
            "  WHERE place > 1 AND dead_at < ?)", (now,))
        self.conn.commit()
        return cur.rowcount

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
