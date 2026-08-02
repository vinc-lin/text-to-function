-- The simulated car, and the record of everything that reached it.
--
-- Every statement here is IF NOT EXISTS, because this script runs on every open -- including
-- on a `--db` file written months ago. It CREATES what is absent and never touches what is
-- present, which is exactly why a table that has to CHANGE shape cannot be changed here: on
-- an existing table these statements are silent no-ops. That is sim/migrate.py's job, and the
-- two never both own one change.

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
    at         REAL NOT NULL,
    -- What the car did, tied to why it did it. NULL on every row written before turns
    -- existed, and that NULL is the honest answer -- those operations have no turn to point
    -- at, and synthesising one would put a fact in the record that never happened.
    --
    -- `turn` is declared further down this file and that is fine: SQLite resolves a foreign
    -- key's parent when a statement is PREPARED, not when the table is created, and by then
    -- the whole script has run. It is NOT fine outside this file -- see sim/migrate.py.
    turn_id    INTEGER REFERENCES turn(id)
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

-- --- the store: what arrived, and what was decided about it ------------------------------
--
-- Inputs are an interface, outputs are a record. A producer -- the vision process, a CAN
-- reader, ASR -- writes a row and is done, from another process in another language, with no
-- binding into our runtime. The reply still comes back synchronously from `route()`; it is
-- ALSO written down, so a run can be asked afterwards why it did what it did.
--
-- `intake/store.py` is the only module that reads or writes any of this. `intake/ingest.py`
-- fills it -- a raw row per input, a turn and its decisions per dispatch -- and drains
-- `observation_raw` through `process_pending`. `scene/context.py` reads `perception`.

CREATE TABLE IF NOT EXISTS observation_raw (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    at           REAL NOT NULL,
    source       TEXT NOT NULL,     -- declared in intake/sources.py
    payload      TEXT NOT NULL,     -- the caption, the transcript, the frame
    processed_at REAL,              -- NULL = pending
    error        TEXT,              -- set when processing raised; the row is still marked done
    expires_at   REAL               -- NULL = keep. Retention reads only this.
);
-- `processed_at` is a column rather than a separate queue table, so "pending" is a queryable
-- state that cannot disagree with the row it describes.
CREATE INDEX IF NOT EXISTS raw_pending ON observation_raw(processed_at, at, id);

CREATE TABLE IF NOT EXISTS perception (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    -- ON DELETE SET NULL, not a plain reference: retention deletes the raw row and the belief
    -- has to survive it. A plain FK makes the sweep raise, and CASCADE would take the parsed
    -- row with it -- both of which forbid the thing the raw/parsed split exists to allow.
    raw_id     INTEGER REFERENCES observation_raw(id) ON DELETE SET NULL,
    at         REAL NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    confidence REAL NOT NULL,
    ttl        REAL NOT NULL,
    source     TEXT NOT NULL
);
-- `perception` is append-only while `signal` is overwritten in place, and that asymmetry is
-- deliberate: a belief expires and Scene Context needs *newest per key*, whereas a signal
-- holds until something commands it. The history of the signals is in `observation_raw`,
-- because every frame that arrived is there.
--
-- Append-only within a session, and NOT forever: `Store.compact_perception` deletes rows that
-- are not the newest for their key and are past their own ttl, because no read can return one
-- and `live()` walks the whole table looking for DISTINCT key. That is the only thing keeping
-- this table small enough for the scene engine's hot path.
CREATE INDEX IF NOT EXISTS perception_newest ON perception(key, at DESC);

CREATE TABLE IF NOT EXISTS utterance (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_id INTEGER REFERENCES observation_raw(id) ON DELETE SET NULL,
    at     REAL NOT NULL,
    text   TEXT                      -- nullable: retention clears it, the row survives
);

CREATE TABLE IF NOT EXISTS turn (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    -- ON DELETE SET NULL for the same reason perception and utterance have it, and it was
    -- missed here: a plain reference PINS the raw row, and every voice or camera input opens
    -- a turn, so the sweep raised on all of them. Only a raw row that reached no handler --
    -- the case the first retention tests happened to use -- could be deleted at all.
    raw_id  INTEGER REFERENCES observation_raw(id) ON DELETE SET NULL,   -- what triggered it
    at      REAL NOT NULL,
    kind    TEXT NOT NULL,           -- route | scene | consent
    reply   TEXT NOT NULL DEFAULT '' -- what the driver heard; '' is silence, and is a fact
);

-- One table serves routing and rules because they are the same shape: a subject, a verdict,
-- what was chosen, and why. Two tables would drift. The alternative -- a JSON blob -- is
-- unqueryable, and being able to ask *why* is the entire point of writing this down.
CREATE TABLE IF NOT EXISTS decision (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id       INTEGER NOT NULL REFERENCES turn(id),
    subject       TEXT NOT NULL,     -- the clause text, or the rule id
    verdict       TEXT NOT NULL,     -- band for routing; Verdict for a rule
    chosen        TEXT,              -- function name, or scene id
    reason        TEXT NOT NULL DEFAULT '',
    suppressed_by TEXT NOT NULL DEFAULT ''
);

-- sim/migrate.py's own ledger. It is listed here so this file states the whole shape, but
-- migrate.py creates it too when it is absent -- it cannot depend on this script having run
-- to find out whether this script has run.
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
