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
