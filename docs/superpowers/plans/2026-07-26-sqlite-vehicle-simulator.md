# SQLite Vehicle Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the always-succeeds `MockExecutor` with a SQLite-backed simulated vehicle whose state
every operation actually changes, and which can **refuse** an operation for reasons validation cannot see.

**Architecture:** `sim/` is a peer of `t2f/`, injected through the existing `execute()` seam and swapped
for a real bus adapter on a vehicle. The DB *is* the car: rows are **signals** — `(entity, attribute)` —
not functions, so `open_window` and `set_window_position` write the same physical window. Operations
are transactional and logged. Physical limits live on the signal and may be tighter than the card's.

**Tech stack:** Python stdlib `sqlite3` only. No new dependencies.

**Decisions already taken (do not relitigate):** signal-level keying — approved. Physical limits
distinct from catalog limits — approved. Simulator may refuse — approved.

---

## Design in one picture

```
route() → validate (catalog limits) → execute(ToolCall) ─┐
                                                          ▼
                                         ┌──────── sim/ (the car) ────────┐
                                         │ 1 map function+params→signals  │
                                         │ 2 device available?            │
                                         │ 3 preconditions hold?          │
                                         │ 4 within PHYSICAL limits?      │
                                         │ 5 write signals (1 txn)        │
                                         │ 6 append operation_log         │
                                         └────────────┬───────────────────┘
                                                      ▼
                                          ExecResult(ok, error, detail)
                                                      │
      reply must not claim success when ok is False ◀─┘
```

`snapshot()` reads signals back in `state_key()` form, so `VehicleState.live` finally has a producer
and relative operations (`再开一点`) resolve against a real car.

---

## File structure

| File | Responsibility |
|---|---|
| `sim/__init__.py` | exports `SqliteVehicle`, `SqliteExecutor` |
| `sim/schema.sql` | the four tables |
| `sim/mapping.py` | function+params → `SignalWrite[]`, and the reverse for `snapshot()` |
| `sim/vehicle.py` | `SqliteVehicle` — connect, init, read, write-in-transaction, log, snapshot |
| `sim/executor.py` | `SqliteExecutor.execute(ToolCall) -> ExecResult` |
| `sim/seed.py` | build signals from the catalog + seed preconditions/devices |
| `t2f/types.py` | `+ ExecResult`, `+ ClauseResult.exec_error` |
| `t2f/execute.py` | `MockExecutor` returns `ExecResult` |
| `t2f/pipeline.py`, `t2f/plan.py` | three call sites read the result |
| `t2f/reply.py` | speak the refusal cause |
| `tests/sim/` | unit tests |
| `tests/e2e/test_s5_simulator.py` | workflow tests through `route()` |

---

### Task 1: Schema and `SqliteVehicle`

**Files:** Create `sim/__init__.py`, `sim/schema.sql`, `sim/vehicle.py`; Test `tests/sim/test_vehicle.py`

- [ ] **Step 1: Write `sim/schema.sql`**

```sql
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
```

- [ ] **Step 2: Write the failing test** — `tests/sim/test_vehicle.py`

```python
import json, pytest
from sim.vehicle import SqliteVehicle


@pytest.fixture
def car():
    v = SqliteVehicle(":memory:")
    v.init_schema()
    return v


def test_set_and_get_signal(car):
    car.set_signal("window.driver", "position", 40, unit="percent", limits=(0, 100))
    assert car.get_signal("window.driver", "position") == 40


def test_value_keeps_its_type(car):
    car.set_signal("climate.all", "ac_power", True)
    assert car.get_signal("climate.all", "ac_power") is True
    car.set_signal("climate.driver", "temperature", 22.5, unit="celsius")
    assert car.get_signal("climate.driver", "temperature") == 22.5


def test_missing_signal_is_none(car):
    assert car.get_signal("nope.all", "nothing") is None


def test_limits_are_readable(car):
    car.set_signal("window.driver", "position", 40, unit="percent", limits=(0, 60))
    assert car.limits_of("window.driver", "position") == (0.0, 60.0)


def test_write_many_is_atomic(car):
    car.set_signal("window.driver", "position", 10, limits=(0, 100))
    with pytest.raises(ValueError):
        car.write_many([("window.driver", "position", 50), ("bad", None, 1)])
    assert car.get_signal("window.driver", "position") == 10      # rolled back


def test_log_records_both_outcomes(car):
    car.log("open_window", {"is_open": True}, "executed", None, "")
    car.log("set_temperature", {"temperature": 25}, "refused", "precondition_failed", "空调未开启")
    rows = car.recent_operations()
    assert [r["outcome"] for r in rows] == ["refused", "executed"]   # newest first
    assert rows[0]["error"] == "precondition_failed"


def test_device_availability(car):
    assert car.is_available("window.driver") == (True, None)
    car.set_device("window.driver", False, "执行器无响应")
    assert car.is_available("window.driver") == (False, "执行器无响应")
```

- [ ] **Step 3: Run to verify it fails**

Run: `python3 -m pytest tests/sim/test_vehicle.py -q` → FAIL, `ModuleNotFoundError: No module named 'sim'`

- [ ] **Step 4: Implement `sim/vehicle.py`**

```python
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
```

- [ ] **Step 5: Run** → `python3 -m pytest tests/sim/test_vehicle.py -q` → **7 passed**

- [ ] **Step 6: Commit** — `git add sim tests/sim && git commit -m "feat(sim): SQLite vehicle — signal store, device status, preconditions, operation log"`

---

### Task 2: Function → signal mapping

**Files:** Create `sim/mapping.py`; Test `tests/sim/test_mapping.py`

The default is `entity = f"{domain}.{position or 'all'}"`, `attribute = <the primary param name>`.
Overrides exist **only** where two functions address the same physical thing, or where the default
attribute name would be uselessly generic (`enabled`).

- [ ] **Step 1: Write the failing test**

```python
from t2f.cards import load_catalog
from t2f.types import ToolCall
from sim.mapping import resolve_writes, signal_for_function

CARDS = {c.name: c for c in load_catalog("data/catalog")}


def test_default_mapping_uses_domain_and_position():
    w = resolve_writes(CARDS["set_temperature"], ToolCall("set_temperature",
                                                          {"temperature": 25, "position": "driver"}))
    assert w == [("climate.driver", "temperature", 25)]


def test_missing_position_is_all():
    w = resolve_writes(CARDS["set_ac_power"], ToolCall("set_ac_power", {"enabled": True}))
    assert w == [("climate.all", "ac_power", True)]


def test_open_window_and_set_position_write_the_same_signal():
    a = resolve_writes(CARDS["open_window"],
                       ToolCall("open_window", {"is_open": True, "position": "driver"}))
    b = resolve_writes(CARDS["set_window_position"],
                       ToolCall("set_window_position", {"percent": 40, "position": "driver"}))
    assert a[0][0] == b[0][0] and a[0][1] == b[0][1] == "position"
    assert a[0][2] == 100 and b[0][2] == 40


def test_closing_a_window_is_position_zero():
    w = resolve_writes(CARDS["open_window"],
                       ToolCall("open_window", {"is_open": False, "position": "rear"}))
    assert w == [("window.rear", "position", 0)]


def test_reverse_lookup_for_state_resolver():
    assert signal_for_function(CARDS["set_temperature"], "driver") == ("climate.driver", "temperature")
```

- [ ] **Step 2: Run to verify it fails** → `ModuleNotFoundError: No module named 'sim.mapping'`

- [ ] **Step 3: Implement `sim/mapping.py`**

```python
"""function + params  ->  the signals it writes.

Default: entity = "<domain>.<position|all>", attribute = the primary parameter's name.
Overrides exist ONLY where two functions address the same physical thing (a window can be
addressed as open/closed or as a percentage) or where the default attribute name would be
uselessly generic ("enabled"). Add an entry when a collision appears — not before.
"""
from __future__ import annotations
from t2f.types import FunctionCard, ToolCall
from t2f.state import primary_numeric_param

# function -> (attribute, transform)   transform: raw param value -> stored signal value
_OVERRIDES = {
    "open_window":           ("position", lambda v: 100 if v else 0),
    "set_window_position":   ("position", None),
    "open_sunroof":          ("position", lambda v: 100 if v else 0),
    "set_sunroof_position":  ("position", None),
    "set_ac_power":          ("ac_power", None),
    "set_window_child_lock": ("child_lock", None),
}


def _primary_param(card: FunctionCard) -> str | None:
    p = primary_numeric_param(card)
    if p is not None:
        return p.name
    req = [x for x in card.required_params if x != "position"]
    return req[0] if req else None


def _entity(card: FunctionCard, params: dict) -> str:
    return f"{card.domain}.{params.get('position') or 'all'}"


def resolve_writes(card: FunctionCard, tool_call: ToolCall) -> list[tuple]:
    """[(entity, attribute, value)] for this call. Empty when nothing is addressable."""
    params = tool_call.parameters
    source = _primary_param(card)
    if source is None or source not in params:
        return []
    attribute, transform = _OVERRIDES.get(card.name, (source, None))
    value = params[source]
    return [(_entity(card, params), attribute, transform(value) if transform else value)]


def signal_for_function(card: FunctionCard, position: str | None = None) -> tuple | None:
    """Reverse lookup so StateResolver can read the current value back."""
    source = _primary_param(card)
    if source is None:
        return None
    attribute, _ = _OVERRIDES.get(card.name, (source, None))
    return (f"{card.domain}.{position or 'all'}", attribute)
```

- [ ] **Step 4: Run** → `python3 -m pytest tests/sim/test_mapping.py -q` → **5 passed**

- [ ] **Step 5: Commit** — `git commit -am "feat(sim): function->signal mapping, so aliasing functions share one physical signal"`

---

### Task 3: Seeding the car from the catalog

**Files:** Create `sim/seed.py`; Test `tests/sim/test_seed.py`

- [ ] **Step 1: Write the failing test**

```python
from t2f.cards import load_catalog
from sim.vehicle import SqliteVehicle
from sim.seed import seed_from_catalog

CARDS = load_catalog("data/catalog")


def _car():
    v = SqliteVehicle(":memory:"); v.init_schema(); seed_from_catalog(v, CARDS); return v


def test_numeric_signals_exist_with_limits():
    car = _car()
    lo, hi = car.limits_of("climate.driver", "temperature")
    assert (lo, hi) == (16.0, 32.0)


def test_signals_start_at_a_plausible_value():
    car = _car()
    assert car.get_signal("climate.driver", "temperature") is not None


def test_preconditions_are_seeded():
    car = _car()
    names = [p["attribute"] for p in car.preconditions_for("set_temperature")]
    assert "ac_power" in names


def test_child_lock_precondition_on_rear_window():
    car = _car()
    assert car.preconditions_for("open_window")
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Implement `sim/seed.py`**

```python
"""Build a plausible starting car from the catalog.

Physical limits default to the card's declared min/max but are DELIBERATELY separate: a card
says a window is 0-100, a physical window can be jammed at 60. Validation and actuation are
different questions, and requirement 4b's third branch only exists because they can disagree.
"""
from __future__ import annotations
from t2f.types import FunctionCard
from t2f.state import primary_numeric_param
from sim.mapping import signal_for_function
from sim.vehicle import SqliteVehicle

_POSITIONS = ["driver", "passenger", "rear", "all"]

# (function, entity, attribute, required_value, what the driver is told)
_PRECONDITIONS = [
    ("set_temperature", "climate.all", "ac_power", True, "空调尚未开启"),
    ("set_fan_speed",   "climate.all", "ac_power", True, "空调尚未开启"),
    ("open_window",     "window.all",  "child_lock", False, "车窗儿童锁已开启"),
]


def seed_from_catalog(car: SqliteVehicle, cards: list[FunctionCard]) -> None:
    for card in cards:
        p = primary_numeric_param(card)
        for pos in (_POSITIONS if card.param("position") else [None]):
            sig = signal_for_function(card, pos)
            if sig is None:
                continue
            entity, attribute = sig
            if p is not None:
                lo = p.minimum if p.minimum is not None else 0
                hi = p.maximum if p.maximum is not None else 100
                start = lo + (hi - lo) // 2 if p.type == "integer" else (lo + hi) / 2
                car.set_signal(entity, attribute, start, unit=p.unit, limits=(lo, hi))
            else:
                car.set_signal(entity, attribute, False)
    # a car with the A/C on and the child lock off is the useful default
    car.set_signal("climate.all", "ac_power", True)
    car.set_signal("window.all", "child_lock", False)
    for fn, entity, attr, equals, detail in _PRECONDITIONS:
        car.add_precondition(fn, entity, attr, equals, detail)
```

- [ ] **Step 4: Run** → **4 passed**. If a limit assertion fails, read the real card and use the real number — do not change the card.

- [ ] **Step 5: Commit** — `git commit -am "feat(sim): seed the car from the catalog with physical limits and preconditions"`

---

### Task 4: `ExecResult` and `SqliteExecutor`

**Files:** Modify `t2f/types.py`, `t2f/execute.py`; Create `sim/executor.py`; Test `tests/sim/test_executor.py`

- [ ] **Step 1: Add the result contract to `t2f/types.py`** (after `ValidationError`)

```python
@dataclass
class ExecResult:
    """What the vehicle reports back. The router MUST read this: an operation that was
    dispatched is not an operation that happened."""
    ok: bool
    error: Optional[str] = None    # device_unavailable | precondition_failed | out_of_range
    detail: str = ""               # driver-usable specifics
```

and add to `ClauseResult` (last field, so keyword construction is unaffected):

```python
    exec_error: Optional[ValidationError] = None
```

- [ ] **Step 2: `t2f/execute.py` returns it**

```python
# t2f/execute.py
from .types import ToolCall, ExecResult


class MockExecutor:
    """Always succeeds. Kept for tests that do not care about the vehicle; `sim/` is the
    simulated car."""
    def execute(self, tool_call: ToolCall) -> ExecResult:
        return ExecResult(ok=True)
```

- [ ] **Step 3: Write the failing executor test** — `tests/sim/test_executor.py`

```python
import pytest
from t2f.cards import load_catalog
from t2f.types import ToolCall
from sim.vehicle import SqliteVehicle
from sim.seed import seed_from_catalog
from sim.executor import SqliteExecutor

CARDS = load_catalog("data/catalog")
BY = {c.name: c for c in CARDS}


@pytest.fixture
def ex():
    car = SqliteVehicle(":memory:"); car.init_schema(); seed_from_catalog(car, CARDS)
    return SqliteExecutor(car, BY)


def test_operation_changes_state(ex):
    r = ex.execute(ToolCall("set_temperature", {"temperature": 25, "position": "driver"}))
    assert r.ok
    assert ex.car.get_signal("climate.driver", "temperature") == 25


def test_aliasing_functions_move_the_same_window(ex):
    ex.execute(ToolCall("set_window_position", {"percent": 40, "position": "driver"}))
    assert ex.car.get_signal("window.driver", "position") == 40
    ex.execute(ToolCall("open_window", {"is_open": False, "position": "driver"}))
    assert ex.car.get_signal("window.driver", "position") == 0


def test_precondition_refusal(ex):
    ex.car.set_signal("climate.all", "ac_power", False)
    r = ex.execute(ToolCall("set_temperature", {"temperature": 25, "position": "driver"}))
    assert not r.ok and r.error == "precondition_failed" and "空调" in r.detail


def test_refusal_does_not_change_state(ex):
    before = ex.car.get_signal("climate.driver", "temperature")
    ex.car.set_signal("climate.all", "ac_power", False)
    ex.execute(ToolCall("set_temperature", {"temperature": 25, "position": "driver"}))
    assert ex.car.get_signal("climate.driver", "temperature") == before


def test_device_unavailable_refusal(ex):
    ex.car.set_device("window.driver", False, "执行器无响应")
    r = ex.execute(ToolCall("open_window", {"is_open": True, "position": "driver"}))
    assert not r.ok and r.error == "device_unavailable"


def test_physical_limit_tighter_than_the_card(ex):
    """The card says 0-100. This window is jammed at 60. Validation cannot see that."""
    ex.car.set_signal("window.driver", "position", 10, unit="percent", limits=(0, 60))
    r = ex.execute(ToolCall("set_window_position", {"percent": 90, "position": "driver"}))
    assert not r.ok and r.error == "out_of_range" and "60" in r.detail


def test_every_attempt_is_logged(ex):
    ex.execute(ToolCall("set_temperature", {"temperature": 25, "position": "driver"}))
    ex.car.set_device("window.driver", False, "x")
    ex.execute(ToolCall("open_window", {"is_open": True, "position": "driver"}))
    outcomes = [r["outcome"] for r in ex.car.recent_operations()]
    assert outcomes == ["refused", "executed"]


def test_snapshot_feeds_state_resolver(ex):
    ex.execute(ToolCall("set_temperature", {"temperature": 25, "position": "driver"}))
    assert ex.snapshot()["set_temperature/driver"] == 25
```

- [ ] **Step 4: Run to verify it fails**

- [ ] **Step 5: Implement `sim/executor.py`**

```python
"""The executor seam, backed by the simulated car.

Order matters: availability, then preconditions, then physical limits, then write. A refusal
must leave the car exactly as it was, and every attempt is logged either way.
"""
from __future__ import annotations
from t2f.types import ToolCall, ExecResult, FunctionCard
from t2f.state import state_key, primary_numeric_param
from sim.mapping import resolve_writes, signal_for_function
from sim.vehicle import SqliteVehicle


class SqliteExecutor:
    def __init__(self, car: SqliteVehicle, cards_by_name: dict[str, FunctionCard]):
        self.car = car
        self.cards = cards_by_name

    def execute(self, tool_call: ToolCall) -> ExecResult:
        card = self.cards.get(tool_call.name)
        if card is None:
            return self._refuse(tool_call, "unknown_function", f"{tool_call.name} 不在功能表中")

        writes = resolve_writes(card, tool_call)
        if not writes:
            self.car.log(tool_call.name, tool_call.parameters, "executed", None, "")
            return ExecResult(ok=True)            # nothing addressable; a no-op still succeeds

        entity = writes[0][0]
        available, reason = self.car.is_available(entity)
        if not available:
            return self._refuse(tool_call, "device_unavailable", reason or f"{entity} 当前不可用")

        for pre in self.car.preconditions_for(card.name):
            if self.car.get_signal(pre["entity"], pre["attribute"]) != pre["equals"]:
                return self._refuse(tool_call, "precondition_failed", pre["detail"])

        for ent, attr, value in writes:
            lo, hi = self.car.limits_of(ent, attr)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if lo is not None and value < lo:
                    return self._refuse(tool_call, "out_of_range", f"{attr} 最低只能到 {int(lo)}")
                if hi is not None and value > hi:
                    return self._refuse(tool_call, "out_of_range", f"{attr} 最高只能到 {int(hi)}")

        self.car.write_many(writes)
        self.car.log(tool_call.name, tool_call.parameters, "executed", None, "")
        return ExecResult(ok=True)

    def _refuse(self, tool_call: ToolCall, error: str, detail: str) -> ExecResult:
        self.car.log(tool_call.name, tool_call.parameters, "refused", error, detail)
        return ExecResult(ok=False, error=error, detail=detail)

    def snapshot(self) -> dict:
        """Current car state keyed the way VehicleState/StateResolver expect."""
        out = {}
        for card in self.cards.values():
            if primary_numeric_param(card) is None:
                continue
            for pos in ("driver", "passenger", "rear", "all", None):
                sig = signal_for_function(card, pos)
                if sig is None:
                    continue
                value = self.car.get_signal(*sig)
                if value is not None:
                    params = {"position": pos} if pos else {}
                    out[state_key(card.name, params)] = value
        return out
```

- [ ] **Step 6: Run** → **8 passed**

- [ ] **Step 7: Full suite** — `python3 -m pytest -q`. `MockExecutor` now returns `ExecResult`;
`tests/test_respond.py` asserts `MockExecutor().execute(...)["ok"] is True` and must become
`.ok is True`. Fix that one assertion; report anything else that breaks.

- [ ] **Step 8: Commit** — `git commit -am "feat(sim): SqliteExecutor — refuses, logs, and never half-writes"`

---

### Task 5: Make the router read the result (closes gap 1)

**Files:** Modify `t2f/pipeline.py:64,104`, `t2f/plan.py:43`; Test `tests/e2e/test_s3_execution.py`

**This is the change the whole plan exists for.** Until now `execute()`'s result was discarded, so a
refused operation was still spoken as a success.

- [ ] **Step 1: `t2f/pipeline.py:101-106` (DeterministicResolver)** — replace the unconditional render:

```python
        tc, errs = validate_tool_call(decision.chosen, params, self.cards, cand_names)
        if tc is None:
            return ClauseResult(clause=clause, decision=decision, validation_errors=errs)
        res = self.executor.execute(tc)
        if not res.ok:
            return ClauseResult(clause=clause, decision=decision, tool_call=tc,
                                exec_error=ValidationError(res.error or "exec_failed", res.detail))
        return ClauseResult(clause=clause, decision=decision, tool_call=tc,
                            response=render_response(card, tc))
```

- [ ] **Step 2: `t2f/pipeline.py:60-66` (LLMResolver)** — same shape:

```python
            tc, errs = validate_tool_call(res.tool_call.name, res.tool_call.parameters, cards_by_name, offered_names)
            if tc is not None:
                card = cards_by_name[tc.name]
                exec_res = executor.execute(tc) if executor is not None else None
                if exec_res is not None and not exec_res.ok:
                    return ClauseResult(clause=clause, decision=decision, tool_call=tc, needs_llm=True,
                                        exec_error=ValidationError(exec_res.error or "exec_failed", exec_res.detail))
                return ClauseResult(clause=clause, decision=decision, tool_call=tc,
                                    response=render_response(card, tc), needs_llm=True)
```

- [ ] **Step 3: `t2f/plan.py:39-49`** — do not mark executed, and do not commit state, unless it worked:

```python
        # Phase 2: barrier passed — execute the valid subset in order
        executed = []
        for a in plan.actions:
            if a.status == "valid":
                res = self.executor.execute(a.tool_call)
                if not res.ok:
                    a.status = "failed"
                    a.error = res.error or "exec_failed"
                    a.detail = res.detail
                    continue
                p = primary_numeric_param(self.cards[a.function])
                if p is not None and p.name in a.tool_call.parameters:
                    self.state.set(state_key(a.function, a.tool_call.parameters),
                                   a.tool_call.parameters[p.name], layer="confirmed")
                a.status = "executed"
                executed.append(a)
```

Add `detail: str = ""` to `PlannedAction` in `t2f/types.py`. In `t2f/pipeline.py::_route_plan`, a
`failed` action must set `cr.exec_error`, not a clarification:

```python
            elif a is not None and a.status == "failed":
                cr.exec_error = ValidationError(a.error or "exec_failed", a.detail)
                cr.needs_llm = (source == "llm")
            elif a is not None:
                cr.clarification = clar
```

Import `ValidationError` in `t2f/pipeline.py` if it is not already imported.

- [ ] **Step 4: Promote the two red cases**

`test_s3_08_failed_actuation_is_not_confirmed_as_success` and
`test_s3_09_failed_action_does_not_commit_vehicle_state` will now PASS, which under
`xfail(strict=True)` is reported as a FAILURE. That is the mechanism working. **Remove both
`@pytest.mark.xfail` decorators and the now-unused `GAP1` constant**, and update the docstrings to say
the behaviour is now guaranteed rather than pending.

`tests/e2e/doubles.py::FailingExecutor` must also return `ExecResult(ok=False, ...)`.

- [ ] **Step 5: Run** → `python3 -m pytest -q`. Expect **9 xfailed**, not 11, and zero failures.

- [ ] **Step 6: Commit** — `git commit -am "fix: the router reads the executor result — a refused operation is no longer spoken as success"`

---

### Task 6: Speak the refusal cause (4b's third branch)

**Files:** Modify `t2f/reply.py`; Test `tests/test_reply_exec.py`

Scope discipline: this maps **executor** causes only. The full ten-code validation table is gap 2 and
stays out of this plan.

- [ ] **Step 1: Write the failing test**

```python
from t2f.reply import compose_reply
from t2f.types import RouteResult, ClauseResult, Decision, Band, ValidationError


def _clause(exec_error=None, response=None):
    return ClauseResult(clause="x", decision=Decision(Band.HIGH, "f", []),
                        response=response, exec_error=exec_error)


def test_refusal_states_the_cause():
    r = RouteResult(utterance="u", clauses=[_clause(ValidationError("precondition_failed", "空调尚未开启"))])
    assert "空调尚未开启" in compose_reply(r)


def test_refusal_never_claims_success():
    r = RouteResult(utterance="u", clauses=[_clause(ValidationError("device_unavailable", "执行器无响应"))])
    reply = compose_reply(r)
    assert "已" not in reply and "执行器无响应" in reply


def test_a_refusal_beside_a_success_reports_both():
    r = RouteResult(utterance="u", clauses=[
        _clause(response="已开启车窗。"),
        _clause(ValidationError("precondition_failed", "空调尚未开启"))])
    reply = compose_reply(r)
    assert "已开启车窗。" in reply and "空调尚未开启" in reply


def test_exec_error_without_detail_falls_back_to_the_generic_line():
    r = RouteResult(utterance="u", clauses=[_clause(ValidationError("exec_failed", ""))])
    assert compose_reply(r) == "抱歉，这个操作没能完成。"
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Implement** — in `t2f/reply.py`, add after `_questions`:

```python
def _exec_failures(clauses) -> list[str]:
    """Vehicle-reported refusals, in clause order, de-duplicated. The detail is authored by
    the simulator for the driver, so it is spoken verbatim."""
    out: list[str] = []
    for cl in clauses:
        err = getattr(cl, "exec_error", None)
        detail = ((err.message if err else "") or "").strip()
        if detail and detail not in out:
            out.append(detail)
    return out
```

and in `compose_reply`, between the confirmations and the question:

```python
    parts = [_sentence(t) for t in _confirmations(clauses)]
    parts += [_sentence(t) for t in _exec_failures(clauses)]
    questions = _questions(clauses)
    if questions:
        parts.append(_sentence(questions[0]))
    elif _has_failure(clauses) and not _exec_failures(clauses):
        parts.append(_FAILURE)
    return "".join(parts) if parts else _ACK
```

`_has_failure` must also treat a clause carrying only an `exec_error` as *not* silently fine — it
already does, because such a clause has no `response` and no question.

- [ ] **Step 4: Run** → **4 passed**, then the full suite → zero failures.

- [ ] **Step 5: Commit** — `git commit -am "feat: speak the vehicle's refusal cause — requirement 4b's third branch"`

---

### Task 7: End-to-end through `route()` with a real simulated car

**Files:** Create `tests/e2e/test_s5_simulator.py`

- [ ] **Step 1: Write the workflow tests**

```python
"""S5 — the four-step workflow against a SQLite-simulated car.

Unlike the rest of tests/e2e/, these use the REAL 92-card catalog, because the point is that
operations change a real vehicle's state, not that routing works on three fixture cards.
"""
import pytest
from t2f.cards import load_catalog
from t2f.config import Config
from t2f.embed import FakeEmbedder
from t2f.gate import ConfidenceGate, Thresholds
from t2f.pipeline import Pipeline, DeterministicResolver
from t2f.score import Scorer
from sim.vehicle import SqliteVehicle
from sim.seed import seed_from_catalog
from sim.executor import SqliteExecutor

CARDS = load_catalog("data/catalog")
BY = {c.name: c for c in CARDS}


def _pipeline(tmp_path=None):
    car = SqliteVehicle(":memory:"); car.init_schema(); seed_from_catalog(car, CARDS)
    ex = SqliteExecutor(car, BY)
    cfg = Config.default(); cfg.thresholds = Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)
    pipe = Pipeline(CARDS, FakeEmbedder(256), Scorer(cfg.weights, cfg.domain_keywords),
                    ConfidenceGate(cfg.thresholds), cfg,
                    resolver=DeterministicResolver(BY, executor=ex))
    return pipe, ex


def test_step3_an_operation_changes_the_car():
    pipe, ex = _pipeline()
    before = ex.car.get_signal("climate.driver", "temperature")
    pipe.route("把主驾温度调到25度")
    after = ex.car.get_signal("climate.driver", "temperature")
    assert after != before or after == 25


def test_step4b_a_refused_operation_is_not_confirmed():
    """The whole point: the car says no, and the driver is told why."""
    pipe, ex = _pipeline()
    ex.car.set_signal("climate.all", "ac_power", False)
    result = pipe.route("把主驾温度调到25度")
    assert "空调尚未开启" in result.reply
    assert "已将" not in result.reply


def test_a_refusal_leaves_the_car_untouched():
    pipe, ex = _pipeline()
    ex.car.set_signal("climate.all", "ac_power", False)
    before = ex.car.get_signal("climate.driver", "temperature")
    pipe.route("把主驾温度调到25度")
    assert ex.car.get_signal("climate.driver", "temperature") == before


def test_every_attempt_reaches_the_operation_log():
    pipe, ex = _pipeline()
    pipe.route("把主驾温度调到25度")
    assert len(ex.car.recent_operations()) >= 1


def test_snapshot_lets_a_relative_command_resolve():
    """The live state layer finally has a producer."""
    pipe, ex = _pipeline()
    ex.car.set_signal("climate.driver", "temperature", 22)
    pipe.state.reset(live=ex.snapshot())
    assert pipe.state.get("set_temperature/driver") == 22
```

- [ ] **Step 2: Run.** `FakeEmbedder` over the full 92-card catalog may not route every utterance as
intended. **Probe first** — if `把主驾温度调到25度` does not reach `set_temperature`, print what it
actually routed to and pick an utterance that does, or seed the assertion from the measured result.
**Do not weaken an assertion to make it pass** — if the workflow genuinely does not work, report it.

- [ ] **Step 3: Full suite** → zero failures, 9 xfailed.

- [ ] **Step 4: Commit** — `git commit -am "test(e2e): the four-step workflow against a SQLite-simulated car"`

---

### Task 8: Package, document, verify

- [ ] **Step 1:** `pyproject.toml` — add `sim*` to `[tool.setuptools.packages.find] include`.

- [ ] **Step 2:** `README.md` — add `sim/` to the layout block and a short section: the DB is the car,
signals not functions, operations are transactional and logged, and it swaps for the bus adapter.

- [ ] **Step 3:** `docs/superpowers/RESULTS.md` — append a section recording what the simulator closed
(gap 1, and 4b's vehicle-refusal branch), the red count going 11 → 9, and what is still unmet
(the ten-code validation table, gap 2).

- [ ] **Step 4: Regression proof.** `python3 -m eval.run_eval --arm C --dataset data/eval/gold.jsonl --calibrate`
must reproduce: recall@1 **0.8644**, set-recall **0.8194**, param_exact **0.2733**, e2e_det **0.1067**,
incorrect **0.0312**, OOD/context **0.0000**, `invalid_no_execution_rate` **1.0000**. The eval uses
`MockExecutor`, so nothing should move. **Any movement is a bug in Task 5.**

- [ ] **Step 5: Commit.**

## Definition of done

1. `python3 -m pytest -q` — zero failures, **9 xfailed** (down from 11).
2. Arm C reproduces every metric to four decimals.
3. An operation changes a signal; a refusal changes nothing and is spoken with its cause.
4. `open_window` and `set_window_position` demonstrably move the same signal.
5. Nothing under `research/` touched; `t2f/` imports nothing from `sim/`.
