# Intake and WorldView Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** one door for everything entering the system (`intake/`), and one read-through view of everything currently true (`WorldView`), with vehicle signals finally subject to a freshness discipline.

**Design:** [`docs/superpowers/specs/2026-08-01-intake-and-worldview-design.md`](../specs/2026-08-01-intake-and-worldview-design.md). Read it before starting any task — the reasoning behind each decision lives there, and several of them look arbitrary without it.

**Two proof obligations, checked at the end and after any task that touches `scene/`:**

```bash
python3 -m eval.run_eval --arm C --dataset data/eval/gold.jsonl --fake --permissive   # byte-identical
python3 -m eval.run_scene_eval --arm S                                                 # byte-identical
```

Baseline: `761 passed, 1 skipped, 5 deselected, 0 xfailed`. Run tests **from the repo root** — the catalog path is relative.

---

## Task 1: the envelope and the source registry

**Files:** create `intake/__init__.py`, `intake/envelope.py`, `intake/sources.py`, `tests/intake/__init__.py`, `tests/intake/test_envelope.py`

- [ ] **Step 1 — write the failing tests**

```python
"""An input names where it came from, and a source can only produce what it declares."""
import pytest

from intake.envelope import Input, Percept, SignalWrite, Utterance


def test_an_utterance_carries_its_source_and_time():
    i = Input(source="mic", at=100.0, payload=Utterance("开车窗"))
    assert i.source == "mic" and i.payload.text == "开车窗"


def test_a_percept_carries_confidence_and_ttl():
    i = Input(source="cabin_cam", at=100.0,
              payload=Percept("inside.rear_occupant", "child", 0.9, 300.0))
    assert i.payload.confidence == 0.9 and i.payload.ttl == 300.0


def test_a_signal_write_names_entity_and_attribute():
    i = Input(source="can0", at=100.0, payload=SignalWrite("vehicle.all", "speed_kph", 45.0))
    assert i.payload.entity == "vehicle.all"


def test_a_source_cannot_produce_what_it_does_not_declare():
    """This is what turns `source` from decoration into a claim. Today every observation
    defaults to "cabin_cam" — including vehicle-namespace ones — and nothing notices."""
    with pytest.raises(ValueError, match="cabin_cam"):
        Input(source="cabin_cam", at=100.0,
              payload=SignalWrite("vehicle.all", "speed_kph", 45.0))


def test_an_undeclared_source_is_refused():
    with pytest.raises(ValueError, match="nowhere"):
        Input(source="nowhere", at=100.0, payload=Utterance("hi"))


def test_there_is_no_kind_field():
    """The payload's type IS the kind. A `kind` beside a payload is two statements about one
    fact, and eventually they differ."""
    assert not hasattr(Input(source="mic", at=0.0, payload=Utterance("x")), "kind")


def test_an_input_is_frozen():
    i = Input(source="mic", at=0.0, payload=Utterance("x"))
    with pytest.raises(Exception):
        i.source = "can0"


def test_every_declared_source_names_a_payload_type():
    from intake.sources import SOURCES
    assert SOURCES
    for name, src in SOURCES.items():
        assert src.name == name
        assert src.accepts in (Utterance, Percept, SignalWrite)


def test_only_a_signal_source_may_publish():
    """Publishing means re-stamping held values. Only a continuous measurement has anything
    to re-stamp; an utterance is an event, not a level."""
    from intake.sources import SOURCES
    for src in SOURCES.values():
        if src.publishes:
            assert src.accepts is SignalWrite, src.name
```

- [ ] **Step 2 — run, confirm `ModuleNotFoundError: No module named 'intake'`**

- [ ] **Step 3 — implement**

```python
# intake/sources.py
"""Where an input can come from, and what it is allowed to produce.

A declared registry rather than a free string. Before this, `source` was decoration: every
observation defaulted to "cabin_cam" including vehicle-namespace ones, nothing validated it,
and only the display ever read it. Declaring what a source produces makes the field a claim
something can check.
"""
from __future__ import annotations
from dataclasses import dataclass

from .envelope import Percept, SignalWrite, Utterance


@dataclass(frozen=True)
class Source:
    name: str
    accepts: type
    # Re-stamps its held values when pumped. Only a continuous measurement has anything to
    # re-stamp -- an utterance is an event, not a level -- which test_only_a_signal_source_
    # may_publish enforces.
    publishes: bool = False


SOURCES: dict[str, Source] = {
    "mic":       Source("mic",       accepts=Utterance),
    "cabin_cam": Source("cabin_cam", accepts=Percept),
    "front_cam": Source("front_cam", accepts=Percept),
    "can0":      Source("can0",      accepts=SignalWrite, publishes=True),
}
```

```python
# intake/envelope.py
"""What arrives, and where it came from.

One shape for all three inputs, so provenance and timing are captured once at the edge rather
than three different ways (or, for voice, not at all).

There is deliberately NO `kind` field: the payload's type is the kind. A `kind` beside a
payload is two statements about one fact, and eventually they disagree.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Union


@dataclass(frozen=True)
class Utterance:
    text: str


@dataclass(frozen=True)
class Percept:
    key: str
    value: Any
    confidence: float
    ttl: float


@dataclass(frozen=True)
class SignalWrite:
    entity: str
    attribute: str
    value: Any


Payload = Union[Utterance, Percept, SignalWrite]


@dataclass(frozen=True)
class Input:
    source: str
    at: float
    payload: Payload

    def __post_init__(self):
        # Validated at construction, not at dispatch: an Input that exists is one that could
        # have happened, so nothing downstream needs to re-ask.
        from .sources import SOURCES
        src = SOURCES.get(self.source)
        if src is None:
            raise ValueError(f"{self.source!r} is not a declared source")
        if not isinstance(self.payload, src.accepts):
            raise ValueError(
                f"{self.source!r} produces {src.accepts.__name__}, "
                f"not {type(self.payload).__name__}")
```

- [ ] **Step 4** — `python3 -m pytest tests/intake -q`
- [ ] **Step 5** — commit: `feat(intake): one envelope, and sources that declare what they produce`

---

## Task 2: `WorldView`

**Files:** create `intake/hub.py`, `tests/intake/test_hub.py`

- [ ] **Step 1 — write the failing tests**

```python
"""One view of everything currently true. It owns nothing."""
import pytest

from intake.hub import WorldView
from scene.context import Observation, SceneContext


class SpyCar:
    def __init__(self, **signals):
        self._s = signals
        self.writes = []

    def get_signal(self, entity, attribute):
        return self._s.get((entity, attribute))

    def signal_age(self, entity, attribute, now):
        return 0.0 if (entity, attribute) in self._s else None

    def set_signal(self, *a, **k):
        self.writes.append(a)


def _ctx(**kw):
    ctx = SceneContext()
    ctx.update(Observation(kw.get("key", "inside.rear_occupant"), kw.get("value", "child"),
                           kw.get("confidence", 0.9), "cabin_cam",
                           kw.get("at", 100.0), kw.get("ttl", 300.0)))
    return ctx


def test_an_observation_reads_through_to_the_perception_store():
    world = WorldView(_ctx(), SpyCar())
    assert world.observation("inside.rear_occupant", now=150.0).value == "child"


def test_a_stale_observation_reads_as_absent():
    world = WorldView(_ctx(ttl=10.0), SpyCar())
    assert world.observation("inside.rear_occupant", now=200.0) is None


def test_a_signal_reads_through_to_the_car():
    world = WorldView(SceneContext(), SpyCar(**{("vehicle.all", "speed_kph"): 45.0}))
    assert world.signal("vehicle.all", "speed_kph", now=100.0) == 45.0


def test_a_missing_signal_reads_as_absent():
    assert WorldView(SceneContext(), SpyCar()).signal("no.such", "thing", now=100.0) is None


def test_live_facts_covers_both_stores():
    world = WorldView(_ctx(), SpyCar(**{("vehicle.all", "speed_kph"): 45.0}))
    facts = world.live_facts(now=150.0)
    assert "inside.rear_occupant" in facts
    assert "vehicle.all/speed_kph" in facts


def test_live_facts_omits_what_has_expired():
    world = WorldView(_ctx(ttl=10.0), SpyCar())
    assert world.live_facts(now=200.0) == {}


def test_the_hub_holds_nothing_writable():
    """`read-through` in a docstring is not an enforcement mechanism. VehicleFacts carried this
    property and WorldView absorbs it: a hub that could write would be a second route to the
    car with none of the executor's checks."""
    world = WorldView(SceneContext(), SpyCar())
    reachable = [v for v in vars(world).values() if hasattr(v, "set_signal")]
    assert reachable == []


def test_reading_never_writes():
    car = SpyCar(**{("vehicle.all", "speed_kph"): 45.0})
    world = WorldView(_ctx(), car)
    world.observation("inside.rear_occupant", now=150.0)
    world.signal("vehicle.all", "speed_kph", now=150.0)
    world.live_facts(now=150.0)
    assert car.writes == []
```

- [ ] **Step 2** — run, confirm the import error.

- [ ] **Step 3 — implement `intake/hub.py`.** `WorldView(perception, car)`.

Bind the car's *readers*, not the car — the same move `VehicleFacts.__init__` made, and for the same reason: holding the whole `SqliteVehicle` leaves `set_signal` one attribute access away from any consumer, and `test_the_hub_holds_nothing_writable` walks the instance to prove it does not.

`live_facts(now)` returns `{key: value}` for live observations plus `{f"{entity}/{attribute}": value}` for live signals. It is what the fallback prompt is built from, which is why it must cover both stores — the prompt has never seen vehicle state.

Say in the module docstring that it is **read-through, never a cache**: the moment it holds a value, the two-beliefs-about-one-actuator problem that signal-keyed state was built to prevent comes back one level up.

- [ ] **Step 4** — `python3 -m pytest tests/intake -q`
- [ ] **Step 5** — commit: `feat(intake): WorldView — one view, owning nothing`

---

## Task 3: signals that can go stale

**Files:** modify `sim/seed.py`, `sim/vehicle.py`; modify `tests/sim/test_seed.py`, add `tests/sim/test_staleness.py`

- [ ] `_SENSED` rows gain a trailing `max_age`: `("vehicle.all", "speed_kph", 0.0, "kph", 0.0, 240.0, 2.0)`. Every unpacking site changes — `cli/session.py` has two.
- [ ] `SqliteVehicle.signal_age(entity, attribute, now) -> Optional[float]` reads `updated_at`, which has been written since Spec 7 and read by nothing. `None` for a signal the car does not hold.
- [ ] `sensed_max_age(entity, attribute) -> Optional[float]` in `sim/seed.py`, so one definition answers "how fast does this decay".
- [ ] Tests: a freshly written signal has age ~0; age grows with `now`; a signal the car lacks has age `None`; **an actuated signal has no max age at all** — a window position holds until commanded and does not decay.
- [ ] commit: `feat(sim): a sensed signal can go stale`

---

## Task 4: rules and the engine read the world

The migration. **Broad diff expected** — treat it as an opportunity to check, per the design's §10.

**Files:** modify `scene/rules.py`, `scene/engine.py`; delete `scene/facts.py`; update every construction site and its tests

- [ ] `evaluate(rule, world, now)` and `evaluate_explained(rule, world, now)` replace the `(context, facts, now)` pair. `Signal` and `SignalAbove` call `world.signal(...)`, which now returns `None` when stale.
- [ ] The stale rejection **names both ages**: `vehicle.all/speed_kph is stale (4.2s > 2.0s)`. A bare "not above 5.0" on a stale value would be actively misleading — it reads as "the car is slow" when the truth is "the bus is quiet". `WorldView` therefore needs a way to say *why* a signal read as absent; give it `signal_status(entity, attribute, now)` returning `("live"|"stale"|"missing", detail)` and have `evaluate_explained` use it.
- [ ] `SceneEngine(cards_by_name, world, executor, ...)` replaces the `facts` parameter.
- [ ] **Delete `scene/facts.py`.** Move its read-only test into `tests/intake/test_hub.py` if not already covered — the property must not be lost with the file.
- [ ] `tests/scene/test_contract_sweep.py`'s fact stubs become `WorldView`s over a real `SceneContext` and a spy car. **Every sweep property must still hold for both rules.** If one starts skipping, that is the sweep shrinking to fit again — the exact failure `SignalAbove` caused — and it is a finding, not a licence to test less.
- [ ] Both proof obligations run here and must be byte-identical.
- [ ] commit: `refactor(scene): rules read one world, not two stores`

---

## Task 5: ingest, and the pumped bus

**Files:** create `intake/ingest.py`, `tests/intake/test_ingest.py`

- [ ] `Intake(pipeline, engine, car, world)` — the composition root. `ingest(input) -> Outcome` dispatches on payload type: `Utterance` → `pipeline.route`, `Percept` → `engine.observe`, `SignalWrite` → the car.
- [ ] It holds **no logic of its own** — dispatch and provenance only, every decision still made by the module that owns it. Say so in the docstring; §12 names "intake becomes the god object" as the standing risk.
- [ ] A `SignalWrite` from a publishing source records the value as *held*. `pump(now)` re-stamps every held value for sources currently publishing. `set_publishing(source, on)` stops and starts it.
- [ ] **Pumped, not threaded**, and the docstring says why: `sim/vehicle.py` opens SQLite with thread affinity, and `ui/state.py`'s defensive panes mean a background thread would not raise — it would serve a snapshot with an empty car while everything else rendered fine. It would lie rather than break.
- [ ] Tests: each payload type reaches the right handler; a held value stays fresh across pumps; stops going fresh once publishing is off; `pump` on a non-publishing source is a no-op; an undeclared source cannot reach `ingest` at all because `Input` refuses to exist.
- [ ] commit: `feat(intake): one door, and a bus that keeps publishing`

---

## Task 6: the session becomes a consumer

**Files:** modify `cli/session.py`, `cli/__main__.py`; tests alongside

- [ ] `Session` builds an `Intake` and delegates. `handle`, `observe` and `set_signal` become thin `Input` builders — the public shape stays identical so every existing caller and test keeps working.
- [ ] Each command pumps first, so a live bus is fresh whenever you look.
- [ ] `/bus on|off` toggles publishing, listed in `/help`. `/signal` reports the publishing state: `→ 45.0 kph · publishing @2Hz`.
- [ ] Tests: `/bus off` then a clock advance makes the animal rule reject with the stale reason; `/bus on` restores it; the CLI's existing behaviour is otherwise unchanged.
- [ ] commit: `feat(cli): the session consumes intake, and the bus can stop`

---

## Task 7: the UI

**Files:** modify `ui/state.py`, `ui/actions.py`, `ui/server.py`, `ui/page.html`

- [ ] The snapshot's `sensed` rows gain `age` and `stale`. The Vehicle pane shows age, and a stale row reads clearly as stale rather than as a value.
- [ ] A bus toggle in the Vehicle pane, posting to a `CONTROLS` entry. `CONTROLS` grows to two; the disjointness test with `ACTIONS` still holds.
- [ ] The poll pumps, so the page behaves like the terminal.
- [ ] commit: `feat(ui): the bus, and signals that show their age`

---

## Task 8: verification and docs

- [ ] Both proof obligations, before and after, pasted into the report.
- [ ] Drive the full staleness story by hand in the CLI and paste it: set speed, animal warns, `/bus off`, clock forward, animal rejects naming both ages, `/bus on`, warns again.
- [ ] `docs/TRYING_IT.md`: `/bus` in the commands table, and the staleness story as a worked example.
- [ ] `pyproject.toml`: add `intake*` to the packaged set, and update the exclusion comment.
- [ ] `README.md`: `intake/` in the Layout block.
- [ ] commit: `docs: one door in, one view out`

## Done criteria

- both eval reports byte-identical
- a stale signal rejects with both ages named; a live bus never goes stale
- `Input` cannot be constructed with a source/payload mismatch
- `WorldView` holds nothing writable, asserted by walking the instance
- `scene/facts.py` gone, its property preserved
- `python3 -m pytest -q` green, no new dependency
