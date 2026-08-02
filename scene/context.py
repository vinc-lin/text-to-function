"""What perception currently believes, and for how long.

Holds perception ONLY. Vehicle state is read live from the car (intake/hub.py), because
copying it here would recreate the two-beliefs-about-one-actuator problem that signal-keyed
state was built to prevent — see sim/mapping.py's module docstring.

Staleness is evaluated at read time rather than by a sweeper: there is no clock to run on the
target SoC, and every test can state `now` instead of sleeping.

**The beliefs are rows now, not a dict.** This class kept its name and its four methods
because `WorldView` binds `get` and `live` and every rule sits above them; only the backing
moved, into `intake.store.Store`. What stayed here is the part that must not move: liveness is
`Observation.is_live`, in Python, applied to the newest row — never a `WHERE at + ttl >= ?`,
which would hand back an older *live* row when the newest has expired and resurrect a belief
this class reports as gone. See §5 of docs/superpowers/specs/2026-08-02-the-store-design.md.

The store is duck-typed rather than imported: `intake` imports `scene`, and that edge points
one way. Whoever composes the two hands the store in.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Observation:
    key: str            # "inside.rear_occupant"
    value: Any          # "child"
    confidence: float
    source: str         # "cabin_cam"
    at: float
    ttl: float

    def is_live(self, now: float) -> bool:
        """Inclusive at the boundary: at + ttl is the last live instant."""
        return now <= self.at + self.ttl


def _observation(row) -> Observation:
    """One row, as the rules already know how to read it.

    Rebuilt into the same frozen dataclass rather than passed around as a row, because every
    rule, every report and both display panes take an `Observation` — handing them a mapping
    would put the store's column names into scene/rules.py, and the schema would stop being
    something only `intake` knows about.
    """
    return Observation(key=row["key"], value=row["value"], confidence=row["confidence"],
                       source=row["source"], at=row["at"], ttl=row["ttl"])


class SceneContext:
    def __init__(self, store):
        """`store` is required, and there is deliberately no in-memory fallback.

        A defaulted store would be a second place beliefs can live, and writing to the wrong
        one fails silently: every rule reads an empty context and nothing raises. `SceneEngine`
        already refuses a world built over a different store for exactly that reason; a default
        here would reopen the same hole one level down.
        """
        self._store = store

    def update(self, obs: Observation) -> None:
        """Record a belief — every one, including a late one.

        The dict had to compare timestamps before writing (`obs.at >= prev.at`), because a
        write there destroyed what it replaced and a delayed frame would have overwritten a
        fresher belief. Rows do not replace each other, and `newest_perception` orders by `at`,
        so the out-of-order rule is now enforced by the read instead of by a guard here. Same
        answer, one place: a guard on the write AND an order on the read would be two rules
        about which belief wins, and they can only ever agree by accident.

        `raw_id` is None because an `Observation` has already lost its envelope. What arrived
        is the caller's to record alongside this (intake/ingest.py); inventing a raw row here
        would be a second, thinner account of the same event.
        """
        self._store.put_perception(None, obs.at, obs.key, obs.value, obs.confidence,
                                   obs.ttl, obs.source)

    def clear(self) -> None:
        """Forget everything, in place.

        In place rather than by rebinding, because anything holding this store keeps holding
        it across a reset. A caller that rebound `engine.context = SceneContext(store)` would
        leave every existing reader pointed at the discarded instance, and the failure is
        silence — perception reads empty forever, with nothing raising. A DELETE now rather
        than `dict.clear()`, which changes nothing about that: the object every reader holds is
        the same object afterwards, and it is still over the store that is still being written.
        """
        self._store.clear_perception()

    def get(self, key: str, now: float) -> Optional[Observation]:
        """The newest belief about this key, or None if there is none live.

        Two steps, and the order is the whole point: newest first, liveness second. Asking the
        database for the newest *live* row instead would answer with an older belief whenever
        the newest one has expired — the rule would then act on something perception no longer
        holds, and nothing in the trace would say so.
        """
        row = self._store.newest_perception(key)
        if row is None:
            return None
        obs = _observation(row)
        return obs if obs.is_live(now) else None

    def live(self, now: float) -> dict[str, Observation]:
        """Every live belief, keyed. Same two steps, for every key at once.

        Grows with the table, not with the number of keys — perception is append-only, and at
        10 Hz that is 36,000 rows an hour. Measured at 1.26 ms there, against 8 µs for a
        single-key `get` on its index. Three query shapes were tried and all three are O(rows),
        so the answer was retention on `perception` rather than a cleverer read: the store
        compacts away rows no read can return, which holds this near the `get` figure. The
        table it walks is small because something empties it, not because this query is smart.
        """
        beliefs = (_observation(row) for row in self._store.live_perception_keys())
        return {o.key: o for o in beliefs if o.is_live(now)}
