"""What perception currently believes, and for how long.

Holds perception ONLY. Vehicle state is read live from the car (scene/facts.py), because
copying it here would recreate the two-beliefs-about-one-actuator problem that signal-keyed
state was built to prevent — see sim/mapping.py's module docstring.

Staleness is evaluated at read time rather than by a sweeper: there is no clock to run on the
target SoC, and every test can state `now` instead of sleeping.
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


class SceneContext:
    def __init__(self):
        self._by_key: dict[str, Observation] = {}

    def update(self, obs: Observation) -> None:
        prev = self._by_key.get(obs.key)
        # Keep the newest by its OWN timestamp, not by arrival order: a delayed frame must
        # not overwrite a fresher belief about the same key.
        if prev is None or obs.at >= prev.at:
            self._by_key[obs.key] = obs

    def clear(self) -> None:
        """Forget everything, in place.

        In place rather than by rebinding, because anything holding this store keeps holding
        it across a reset. A caller that rebound `engine.context = SceneContext()` would leave
        every existing reader pointed at the discarded instance, and the failure is silence —
        perception reads empty forever, with nothing raising.
        """
        self._by_key.clear()

    def get(self, key: str, now: float) -> Optional[Observation]:
        obs = self._by_key.get(key)
        return obs if obs is not None and obs.is_live(now) else None

    def live(self, now: float) -> dict[str, Observation]:
        return {k: o for k, o in self._by_key.items() if o.is_live(now)}
