"""One view of everything currently true — perception and the car, one liveness question.

**Read-through, never a cache.** It knows who to ask, not the answer. The moment it holds a
value, the two-beliefs-about-one-actuator problem that signal-keyed state was built to prevent
comes back one level up, with its own staleness: `open_window` and `set_window_position` are
keyed by signal so they cannot hold two contradictory beliefs about one window, and a hub
holding a copy of `window_child_lock` rebuilds exactly that, so the car and the hub can
disagree about a lock. Nothing here is stored, so nothing here can be wrong on its own.

It absorbs `scene/facts.py`, whose entire job was the read-only car port, and its safety
property comes with it: a test walks the instance and asserts nothing it holds exposes
`set_signal`, because "read-only" in a docstring is not an enforcement mechanism.

**Absence is the failure mode, and that is deliberate.** A signal past its declared age reads
as `None` — identical to a signal the car does not hold — so every condition on it rejects and
the engine falls silent rather than acting on a value from ten minutes ago. The cost is that a
consumer which forgets to keep the bus pumped sees everything as absent and gets silence with
no error; `signal_status` exists so that silence can at least explain itself.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:                      # a type-only edge: intake does not import scene at run time
    from scene.context import Observation


def _declared_sensed() -> list[tuple[str, str]]:
    """Every signal `live_facts` can ask the car about.

    From the declaration rather than from the car, because a car has no enumeration to offer
    that would be useful here: the signal table is mostly actuator state the router already
    reasons about through the executor, and thirty rows of window positions in the fallback's
    prompt would bury the one fact a scene rule actually conditions on.

    Only the first two fields are read, so Task 3 can append `max_age` to the rows without
    touching this.
    """
    try:
        from sim.seed import sensed_signals
    except ImportError:                # no simulator in this deployment; perception only
        return []
    return [(row[0], row[1]) for row in sensed_signals()]


def _declared_max_age(entity: str, attribute: str) -> Optional[float]:
    """How fast this signal decays, from the one definition that owns the answer.

    Freshness is declared on the SIGNAL, not on the rule, because how fast a value decays is a
    property of the source: speed at 10 Hz is dead after a second, a door-ajar flag is fine for
    a minute. One declaration, every rule inherits it, and no rule can forget to ask.

    `None` means never stale, which is right for an actuated signal — a window position holds
    until something commands it otherwise — and is also the answer before Task 3 lands
    `sensed_max_age`. Tests supply their own rather than trusting this, because a staleness
    check that silently never fires is worse than no staleness check.
    """
    try:
        from sim.seed import sensed_max_age
    except ImportError:
        return None
    return sensed_max_age(entity, attribute)


class WorldView:
    def __init__(self, perception, car):
        # Readers only, from BOTH stores — the move VehicleFacts.__init__ made, for the same
        # reason. Holding the whole SqliteVehicle leaves `set_signal` one attribute access away
        # from any consumer, which is a second write path to the car with none of the
        # availability, precondition, physical-limit or operation-log checks that only
        # executor.execute performs. Holding the whole SceneContext is the same shape one store
        # over: `update` reachable from the hub is a second door into perception, and a belief
        # arriving through it carries no source and no provenance — the exact thing the
        # envelope exists to make impossible.
        #
        # Honest about the boundary: a bound method still carries `__self__`, so both stores
        # remain reachable by introspection. What is closed is the accidental route — reading
        # an attribute and finding something writable — not a determined one. Nothing short of
        # a proxy object closes the determined one, and a proxy is a second thing to keep
        # correct in exchange for stopping code that is already trying to misbehave.
        self._observation = perception.get
        self._live = perception.live
        self._read = car.get_signal
        # Task 3 adds `signal_age`. Absent means "no age information", which must read as
        # LIVE: a signal with no known age is not a stale signal, and treating it as one would
        # silence every rule the moment staleness landed.
        self._age = getattr(car, "signal_age", None)

    # --- perception ---------------------------------------------------------------------
    def observation(self, key: str, now: float) -> Optional["Observation"]:
        return self._observation(key, now)

    # --- the car ------------------------------------------------------------------------
    def signal(self, entity: str, attribute: str, now: float) -> Optional[Any]:
        """A stale signal reads exactly like one the car does not hold.

        One return value for both, so no rule has to learn a third case: a dead bus and a
        stationary car were indistinguishable before this, and the fix is not to make every
        condition ask two questions but to make the absent value mean what it says.
        """
        status, _ = self.signal_status(entity, attribute, now)
        return self._read(entity, attribute) if status == "live" else None

    def signal_status(self, entity: str, attribute: str, now: float) -> tuple[str, str]:
        """Why a signal reads as it does: ("live"|"stale"|"missing", detail).

        The detail is a whole sentence rather than a fragment, so the caller cannot phrase the
        same fact a second, different way. A rejection saying `not above 5.0` about a value
        frozen ten minutes ago is actively misleading — it reads as "the car is slow" when the
        truth is "the bus is quiet" — and the cure is that the world says why, once.

        The "missing" wording is the one `evaluate_explained` already used, so adopting this
        changes no message a rule was already producing.
        """
        if self._read(entity, attribute) is None:
            return "missing", f"{entity}/{attribute} is not a signal this car holds"
        age = self._age(entity, attribute, now) if self._age is not None else None
        max_age = _declared_max_age(entity, attribute)
        # Inclusive at the boundary, matching Observation.is_live: one notion of "still live"
        # across both stores, or the hub would answer the same question two ways.
        if age is None or max_age is None or age <= max_age:
            return "live", ""
        return "stale", f"{entity}/{attribute} is stale ({age:.1f}s > {max_age:.1f}s)"

    # --- both -------------------------------------------------------------------------
    def live_facts(self, now: float) -> dict:
        """Everything true right now, flattened to {name: value}.

        This is what the fallback's prompt is built from, which is why it must cover both
        stores: that prompt was built from perception alone, so "complex relationships between
        multiple context states" — named as a fallback job in the original scene-engine brief
        — has been quietly impossible whenever one of those states was the car's.

        Signals are named `entity/attribute`, not `entity.attribute`, so a vehicle signal can
        never be mistaken for an observation in the `vehicle.` namespace — the two stores are
        separated by a string prefix today, and a flattened dict is the one place that
        convention would actually collide.
        """
        facts = {key: obs.value for key, obs in self._live(now).items()}
        for entity, attribute in _declared_sensed():
            value = self.signal(entity, attribute, now)
            if value is not None:
                facts[f"{entity}/{attribute}"] = value
        return facts
