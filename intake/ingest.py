"""One door in, and the bus that keeps a measurement true.

Everything entering the system arrives here as an `Input` and is handed to the module that owns
the decision: an `Utterance` to the router, a `Percept` to the scene engine, a `SignalWrite` to
the car. Three doors became one, so provenance and timing are captured once at the edge instead
of three different ways -- or, for voice, not at all.

**This module holds no logic of its own, and that is a standing obligation rather than a
description.** Dispatch and provenance only: no threshold, no rule, no phrasing, no decision
about what an input means. It is the composition root, so everything is reachable from it, which
is exactly why it is the natural place for the next small convenience to land -- and the design's
§12 names "intake becomes the god object" as the open risk for that reason. The test of any
addition here is whether some other module already owns the question. If one does, it belongs
there, and being able to reach it from this file is not an argument.

**The bus is pumped, not threaded, and this is forced rather than chosen.** `sim/vehicle.py`
opens SQLite with default thread affinity, so a background republish thread would hit a
connection created on another thread -- and it would not raise anywhere anyone would see it:
`ui/state.py` wraps each pane defensively, so the page would render normally around a Vehicle
pane that had quietly become empty. It would lie rather than break, and a simulator that lies
about the car is worse than one that stops. So `pump(now)` is called by the loops that already
exist: the CLI on each command, the UI on each poll.

The semantics that fall out are the right ones anyway. A live bus is fresh whenever you look at
it, which is true of a real bus too; stopping it is what makes a value stale, not the passage of
time; and a clock offset cannot manufacture staleness while the bus is running, because the pump
stamps on the clock it is handed.

**The pump is only as good as its callers**, and that is the footgun this file ships with. A
consumer that forgets to pump sees every sensed signal age past its declared limit and read as
absent, so every rule conditioned on one falls silent with no error anywhere. Better than the
reverse -- a wrong action is worse than no action -- but silence is a bad way to learn about a
missing call, so a new loop that reads signals must add itself to the list above.
"""
from __future__ import annotations
from typing import Any

from scene.context import Observation

from .envelope import Input, Percept, SignalWrite, Utterance
from .sources import SOURCES


class Intake:
    def __init__(self, pipeline, engine, car, world):
        """The four things that between them make a whole system, assembled in one place.

        Before this the only assembly of router + scene engine + car was `cli/session.py`, and
        `cli/` is deliberately not packaged -- so a real integration would have had to
        reimplement wiring the dev tool had already worked out. This is that wiring, shipped.

        `world` is held rather than used: it is the read side of the same composition, so a
        consumer holding the door also holds the view and never builds a second one. What it
        buys immediately is the check below.
        """
        self.pipeline = pipeline
        self.engine = engine
        self.car = car
        self.world = world
        # A door and an engine over different worlds is the silent failure `SceneEngine.reads`
        # exists to catch, one level up: writes land where nothing reads, every rule sees an
        # empty world, nothing raises and the system is merely quiet. Duck-typed -- a test may
        # pass any object that observes -- so it only fires when the engine can answer.
        engine_world = getattr(engine, "world", None)
        if engine_world is not None and engine_world is not world:
            raise ValueError("the intake and the engine must read the same world")
        # source -> {(entity, attribute): value}. Held per source because publishing is a
        # property of the source, and one bus stopping must not silence another.
        self._held: dict[str, dict[tuple, Any]] = {}
        # Seeded from the declaration and mutable afterwards: `publishes` says what a source is
        # capable of, this says what it is doing. Conflating them would make /bus off a lie
        # about the registry rather than a fact about this run.
        self._publishing = {name: src.publishes for name, src in SOURCES.items()}
        # The lookup IS the dispatch, in the shape ui/actions.py uses for the same reason: a
        # payload type absent from this table has no route into the system at all, so a new one
        # cannot reach a handler without appearing here.
        self._handlers = {Utterance: self._route,
                          Percept: self._observe,
                          SignalWrite: self._write}

    # --- the door -----------------------------------------------------------------------
    def ingest(self, item: Input) -> Any:
        """One input in, whatever the owning module returned back out.

        Deliberately NOT wrapped in a common result type. A `RouteResult`, a `SceneOutcome` and
        a written value answer three different questions, and flattening them would mean this
        file inventing a vocabulary for outcomes it does not produce -- which is precisely the
        logic it is not allowed to hold. The caller already knows what it sent.

        Nothing here re-validates the source: an `Input` that exists is one that could have
        happened, checked at construction, so an undeclared source never reaches this method.
        """
        handler = self._handlers.get(type(item.payload))
        if handler is None:
            # The one place in this system where silence is not the safe default. Everything
            # else degrades quietly because the alternative is a wrong action; here there is no
            # action to get wrong, only an input dropped with nothing said about it.
            raise TypeError(f"no handler for {type(item.payload).__name__}")
        return handler(item)

    def _route(self, item: Input):
        return self.pipeline.route(item.payload.text)

    def _observe(self, item: Input):
        # Where provenance actually lands. The Observation's `source` and `at` come from the
        # envelope, not from a default the caller forgot to override -- which is how every
        # observation, including the outside. and vehicle. ones, used to end up claiming to be
        # the cabin camera.
        p = item.payload
        return self.engine.observe(
            Observation(p.key, p.value, p.confidence, item.source, item.at, p.ttl), item.at)

    def _write(self, item: Input):
        p = item.payload
        # `at` from the envelope, never `time.time()`: see the pump below and set_signal's own
        # docstring. The write and the re-stamp must agree about which clock they are on.
        self.car.set_signal(p.entity, p.attribute, p.value, at=item.at)
        if SOURCES[item.source].publishes:
            # Held on what the source CAN do, not on what it is doing. A value written while
            # the bus is stopped is still the last thing that source said, so starting the bus
            # republishes it -- otherwise /bus on would do nothing until someone happened to
            # write again, and the toggle would look broken in the case it exists for.
            self._held.setdefault(item.source, {})[(p.entity, p.attribute)] = p.value
        return p.value

    # --- the bus ------------------------------------------------------------------------
    def pump(self, now: float) -> int:
        """Re-stamp every held value for a source that is currently publishing.

        `now` is the caller's clock and is written as given. The session's clock carries an
        offset and the car stamps `time.time()`, so a pump reading a clock of its own would
        publish values that are already `offset` seconds old -- stale on arrival, with nothing
        raising. Returns how many values were re-stamped, so a caller can tell "the bus is
        stopped" from "the bus is running and there is nothing on it".
        """
        written = 0
        for source, held in self._held.items():
            if not self._publishing.get(source):
                continue
            for (entity, attribute), value in held.items():
                self.car.set_signal(entity, attribute, value, at=now)
                written += 1
        return written

    def set_publishing(self, source: str, on: bool) -> None:
        """Start or stop one source's publisher.

        Refuses a source that cannot publish rather than accepting it and doing nothing. Only a
        continuous measurement has anything to re-stamp -- an utterance is an event, not a level
        -- so switching the microphone "on" could only ever report a bus that was not running.
        """
        src = SOURCES.get(source)
        if src is None:
            raise ValueError(f"{source!r} is not a declared source")
        if on and not src.publishes:
            raise ValueError(f"{source!r} does not publish -- it produces "
                             f"{src.accepts.__name__}, which is an event, not a level")
        self._publishing[source] = bool(on)

    def publishing(self, source: str) -> bool:
        return bool(self._publishing.get(source))

    def forget(self) -> None:
        """Drop every held value. For when the car underneath is replaced.

        The publishing state deliberately survives: whether the bus is running is a setting on
        this instrument, in the same category as /llm and /gate, while a held value is a
        measurement of a specific vehicle. Republishing 45 kph into a freshly seeded car would
        put a reading from a vehicle that no longer exists into one that never moved -- and it
        would arrive looking perfectly live.
        """
        self._held.clear()
