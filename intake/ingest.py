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

**The door also writes down what came through it.** Every input lands in `observation_raw`
before anything is done with it, and what the owning module then decided lands in `turn` and
`decision`. That is recording, not logic: the bands come from `RouteResult.clauses` and the
verdicts from `SceneEngine.explain()`, both already computed by the module that owns the
question. Nothing here decides anything it did not decide before.

**And the same door can be reached by writing a row.** `process_pending(now)` takes rows a
producer wrote directly -- another process, another language, no binding into this runtime --
rebuilds an `Input` from each and sends it through the identical dispatch. Explicit, never a
timer, for the reason the pump is: this file owns no clock and no thread.
"""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Optional

from scene.context import Observation

from .envelope import Input, Percept, SignalWrite, Utterance
from .sources import SOURCES
from .store import Store


def encode_payload(payload) -> str:
    """A payload as the one TEXT column `observation_raw` gives it.

    JSON of the payload's own fields, and deliberately NO type tag beside them: the source
    already declares what it produces (`SOURCES[name].accepts`), so a tag would be a second
    statement about one fact -- the thing `envelope.py` refuses to have for the same reason.
    A row whose payload does not fit its source's type therefore cannot be decoded at all,
    which is the point: the decode IS the check that the row was well formed.

    `ensure_ascii=False` so the column holds 后排坐着一个小孩 and not a row of escapes. This
    store is meant to be read by hand.

    The non-dataclass branch is not dead code. `ingest` writes the raw row BEFORE it looks for
    a handler, so a payload type nobody wired up still leaves a record of having arrived --
    and it is exactly the case where a `repr` is all anyone can honestly write down.
    """
    if is_dataclass(payload):
        return json.dumps(asdict(payload), ensure_ascii=False)
    return repr(payload)


def input_from_row(row) -> Input:
    """One `observation_raw` row, back as the `Input` that produced it.

    The inverse of `encode_payload`, and the validation of the row in one step: the payload
    type comes from the source's declaration, `Input.__post_init__` re-checks that the two
    agree, and a row that cannot satisfy both raises here rather than reaching a handler.
    A malformed row is an error to record against that row, never a crash of the queue --
    which is what `process_pending` does with what this raises.
    """
    source = row["source"]
    src = SOURCES.get(source)
    if src is None:
        raise ValueError(f"{source!r} is not a declared source")
    fields = json.loads(row["payload"])
    if not isinstance(fields, dict):
        raise ValueError(f"row {row['id']}: payload is not an object")
    return Input(source=source, at=row["at"], payload=src.accepts(**fields))


@dataclass(frozen=True)
class Processed:
    """What became of one queued row.

    `ingest` deliberately returns the owning module's own result unwrapped, because the caller
    already knows what it sent. A queue caller does not: it sent nothing, the rows came from
    somewhere else, and a bare list of mixed results could not be matched back to the inputs
    that produced them. So the row id and the failure travel WITH the outcome -- which is not
    a vocabulary for outcomes, since `outcome` is still whatever the module returned.
    """
    raw_id: int
    outcome: Any = None
    error: str = ""            # "" unless this row raised; the row is processed either way


def _note(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _why(clause) -> str:
    """Every fault `t2f` already attached to this clause, written down verbatim.

    Concatenated in the order they could have happened -- validation, then the vehicle -- and
    deliberately NOT ranked. `cli/session.py::_span` already owns which fault a display leads
    with, and a second precedence here would be a second opinion about one clause that can
    only agree with the first by accident. The record takes all of them; whoever reads it
    decides which mattered.

    The band already says why a LOW clause did nothing, and `operation_log` already says why
    the car refused. What is here and nowhere else is the unusable parameter -- a clause that
    never reached the executor and so left no trace outside the sentence the driver heard.
    """
    faults = [e.message for e in clause.validation_errors]
    if clause.exec_error:
        faults.append(clause.exec_error.message)
    return "; ".join(faults)


class Intake:
    def __init__(self, pipeline, engine, car, world, store=None):
        """The four things that between them make a whole system, assembled in one place.

        Before this the only assembly of router + scene engine + car was `cli/session.py`, and
        `cli/` is deliberately not packaged -- so a real integration would have had to
        reimplement wiring the dev tool had already worked out. This is that wiring, shipped.

        `world` is held rather than used: it is the read side of the same composition, so a
        consumer holding the door also holds the view and never builds a second one. What it
        buys immediately is the check below.

        `store` is defaulted where `SceneEngine`'s `perception` is required, and the asymmetry
        is not laziness. The engine refuses to invent a store because the world is a view over
        one, so a store built in that constructor would be one no caller could read through --
        silent, and fatal. Here the door already holds the car and the car owns the only
        connection, so the store this line builds is the only store this composition could
        have. Passing one in says the same thing out loud, and `cli/session.py` does.
        """
        self.pipeline = pipeline
        self.engine = engine
        self.car = car
        self.world = world
        self.store = Store(car.conn) if store is None else store
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

        The raw row is written FIRST and unconditionally -- before the handler lookup, before
        any dispatch -- so "what arrived" is a fact independent of whether anything could be
        done with it. It is marked processed on the way out because it was: an input that came
        through this method has been handled synchronously, and leaving it pending would put it
        in the queue for `process_pending` to run a second time. For a command that reaches the
        car, twice is not a cosmetic difference.
        """
        raw_id = self.store.put_raw(item.source, item.at, encode_payload(item.payload))
        # Stamped `item.at`, not a clock of this method's own. The envelope already says when
        # this happened and the work is synchronous, so any other stamp is a second opinion
        # about one instant -- the same trap the pump's `now` exists to close.
        try:
            return self._dispatch(item, raw_id, item.at)
        finally:
            # One commit per input, in a `finally` so a handler that raises still leaves the
            # raw row and whatever got as far as being written. The store's writers do not
            # commit individually: SQLite fsyncs on commit, and a voice input touches six of
            # them, which measured 17.4 ms on a file database against 2.86 ms for one. An
            # input is the right boundary because an input is either recorded or it is not,
            # and half a turn in the record is a worse artifact than none.
            self.store.commit()

    def process_pending(self, now: float, limit: Optional[int] = None) -> list:
        """Run every row a producer wrote directly, oldest first. Returns one `Processed` each.

        The payoff of inputs being an interface: a vision process on another accelerator, a CAN
        reader in C++, an ASR service on its own schedule all integrate by writing a row, and
        this turns those rows into exactly the same dispatch a Python caller gets.

        **Explicit, never a timer.** The same discipline `pump` follows and for the same reason:
        SQLite here has thread affinity, so a worker loop would touch the connection from a
        thread that did not create it -- and `ui/state.py` wraps each pane defensively, so the
        page would render fine around a car that had quietly become empty. Called by the loop
        that owns the clock, and `now` is that loop's clock.

        A row that raises is marked processed WITH its error, and the loop continues. It must
        not block the queue behind it and it must not vanish: a dropped input is silence, and
        unexplained silence is the failure this system spends the most effort preventing.

        Rows are stamped `processed_at = now` rather than `at`, because those are genuinely two
        different facts here -- when the world produced the input, and when we got to it. The
        gap between them is the only measure of a queue falling behind.
        """
        # Here as well as in `pump`, because a deployment where producers write rows and
        # nothing re-stamps a bus -- vision and ASR only, no CAN -- would otherwise never pump
        # and never age anything out. Before the drain rather than after: a row already past
        # its window is not one to act on, and running it first would command the car about a
        # world that has moved on.
        self.store.apply_retention(now)
        done = []
        for row in self.store.pending(limit):
            raw_id = row["id"]
            try:
                item = input_from_row(row)
            except Exception as exc:
                # Never reached a handler, so nothing else will mark it. A row that cannot
                # become an `Input` is a malformed row -- a producer's bug -- and the honest
                # place to record that is against the row that carries it.
                self.store.mark_processed(raw_id, now, _note(exc))
                # Committed per row rather than once at the end: a batch that dies half way
                # must leave the rows it already handled marked, or the next call re-runs them
                # -- and for a command that reaches the car, twice is not cosmetic.
                self.store.commit()
                done.append(Processed(raw_id, error=_note(exc)))
                continue
            try:
                done.append(Processed(raw_id, self._dispatch(item, raw_id, now)))
            except Exception as exc:
                done.append(Processed(raw_id, error=_note(exc)))
            finally:
                # Per row, and in a `finally` so a row that raised still has its mark made
                # durable. A batch that dies half way must leave what it handled marked, or
                # the next call re-runs it -- and for a command that reaches the car, twice
                # is not a cosmetic difference.
                self.store.commit()
        return done

    def _dispatch(self, item: Input, raw_id: int, processed_at: float) -> Any:
        """The shared spine: hand the input to its owner, then close the raw row either way.

        Both entrances run through here so that a queued row and a direct call cannot drift
        into two behaviours -- which is the whole reason the simulator (Task 7) is allowed to
        write rows and trust they travel the real path.
        """
        try:
            handler = self._handlers.get(type(item.payload))
            if handler is None:
                # The one place in this system where silence is not the safe default. Everything
                # else degrades quietly because the alternative is a wrong action; here there is
                # no action to get wrong, only an input dropped with nothing said about it. It
                # is now also written down, on the raw row, in the `except` below.
                raise TypeError(f"no handler for {type(item.payload).__name__}")
            out = handler(item, raw_id)
        except Exception as exc:
            self.store.mark_processed(raw_id, processed_at, _note(exc))
            raise
        self.store.mark_processed(raw_id, processed_at)
        return out

    def _route(self, item: Input, raw_id: int):
        self.store.put_utterance(raw_id, item.at, item.payload.text)
        turn_id = self.store.open_turn(raw_id, item.at, "route")
        # Taken with the turn open and applied when it closes, so every operation the car
        # logged in between belongs to this turn. See `Store.operations_watermark`.
        since = self.store.operations_watermark()
        reply = ""
        try:
            result = self.pipeline.route(item.payload.text)
            reply = result.reply
            for clause in result.clauses:
                # Recorded, not computed. The band and the chosen function are what the gate
                # already decided; this writes them down against the clause they were decided
                # about.
                # The clause goes through `spoken` because it IS speech -- a slice of what the
                # driver said. Most utterances are one clause, so a privacy switch that blanked
                # the payload and left this would keep the sentence it claimed to drop.
                self.store.put_decision(turn_id, self.store.spoken(clause.clause) or "",
                                        clause.decision.band.value,
                                        clause.decision.chosen, _why(clause))
            return result
        finally:
            # Closed even when route() raised. A turn that actuated something and then failed
            # still has to own what it did, and `close_turn` is where that link is made. The
            # empty reply cannot be misread as deliberate silence: this turn points at a raw
            # row, and `_dispatch` writes the error onto that row as the exception passes it.
            self.store.close_turn(turn_id, reply, since_operation=since)

    def _observe(self, item: Input, raw_id: int):
        # Where provenance actually lands. The Observation's `source` and `at` come from the
        # envelope, not from a default the caller forgot to override -- which is how every
        # observation, including the outside. and vehicle. ones, used to end up claiming to be
        # the cabin camera.
        p = item.payload
        turn_id = self.store.open_turn(raw_id, item.at, "scene")
        since = self.store.operations_watermark()
        # The same trick one table across, and for the same reason -- see
        # `Store.perception_watermark`. Taken before the work, applied after it.
        beliefs = self.store.perception_watermark()
        outcome = None
        try:
            # The `perception` row is written from inside here, by SceneContext.update. Not
            # duplicated at this level: two accounts of one belief is exactly what the store
            # exists to stop being normal.
            #
            # The belief therefore lands with `raw_id` NULL -- an `Observation` has already
            # lost its envelope by the time the context sees it -- and the id is applied from
            # out here afterwards, off a watermark. Not through a parameter on
            # `SceneEngine.observe`, whose signature the design freezes, and not off an ambient
            # "current raw row": see `Store.perception_watermark` for why that one fails in the
            # wrong direction. Retention empties the column again when the frame is swept (ON
            # DELETE SET NULL in sim/schema.sql), which is the honest end state -- the belief
            # outlives the frame, and the link does not.
            outcome = self.engine.observe(
                Observation(p.key, p.value, p.confidence, item.source, item.at, p.ttl), item.at)
            for report in self.engine.explain():
                self.store.put_decision(
                    turn_id, report.rule_id, report.verdict,
                    # The scene id goes only on the rule that produced the outcome, so
                    # `chosen` answers "what was acted on" in this table the same way it does
                    # for a clause. On every row it would say every rule chose it.
                    outcome.scene if report.rule_id == outcome.scene else None,
                    report.reason, report.suppressed_by)
            return outcome
        finally:
            # Beside `close_turn` and in the same `finally`, because they answer the same
            # obligation: a frame that died mid-way still wrote a belief and still opened a
            # turn, and both have to point at the row that arrived or the record of the
            # failure is the one with no provenance on it.
            self.store.attribute_perception(raw_id, beliefs)
            self.store.close_turn(turn_id, outcome.speech if outcome else "",
                                  since_operation=since)

    def _write(self, item: Input, raw_id: int):
        p = item.payload
        # No turn, and that is the whole shape of this path: a signal write is the world
        # moving, so there is no decision to record and nothing was said. Its parsed form is
        # the `signal` row itself, overwritten in place, with the history of every frame that
        # arrived sitting in `observation_raw` behind it. A turn per frame would also put a
        # row on the bus at 10 Hz for a turn in which nothing happened.
        #
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
        # Retention rides on the pump because the pump is the one call every loop already
        # makes on its own clock. A second discipline -- "and also call sweep" -- is a second
        # thing a new loop can forget, and forgetting this one is silent in the worst
        # direction: nothing raises, the store simply keeps every word anyone said. At most
        # one pass a minute of the caller's clock; `Store.apply_retention` owns that.
        self.store.apply_retention(now)
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
