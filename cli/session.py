"""The interactive session's logic: one utterance in, one structured Turn out.

Deliberately free of I/O so it can be tested. `cli/render.py` turns a Turn into text and
`cli/__main__.py` does the reading and printing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

import json
import time as _time

from t2f.build import build_pipeline
from t2f.cards import load_catalog, load_ood_prototypes
from t2f.config import Config
from t2f.gate import Thresholds
from t2f.types import LLMResult, ToolCall
from intake.envelope import Input, Percept, SignalWrite, Utterance
from intake.hub import WorldView
from intake.ingest import Intake, encode_payload
from intake.store import Store
from scene.consent import Answer
from scene.context import SceneContext
from scene.engine import SceneEngine
from sim.executor import SqliteExecutor
from sim.mapping import resolve_writes
from sim.seed import seed_from_catalog, sensed_signals
from sim.vehicle import SqliteVehicle

PERMISSIVE = Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)

# Which declared source this session speaks for, per payload type. Named once, here, rather
# than spelled at each call: `Input` refuses a source that does not produce what is being sent,
# so a typo would surface as a ValueError at the door — but three literals scattered through
# the file are three places to keep in step with the registry, and the registry is the thing
# that is allowed to change.
#
# `/signal` claims to be the bus because that is what it stands in for: the world moving, not
# an instruction to the car. Sending it under any other name would put a measurement into the
# system with provenance that is not true.
UTTERANCE_SOURCE = "mic"
# Which camera saw it, from the namespace the key names. A single default made every
# `outside.*` observation claim the cabin camera saw the road -- all six front_object rows in
# the scene gold say exactly that -- which is the decorative `source` that intake/sources.py
# was declared to end. An unknown namespace falls to the cabin because that is where an
# unprefixed key already lands.
PERCEPT_SOURCE = "cabin_cam"
_PERCEPT_SOURCE_BY_NAMESPACE = {"inside": "cabin_cam", "outside": "front_cam"}


def percept_source(key: str) -> str:
    """The declared source a key of this namespace comes from."""
    namespace, _, _ = key.partition(".")
    return _PERCEPT_SOURCE_BY_NAMESPACE.get(namespace, PERCEPT_SOURCE)
SIGNAL_SOURCE = "can0"


@dataclass
class Delta:
    entity: str
    attribute: str
    before: Any
    after: Any


@dataclass
class SpanOutcome:
    """What became of one action span."""
    clause: str
    function: Optional[str]
    parameters: dict
    band: str
    escalated: bool                      # MEDIUM, handed to the model
    outcome: str                         # executed | refused | rejected | asked | unresolved
    detail: str = ""                     # the cause, when there is one
    deltas: list = field(default_factory=list)
    writes_signals: bool = True          # False when the function addresses no signal at all


@dataclass
class ContextRow:
    """One live observation, as a display needs it rather than as the engine stores it.

    `age` and `expires_in` are derived against the session's clock at the moment of the
    read, so a row is a photograph and never a live handle to an Observation.
    """
    key: str
    value: Any
    confidence: float
    source: str
    age: float
    expires_in: float


@dataclass
class Turn:
    utterance: str
    spans: list = field(default_factory=list)
    reply: str = ""
    error: Optional[str] = None
    # Last and defaulted: every existing construction of a Turn, in code and in tests, must
    # keep meaning what it meant. A scene turn has no spans — nobody said anything.
    scene: str = ""
    deltas: list = field(default_factory=list)
    # The engine's own RuleReports, carried through unwrapped. A parallel CLI-side type would
    # have to be kept in sync with a dataclass that already has exactly these four fields, and
    # the two would drift the first time the engine learned to explain something new.
    rules: list = field(default_factory=list)
    fallback: str = ""      # what the constrained fallback did, or why it was skipped


class Session:
    def __init__(self, pipeline, car, executor, cards, config, *, fake, llm, gate):
        self.pipeline, self.car, self.executor = pipeline, car, executor
        self.cards, self.config = cards, config
        self.fake, self.llm, self.gate = fake, llm, gate
        # Overwritten by `build`. Defaulted here so anything reading it off a session
        # constructed directly — the display, mode_label — gets the shipped answer rather than
        # an AttributeError.
        self.raw_capture = True
        # Everything the scene engine decides is a function of `now`: a TTL, a cooldown, a
        # persistence window. At a terminal those are all measured in minutes, so without a
        # lie to tell about the clock most of the engine is unreachable by hand.
        self.clock_offset = 0.0

    # --- construction ------------------------------------------------------------------
    @classmethod
    def build(cls, *, fake=False, llm=True, gate="shipped", db=":memory:",
              catalog="data/catalog", config_path="config.yaml", scene_llm=None,
              raw_capture=True):
        cards = load_catalog(catalog)
        config = Config.default() if fake else Config.load(config_path)
        embedder = cls._embedder(config, fake)
        client = cls._llm(config, fake) if llm else None
        car = SqliteVehicle(db)
        car.init_schema()
        seed_from_catalog(car, cards)
        executor = SqliteExecutor(car, {c.name: c for c in cards})
        ood = load_ood_prototypes(config.ood_prototypes) if (llm and not fake) else None
        pipe = build_pipeline(cards, embedder, config, llm_client=client, executor=executor,
                              ood_texts=ood,
                              thresholds=PERMISSIVE if gate == "permissive" else None)
        session = cls(pipe, car, executor, cards, config, fake=fake, llm=llm, gate=gate)
        # The car and the executor are shared, not copied. A scene reasoning about its own
        # vehicle could not see what the driver's own commands did, and would ask to open a
        # lock that is already open — and its consent would write to a car nobody is driving.
        #
        # Perception is built HERE rather than inside the engine, because the world is a view
        # over it and has to exist first. Both objects are over the one store, which the engine
        # checks: a second SceneContext would be written by the engine and read by nobody.
        #
        # The car's own connection, not a second one to the same file: two connections are two
        # opinions about a transaction, and the car already owns the only one. It also decides
        # what `--db` now means — a persisted car is a file that remembers what the cameras
        # said, which is why retention and the privacy switch are part of this work rather than
        # a later thought.
        #
        # `raw_capture` is carried here rather than read from a config, because it is the one
        # decision in this constructor that is about people rather than about the machine: with
        # it off nothing this session hears is written down verbatim — not the payload, not the
        # transcript, not the clause a decision names. Everything else is unchanged, which is
        # what makes it a switch and not a mode.
        store = Store(car.conn, raw_capture=raw_capture)
        session.raw_capture = bool(raw_capture)
        perception = SceneContext(store)
        world = WorldView(perception, car)
        session.scene = SceneEngine(cards_by_name={c.name: c for c in cards},
                                    world=world, executor=executor,
                                    llm=scene_llm, perception=perception)
        # The session stops being the composition root and becomes a consumer of one. It used
        # to be the only thing that assembled router + scene engine + car, which is why it grew
        # `route`, `observe` and `set_signal` as three unrelated methods with three different
        # disciplines — and `cli/` is deliberately not packaged, so a real integration would
        # have had to reimplement that wiring. It is `intake`'s now; the three methods below
        # are envelope builders over one door.
        # The same store the engine writes beliefs into, handed over rather than left to the
        # default. The door would build an identical one from `car.conn`, so this is not
        # correctness — it is the composition root saying out loud that there is one store,
        # the way it already says there is one car and one world.
        session.intake = Intake(pipe, session.scene, car, world, store=store)
        session._seeded = session._snapshot()      # baseline for /car
        session._hold_sensed()
        return session

    @staticmethod
    def _embedder(config, fake):
        from t2f.embed import FakeEmbedder
        if fake:
            return FakeEmbedder(256)
        from t2f.embed import TransformersEmbedder
        return TransformersEmbedder(config.model_id, mrl_dim=config.mrl_dim)

    @staticmethod
    def _llm(config, fake):
        from t2f.llm.client import FakeLLMClient, TransformersXGrammarClient
        if fake:
            return FakeLLMClient(default=LLMResult(tool_call=ToolCall("noop", {})))
        return TransformersXGrammarClient(model_id=config.llm.get("model_id", "Qwen/Qwen3-0.6B"),
                                          max_new_tokens=config.llm.get("max_new_tokens", 128))

    def rebuild(self, *, llm=None, gate=None):
        """Switch a mode, keeping the car exactly as it is — the point of the switch is to
        type the same words twice against the same vehicle."""
        self.llm = self.llm if llm is None else llm
        self.gate = self.gate if gate is None else gate
        client = self._llm(self.config, self.fake) if self.llm else None
        ood = (load_ood_prototypes(self.config.ood_prototypes)
               if (self.llm and not self.fake) else None)
        self.pipeline = build_pipeline(
            self.cards, self.pipeline.embedder, self.config, llm_client=client,
            executor=self.executor, ood_texts=ood,
            thresholds=PERMISSIVE if self.gate == "permissive" else None)
        # The door routes through whatever pipeline it holds, so the swap has to reach it:
        # otherwise /llm off would detach the model from the session while every utterance kept
        # going to the router that still had it. Rebound rather than rebuilt, because a fresh
        # Intake would drop the bus's held values and its publishing state — and those belong
        # to the car and to this session, not to the router configuration.
        self.intake.pipeline = self.pipeline
        # `self.scene` is deliberately untouched. A mode switch replaces the router, and the
        # point of the switch is to type the same words twice against the same car; a pending
        # question belongs to the conversation, not to the router configuration, so /llm off
        # between the question and the 好 must not swallow the answer.

    # --- the clock -----------------------------------------------------------------------
    def _now(self) -> float:
        """The only clock the session reads. Every call site goes through here.

        A missed clock read elsewhere would make the offset apply to some decisions and not
        others — an observation that expires while the cooldown that silenced it does not —
        which is worse than having no offset at all.

        **Wall clock, not monotonic, and the car is why.** `SqliteVehicle` stamps `updated_at`
        with `time.time()`, and the car can be persisted with `--db`, so a monotonic stamp
        would be written into a file that outlives the process that produced it and mean
        nothing on the next run. Once `signal_age` began reading those stamps, a session on
        monotonic subtracted ~756 thousand from ~1.79 billion and got an age of roughly minus
        1.78 billion seconds — which is under every max_age, so every signal read LIVE and
        staleness silently never fired. It failed open, in the one direction that leaves a
        discipline looking wired while doing nothing.

        The cost is that a system clock jump moves ages with it. For an instrument that is
        irrelevant; for a vehicle it would not be, and a real integration should pass its own
        clock rather than inherit this one.
        """
        return _time.time() + self.clock_offset

    def advance_clock(self, seconds: float) -> float:
        """Move the session's clock forward (or back), returning the new total offset.

        Nothing sleeps and nothing is re-evaluated: the offset is read the next time anything
        asks what time it is, exactly as a real elapsed interval would be.
        """
        self.clock_offset += seconds
        return self.clock_offset

    # --- the bus -------------------------------------------------------------------------
    def pump(self, now: Optional[float] = None) -> int:
        """Re-stamp the running bus, on THIS session's clock. Returns how many values moved.

        Called by every path that decides something, so a live bus is fresh whenever anything
        looks at it. `now` is passed in where the caller already has one, so the re-stamp and
        the read that follows it are the same instant rather than two microseconds apart — an
        irrelevant difference today, and exactly the kind that becomes a flaky test later.

        The clock is the session's, not the machine's, which is the whole reason `set_signal`
        grew an `at`: the car stamps `time.time()` and this session reads at `time.time() +
        offset`, so a pump on the wall clock would publish values that are already `offset`
        seconds old. `/clock +5` would then make every pumped signal stale on arrival — the
        bus wired, running, and silencing every rule.
        """
        return self.intake.pump(self._now() if now is None else now)

    def _hold_sensed(self) -> None:
        """Put the seeded values on the bus, so a live bus is live from the first moment.

        Without this nothing is held until someone types /signal, and the car's resting speed
        keeps whatever stamp `seed_from_catalog` gave it — so two seconds into any session the
        animal rule would start reporting `speed_kph is stale` with the bus running, which is
        precisely the state that is supposed to be fresh whenever you look. A real bus
        publishes from power-on; this is that.

        Read back from the car rather than from the declaration's resting value, because the
        car is what was actually seeded and the two must not be able to disagree.
        """
        now = self._now()
        for entity, attribute, *_rest in sensed_signals():
            value = self.car.get_signal(entity, attribute)
            if value is None:
                continue                  # declared but unseeded: nothing to publish yet
            self.intake.ingest(Input(source=SIGNAL_SOURCE, at=now,
                                     payload=SignalWrite(entity, attribute, value)))

    def set_bus(self, on: bool) -> bool:
        """Start or stop the publisher. Returns the state it is now in.

        Stopping it is the only way a sensed signal goes stale, which is the point: a value
        does not decay because time passed, it decays because nothing said it again.
        """
        self.intake.set_publishing(SIGNAL_SOURCE, on)
        return on

    def bus_publishing(self) -> bool:
        return self.intake.publishing(SIGNAL_SOURCE)

    def bus_note(self) -> str:
        """One phrase saying whether what was just set will stay true.

        Shown on the line that sets a signal because "45 kph" and "45 kph as of some moment in
        the past" are different facts, and only one of them is what a rule will read. A control
        that silently sets a value which is about to expire is the same defect as `ttl=30` being
        accepted and ignored.
        """
        return ("publishing · re-stamped on every command" if self.bus_publishing()
                else "bus off · this value ages from now on")

    # --- one turn ----------------------------------------------------------------------
    OBSERVATION_TTL = 300.0

    def observe(self, key: str, value, confidence: float = 0.9,
                source: Optional[str] = None, ttl: Optional[float] = None) -> Turn:
        """One perception event in, one Turn out — the scene analogue of handle().

        Validated here rather than in each caller, because the terminal and the browser are
        two doors onto one session and a rule that held at only one of them is not a rule.

        The confidence range is the load-bearing check. Every rule's `floor` and `threshold`
        live in [0, 1], so an observation at 7.0 clears any band trivially — the instrument
        would report a confident MATCH built on a number the design has no meaning for, which
        is worse than refusing the input.
        """
        if not str(key).strip():
            raise ValueError("an observation needs a key")
        if str(value).strip() == "":
            raise ValueError("an observation needs a value")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
        if ttl is not None and float(ttl) <= 0:
            raise ValueError(f"ttl must be positive, got {ttl}")
        now = self._now()
        self.pump(now)
        # The design namespaces keys inside. / outside. / vehicle., and prefixing every key
        # unconditionally made two of those three unreachable — there was no way to state an
        # outside.weather observation at all. A key that already names its namespace is taken
        # as written; a bare one still gets the cabin, so /scene rear_occupant=child works.
        full_key = key if "." in key else f"inside.{key}"
        before = self._snapshot()
        # An envelope, not an Observation: the source and the time are the envelope's to carry,
        # and `intake` is what turns them into perception's own shape. This method's job is now
        # the validation above and nothing else.
        outcome = self.intake.ingest(
            Input(source=source or percept_source(full_key), at=now,
                  payload=Percept(full_key, value, confidence,
                                  self.OBSERVATION_TTL if ttl is None else ttl)))
        return Turn(utterance=f"[scene] {key}={value}", reply=outcome.speech,
                    scene=outcome.scene or "—", deltas=self._deltas_since(before),
                    rules=list(self.scene.explain()), fallback=self.scene.fallback_note())

    def handle(self, utterance: str) -> Turn:
        now = self._now()
        self.pump(now)
        before = self._snapshot()
        try:
            # Taken before `resolve`, because `resolve` is a path that ACTUATES: a watermark
            # read afterwards sits above the operations it is meant to claim and attributes
            # nothing. Discarded when the words turn out not to be an answer — `Intake._route`
            # takes its own for the routing turn that follows.
            since = self.intake.store.operations_watermark()
            # Consent stays here, ahead of the door, and §11 says why it is deferred: a pending
            # question is not a fact about the world, and putting it in intake is how intake
            # becomes the thing every module imports. 好 is only an answer because this session
            # is holding a question — nothing in the envelope can know that.
            consent = self.scene.resolve(utterance, now)
            if consent.answered:
                # The driver was answering the car, not commanding it. Routing these words
                # would treat 好 as an utterance to match against 92 functions.
                self._record_consent(utterance, consent, now, since)
                return Turn(utterance=utterance, reply=consent.speech, scene="consent",
                            deltas=self._deltas_since(before))
            result = self.intake.ingest(
                Input(source=UTTERANCE_SOURCE, at=now, payload=Utterance(utterance)))
        except Exception as exc:                      # a crash costs a 60s reload; survive it
            return Turn(utterance=utterance, error=f"{type(exc).__name__}: {exc}")
        deltas = self._deltas_since(before)
        return Turn(utterance=utterance, reply=result.reply,
                    spans=[self._span(cl, deltas) for cl in result.clauses])

    def _record_consent(self, utterance: str, consent, now: float, since: int) -> None:
        """Write the consent turn down, because nothing else will.

        **This is the only path to the car that does not go through the door**, so it is the
        only one whose record the door cannot write. Without these five statements an action
        the driver authorised lands in `operation_log` with a NULL `turn_id` — an execution
        with nothing in the store saying who asked for it, which is precisely what
        `tests/intake/test_completeness.py` exists to make impossible. It was NULL until this
        method existed; the negative test there is what keeps it filled.

        Written here rather than moved into `intake` because this is where the decision is
        made, and the reason it is made here has not changed: 好 is only an answer because this
        session is holding a question, and nothing in an envelope can know that. **The residual
        gap is worth stating rather than leaving to be discovered** — `cli/` is not packaged,
        so a consumer that assembled its own composition and called `SceneEngine.resolve`
        itself would get no record. There is no such consumer: `resolve` has exactly one caller
        in the repo, above. A second one is the thing to notice.

        Opened AFTER the work, which is the opposite of `_route` and needs its reason said out
        loud: `SceneEngine.resolve` is documented never to raise — it catches everything and
        returns a failure result — so there is no mid-way death for an early row to survive.
        What genuinely has to happen first is the watermark, and it does.
        """
        store = self.intake.store
        # One raw row, and only now that the words have turned out to be an answer. An
        # utterance that is NOT an answer goes on to the door, which writes its own raw row —
        # writing one here as well would put the same sentence in the record twice and read as
        # the driver having said it twice.
        raw_id = store.put_raw(UTTERANCE_SOURCE, now, encode_payload(Utterance(utterance)))
        turn_id = store.open_turn(raw_id, now, "consent")
        store.put_decision(
            turn_id,
            # Through `spoken`, because 好 IS speech: the privacy switch has to reach the answer
            # the same way it reaches a routed clause, or it keeps the words it claims to drop.
            store.spoken(utterance) or "",
            # The classification, not the outcome. "yes, and the car refused" and "no" are
            # different facts and both end up answered-but-not-executed; a verdict derived from
            # `executed` alone would file the first as the second. "" only when `resolve` caught
            # an infrastructure fault before it classified anything.
            consent.answer or "unclassified",
            consent.tool_call.name if consent.tool_call else None,
            # Why nothing happened, when nothing did. Empty for the two cases that need no
            # explanation: it worked, or the driver said no.
            "" if consent.executed or consent.answer == Answer.NO.value else consent.speech)
        store.close_turn(turn_id, consent.speech, since_operation=since)
        # Marked processed for the reason `ingest` marks its own: this row was handled
        # synchronously, and one left pending is one `process_pending` would run a second time.
        store.mark_processed(raw_id, now)
        # One commit per input, the boundary `Intake.ingest` uses. Half a consent turn in the
        # record is a worse artifact than none.
        store.commit()

    def _deltas_since(self, before: dict) -> list:
        return [Delta(e, a, before.get((e, a)), v) for (e, a), v in self._snapshot().items()
                if before.get((e, a)) != v]

    def _span(self, clause_result, deltas) -> SpanOutcome:
        tc = clause_result.tool_call
        if clause_result.response:
            outcome, detail = "executed", ""
        elif clause_result.exec_error:
            outcome, detail = "refused", clause_result.exec_error.message
        elif any(e.code != "missing_required" for e in clause_result.validation_errors):
            # A missing parameter is answerable, so it is a QUESTION, not a stated rejection —
            # the same call t2f/reply.py makes, and the session must agree with what the driver
            # hears. Filtering on the code rather than reordering the branches is deliberate: a
            # detailed error (out_of_range) can co-occur with a clarification, and that detail
            # must still be shown.
            outcome = "rejected"
            detail = next((e.detail for e in clause_result.validation_errors if e.detail), "")
        elif clause_result.clarification:
            outcome, detail = "asked", clause_result.clarification.question
        else:
            outcome, detail = "unresolved", ""
        return SpanOutcome(
            clause=clause_result.clause,
            function=tc.name if tc else clause_result.decision.chosen,
            parameters=dict(tc.parameters) if tc else {},
            band=clause_result.decision.band.value,
            # needs_llm is set by NullMediumResolver too, which has no model to hand the span
            # to — so "escalated" must mean a model actually saw it, not that one was wanted.
            escalated=clause_result.needs_llm and self.llm,
            outcome=outcome, detail=detail,
            deltas=self._own_deltas(clause_result, deltas) if outcome == "executed" else [],
            writes_signals=self._writes_signals(clause_result))

    def _writes_signals(self, clause_result) -> bool:
        """Whether this call addresses any signal at all.

        Distinguishes "the A/C was already on, so nothing moved" from "this function has no
        state to move" — 打开空调 against an already-on A/C is a success with no delta, and
        rendering that as "no signal for this function" would be a lie.
        """
        tc = clause_result.tool_call
        card = {c.name: c for c in self.cards}.get(tc.name) if tc else None
        return bool(card is not None and resolve_writes(card, tc))

    def _own_deltas(self, clause_result, deltas) -> list:
        """Only the signals THIS call wrote.

        `handle` diffs the whole car across one route(), so a multi-intent utterance would
        otherwise show every change under every action. The mapping knows which signals a
        given call addresses, so each span claims only its own.
        """
        tc = clause_result.tool_call
        card = {c.name: c for c in self.cards}.get(tc.name) if tc else None
        if card is None:
            return deltas
        mine = {(e, a) for e, a, _ in resolve_writes(card, tc)}
        # No `or deltas` fallback: a span that changed nothing must not claim another span's
        # change. "Nothing moved" and "moved something" are different facts.
        return [d for d in deltas if (d.entity, d.attribute) in mine]

    def _snapshot(self) -> dict:
        """Values are JSON-encoded in the column; decode them, or a boolean reads as
        'false' -> 'true' and an enum arrives wearing its quotes."""
        rows = self.car.conn.execute("SELECT entity, attribute, value FROM signal").fetchall()
        return {(r["entity"], r["attribute"]): json.loads(r["value"]) for r in rows}

    # --- session commands ---------------------------------------------------------------
    def changed_signals(self) -> list[tuple]:
        """(entity, attribute, value) for signals that differ from the seeded car.

        The baseline is captured at construction, so `/car` does not have to build a second
        vehicle to know what "unchanged" looks like.
        """
        return [(e, a, v) for (e, a), v in self._snapshot().items()
                if self._seeded.get((e, a)) != v]

    def context_rows(self) -> list[ContextRow]:
        """What perception currently believes — the `/car` of the scene subsystem.

        Only live observations: an expired one is not part of what the engine is reasoning
        over, so showing it would explain a decision by a belief that was not held.
        """
        now = self._now()
        return sorted(
            (ContextRow(key=o.key, value=o.value, confidence=o.confidence, source=o.source,
                        age=now - o.at, expires_in=o.at + o.ttl - now)
             for o in self.scene.context.live(now).values()),
            key=lambda r: r.key)

    # How many turns a hand tool shows by default. Small on purpose: this is the answer to
    # "why did it just do that", and a screen of history is a worse answer than four turns.
    STORE_TURNS = 6

    def recent_turns(self, limit: int = STORE_TURNS) -> list[dict]:
        """What the store recorded, newest first — heard, decided, did, said.

        The whole of `/store` and of the browser's record pane, in one place, because the two
        doors must not be able to show different histories of one session.

        This is where the two halves of the record are joined. `intake.store` owns the turn and
        its decisions; `sim` owns `operation_log` and therefore what the car actually did. Both
        are asked here, by the object that already holds both, so neither module has to reach
        into the other's tables. See `Store.recent_turns` and `SqliteVehicle.operations_for_turn`.

        Plain dicts rather than a dataclass, unlike `ContextRow`: every field goes to the page
        as JSON unchanged, and a display type would have to be flattened again on the way out.
        Nothing here derives anything from the session's clock either — these are rows as
        written, and the stamps are the stamps.
        """
        turns = self.intake.store.recent_turns(limit)
        for turn in turns:
            turn["operations"] = [
                {"function": op["function"], "outcome": op["outcome"],
                 "error": op["error"], "detail": op["detail"]}
                for op in self.car.operations_for_turn(turn["id"])]
        return turns

    # --- the world, not the car ----------------------------------------------------------
    def set_signal(self, entity: str, attribute: str, value):
        """Tell the simulator what the car is DOING. Returns the number written.

        A simulator control, not a Central Model action, which is why it is here and not
        behind the executor: the camera seeing a child, /reset re-seeding the vehicle and the
        car now doing 45 are the same category of event — the world changing around a system
        that did not cause it. `executor.execute` is the seam for what the model performs and
        stays the only one.

        Anything not declared sensed is REFUSED, and that refusal is the whole point. A signal
        some function writes, poked directly here, would land in the car without the
        availability check, the precondition and the physical limit that `SqliteExecutor`
        exists to apply — so /signal would be a way to command the car that answers to none of
        the rules commanding it answers to. Checked before the value is even read, so an
        actuated signal is refused whatever you were trying to set it to.
        """
        declared = {(e, a): (lo, hi)
                    for e, a, _resting, _unit, lo, hi, _max_age in sensed_signals()}
        if (entity, attribute) not in declared:
            known = ", ".join(f"{e}/{a}" for e, a in sorted(declared))
            raise ValueError(
                f"{entity}/{attribute} is not a sensed signal — the car senses {known}, and "
                f"everything else it holds is written by a function, not set by hand")
        try:
            number = float(value)
        except (TypeError, ValueError):
            # Raised as a ValueError with the address in it, never a bare float() complaint:
            # both doors print this at a human, and "could not convert string to float" does
            # not say which signal was being set.
            raise ValueError(f"{entity}/{attribute} takes a number, got {value!r}") from None
        lo, hi = declared[(entity, attribute)]
        if lo is not None and hi is not None and not lo <= number <= hi:
            raise ValueError(f"{entity}/{attribute} is {lo}–{hi}, got {number}")
        # Limits are not repeated on the write: `SqliteVehicle.set_signal` keeps the row's
        # existing min/max on conflict, and they were seeded from this same declaration.
        #
        # Through the door rather than straight at the car, which is what puts the value on the
        # bus: a signal set by hand is the last thing the bus said, so it keeps being said
        # until the bus stops. It is also what stamps it on the SESSION's clock — set after
        # /clock +100, a value written at wall time would have been a hundred seconds old the
        # moment it landed.
        self.intake.ingest(Input(source=SIGNAL_SOURCE, at=self._now(),
                                 payload=SignalWrite(entity, attribute, number)))
        return number

    def sensed_rows(self) -> list[dict]:
        """Every declared sensed signal, with its live value and how old that value is — the
        `/car` of the world.

        Always all of them, never only what moved: a speed resting at 0.0 is the answer to
        "why did the animal rule say nothing", and a pane that hides it cannot give it.

        The value comes from the car; the unit and the limits come from the declaration,
        because those limits are the ones `set_signal` above enforces — a control bounded by
        anything else would offer values this session refuses.

        `stale` comes from the WORLD, not from a comparison made here. The hub already owns
        "is this reading still true", every rule reads it through `signal_status`, and a second
        copy of that comparison in this method would be two beliefs about one bus — with the
        display holding the one nothing consults. A pane showing 45 as a value while every rule
        reads it as absent is the exact lie the whole freshness discipline exists to remove.

        The value itself stays on a stale row rather than being blanked: it is still the last
        thing the bus said, it is what the control is bounded by, and it is what starting the
        bus again would republish. `stale` is the flag that says nothing reads it, and it is the
        display's job never to show one as the other.
        """
        now = self._now()
        world = self.intake.world
        rows = []
        for e, a, _resting, unit, lo, hi, _max_age in sensed_signals():
            status, _detail = world.signal_status(e, a, now)
            rows.append({"entity": e, "attribute": a, "value": self.car.get_signal(e, a),
                         "unit": unit, "min": lo, "max": hi,
                         # A number, or None for a signal the car does not hold. Formatted by
                         # whoever displays it: the terminal and the page phrase an age
                         # differently, and a string built here would be a third opinion.
                         "age": self.car.signal_age(e, a, now),
                         "stale": status == "stale"})
        return rows

    def attach_scene_llm(self, client) -> None:
        """Attach or detach the constrained fallback, keeping everything else.

        Deliberately one assignment and no rebuild — the same principle as /llm and /gate: the
        car, the accumulated context and any pending consent all survive, because the point of
        the switch is to replay the same perception against the same vehicle and see whether
        the second half of the system says anything the rules did not. Rebuilding the engine
        would reseed the context and swallow a question already asked.
        """
        self.scene.llm = client

    def reset(self):
        self.car.conn.execute("DELETE FROM signal")
        self.car.conn.execute("DELETE FROM operation_log")
        # `precondition` has no unique key and seed_from_catalog re-inserts every row, so a
        # session that resets twice would check the same precondition three times.
        self.car.conn.execute("DELETE FROM precondition")
        self.car.conn.commit()
        seed_from_catalog(self.car, self.cards)
        self._seeded = self._snapshot()
        # The bus has to forget too, and then start again from the new car. A held 45 kph is a
        # measurement of a vehicle that no longer exists, and the next pump would write it into
        # the freshly seeded one looking perfectly live — a car that reads as doing 45 while
        # /car reports it as it was seeded. Re-held immediately afterwards so the new car's own
        # resting values are what is being published.
        self.intake.forget()
        self._hold_sensed()
        # The scene engine has to forget too. A pending consent asked about the old car would
        # otherwise be answerable against the new one — 好 after a reset would re-open a lock
        # nobody was asked about — and a cooldown carried across would silence a rule for a
        # vehicle it never spoke to.
        self.scene.reset()

    def mode_label(self) -> str:
        # The scene fallback is named the way the eval arms name it, S / S_llm, and is stated
        # in both directions rather than only when attached: half the scene subsystem is
        # unreachable without it, so "why did nothing happen" needs the answer on the prompt.
        parts = ["C_llm" if self.llm else "C", self.gate,
                 "S_llm" if self.scene.llm is not None else "S"]
        if self.fake:
            parts.append("FAKE")
        if not self.raw_capture:
            # Only when OFF. The default is not worth a word on every prompt, and a session
            # that is NOT writing down what it hears is: someone reading the store afterwards
            # and finding no transcripts needs to know that was a setting and not a fault.
            parts.append("no-raw")
        return " · ".join(parts)
