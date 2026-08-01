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
from scene.context import Observation
from scene.engine import SceneEngine
from scene.facts import VehicleFacts
from sim.executor import SqliteExecutor
from sim.mapping import resolve_writes
from sim.seed import seed_from_catalog, sensed_signals
from sim.vehicle import SqliteVehicle

PERMISSIVE = Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)


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
        # Everything the scene engine decides is a function of `now`: a TTL, a cooldown, a
        # persistence window. At a terminal those are all measured in minutes, so without a
        # lie to tell about the clock most of the engine is unreachable by hand.
        self.clock_offset = 0.0

    # --- construction ------------------------------------------------------------------
    @classmethod
    def build(cls, *, fake=False, llm=True, gate="shipped", db=":memory:",
              catalog="data/catalog", config_path="config.yaml", scene_llm=None):
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
        session.scene = SceneEngine(cards_by_name={c.name: c for c in cards},
                                    facts=VehicleFacts(car), executor=executor,
                                    llm=scene_llm)
        session._seeded = session._snapshot()      # baseline for /car
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

    # --- one turn ----------------------------------------------------------------------
    OBSERVATION_TTL = 300.0

    def observe(self, key: str, value, confidence: float = 0.9,
                source: str = "cabin_cam", ttl: Optional[float] = None) -> Turn:
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
        # The design namespaces keys inside. / outside. / vehicle., and prefixing every key
        # unconditionally made two of those three unreachable — there was no way to state an
        # outside.weather observation at all. A key that already names its namespace is taken
        # as written; a bare one still gets the cabin, so /scene rear_occupant=child works.
        full_key = key if "." in key else f"inside.{key}"
        obs = Observation(full_key, value, confidence, source, now,
                          self.OBSERVATION_TTL if ttl is None else ttl)
        before = self._snapshot()
        outcome = self.scene.observe(obs, now, question_open=False)
        return Turn(utterance=f"[scene] {key}={value}", reply=outcome.speech,
                    scene=outcome.scene or "—", deltas=self._deltas_since(before),
                    rules=list(self.scene.explain()), fallback=self.scene.fallback_note())

    def handle(self, utterance: str) -> Turn:
        before = self._snapshot()
        try:
            consent = self.scene.resolve(utterance, self._now())
            if consent.answered:
                # The driver was answering the car, not commanding it. Routing these words
                # would treat 好 as an utterance to match against 92 functions.
                return Turn(utterance=utterance, reply=consent.speech, scene="consent",
                            deltas=self._deltas_since(before))
            result = self.pipeline.route(utterance)
        except Exception as exc:                      # a crash costs a 60s reload; survive it
            return Turn(utterance=utterance, error=f"{type(exc).__name__}: {exc}")
        deltas = self._deltas_since(before)
        return Turn(utterance=utterance, reply=result.reply,
                    spans=[self._span(cl, deltas) for cl in result.clauses])

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
        self.car.set_signal(entity, attribute, number)
        return number

    def sensed_rows(self) -> list[dict]:
        """Every declared sensed signal, with its live value — the `/car` of the world.

        Always all of them, never only what moved: a speed resting at 0.0 is the answer to
        "why did the animal rule say nothing", and a pane that hides it cannot give it.

        The value comes from the car; the unit and the limits come from the declaration,
        because those limits are the ones `set_signal` above enforces — a control bounded by
        anything else would offer values this session refuses.
        """
        return [{"entity": e, "attribute": a, "value": self.car.get_signal(e, a),
                 "unit": unit, "min": lo, "max": hi}
                for e, a, _resting, unit, lo, hi, _max_age in sensed_signals()]

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
        return " · ".join(parts)
