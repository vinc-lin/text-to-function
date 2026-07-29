"""The interactive session's logic: one utterance in, one structured Turn out.

Deliberately free of I/O so it can be tested. `cli/render.py` turns a Turn into text and
`cli/__main__.py` does the reading and printing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

import json

from t2f.build import build_pipeline
from t2f.cards import load_catalog, load_ood_prototypes
from t2f.config import Config
from t2f.gate import Thresholds
from t2f.types import LLMResult, ToolCall
from sim.executor import SqliteExecutor
from sim.mapping import resolve_writes
from sim.seed import seed_from_catalog
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
class Turn:
    utterance: str
    spans: list = field(default_factory=list)
    reply: str = ""
    error: Optional[str] = None


class Session:
    def __init__(self, pipeline, car, executor, cards, config, *, fake, llm, gate):
        self.pipeline, self.car, self.executor = pipeline, car, executor
        self.cards, self.config = cards, config
        self.fake, self.llm, self.gate = fake, llm, gate

    # --- construction ------------------------------------------------------------------
    @classmethod
    def build(cls, *, fake=False, llm=True, gate="shipped", db=":memory:",
              catalog="data/catalog", config_path="config.yaml"):
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

    # --- one turn ----------------------------------------------------------------------
    def handle(self, utterance: str) -> Turn:
        before = self._snapshot()
        try:
            result = self.pipeline.route(utterance)
        except Exception as exc:                      # a crash costs a 60s reload; survive it
            return Turn(utterance=utterance, error=f"{type(exc).__name__}: {exc}")
        after = self._snapshot()
        deltas = [Delta(e, a, before.get((e, a)), v) for (e, a), v in after.items()
                  if before.get((e, a)) != v]
        return Turn(utterance=utterance, reply=result.reply,
                    spans=[self._span(cl, deltas) for cl in result.clauses])

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
            escalated=clause_result.needs_llm,
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

    def reset(self):
        self.car.conn.execute("DELETE FROM signal")
        self.car.conn.execute("DELETE FROM operation_log")
        # `precondition` has no unique key and seed_from_catalog re-inserts every row, so a
        # session that resets twice would check the same precondition three times.
        self.car.conn.execute("DELETE FROM precondition")
        self.car.conn.commit()
        seed_from_catalog(self.car, self.cards)
        self._seeded = self._snapshot()

    def mode_label(self) -> str:
        parts = ["C_llm" if self.llm else "C", self.gate]
        if self.fake:
            parts.append("FAKE")
        return " · ".join(parts)
