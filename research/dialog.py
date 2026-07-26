from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from t2f.types import ClauseResult, Decision, Band


@dataclass
class PendingState:
    """A clarification awaiting the driver's next turn. Lives here, not in t2f/types.py:
    Pipeline.route() never reads it — only this resolver does."""
    pending_function: str
    known_parameters: dict[str, Any]
    missing_parameters: list[str]


@dataclass
class SessionState:
    pending: Optional[PendingState] = None
    turn_count: int = 0

from t2f.params.extract import ParameterExtractor
from t2f.validate import validate_tool_call
from t2f.respond import render_response, build_clarification
from t2f.execute import MockExecutor


class FollowUpResolver:
    def __init__(self, cards_by_name, extractor=None, llm_client=None, max_turns: int = 2, executor=None):
        self.cards = cards_by_name
        self.extractor = extractor or ParameterExtractor()
        self.llm_client = llm_client
        self.max_turns = max_turns
        self.executor = executor or MockExecutor()

    def _extract_missing(self, card, utterance, features, missing):
        params, _ = self.extractor.extract(utterance, features, card)
        return {k: v for k, v in params.items() if k in missing}

    def is_followup(self, session: SessionState, utterance: str) -> bool:
        if not session or not session.pending:
            return False
        card = self.cards.get(session.pending.pending_function)
        if card is None:
            return False
        from t2f.lexical import extract_features
        got = self._extract_missing(card, utterance, extract_features(utterance),
                                    session.pending.missing_parameters)
        return len(utterance) < 12 or len(got) > 0

    def resolve(self, session: SessionState, utterance: str, features):
        pending = session.pending
        card = self.cards[pending.pending_function]
        got = self._extract_missing(card, utterance, features, pending.missing_parameters)
        known = {**pending.known_parameters, **got}
        still_missing = [m for m in pending.missing_parameters if m not in known]
        decision = Decision(Band.MEDIUM, card.name, [])
        if not still_missing:
            tc, errs = validate_tool_call(card.name, known, self.cards, [card.name])
            if tc is not None:
                self.executor.execute(tc)
                return (ClauseResult(clause=utterance, decision=decision, tool_call=tc,
                                     response=render_response(card, tc)), SessionState())
            still_missing = [e.message.split()[-1] for e in errs if e.code == "missing_required"] or pending.missing_parameters
        if session.turn_count + 1 < self.max_turns:
            clar = build_clarification(card, still_missing)
            return (ClauseResult(clause=utterance, decision=decision, clarification=clar),
                    SessionState(pending=PendingState(card.name, known, still_missing),
                                 turn_count=session.turn_count + 1))
        clar = build_clarification(card, still_missing)
        return (ClauseResult(clause=utterance, decision=decision, clarification=clar), SessionState())
