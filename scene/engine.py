"""Perception in, at most one sentence out.

The engine may speak. It may not act. The car moves in exactly one place — `resolve()`, after
an explicit yes — and a contract test asserts that no rule match ever produces a ToolCall on
its own.

Everything degrades to silence: an exception, a missing model, a proposal that fails
validation. A system nobody asked to speak has silence as its safe default.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from t2f.respond import render_response
from t2f.types import ToolCall
from t2f.validate import validate_tool_call

from .consent import Answer, PendingConsent, classify
from .context import SceneContext
from .rules import RULES, Verdict, evaluate
from .speech import speech_for

CONSENT_TTL = 30.0
_FAILURE = "抱歉，这个操作没能完成。"


@dataclass
class SceneOutcome:
    kind: str                        # "notify" | "ask" | "no_action"
    scene: str
    speech: str
    proposal: Optional[ToolCall]
    source: str                      # "rule" | "llm"
    reason: str                      # diagnostic, never spoken


@dataclass
class ConsentResult:
    answered: bool                   # False -> caller routes the utterance normally
    speech: str = ""
    executed: bool = False
    tool_call: Optional[ToolCall] = None


NO_ACTION = SceneOutcome("no_action", "", "", None, "rule", "")


def _sentence(text: str) -> str:
    return text if (not text or text[-1] in "。！？") else text + "。"


class SceneEngine:
    def __init__(self, cards_by_name, facts, executor, rules=RULES, llm=None,
                 consent_ttl: float = CONSENT_TTL):
        self.cards = cards_by_name
        self.facts = facts
        self.executor = executor
        self.rules = tuple(rules)
        self.llm = llm
        self.consent_ttl = consent_ttl
        self.context = SceneContext()
        self._pending: Optional[PendingConsent] = None
        self._last_spoken: dict[str, float] = {}

    # --- perception -------------------------------------------------------------------
    def observe(self, obs, now: float, *, question_open: bool = False) -> SceneOutcome:
        """Never raises. A traceback here would kill a session after the work is done."""
        try:
            self.context.update(obs)
            return self._evaluate(now, question_open=question_open)
        except Exception:
            return NO_ACTION

    def _evaluate(self, now: float, *, question_open: bool) -> SceneOutcome:
        verdicts = [(r, evaluate(r, self.context, self.facts, now)) for r in self.rules]
        matched = [r for r, v in verdicts if v is Verdict.MATCH and self._speakable(r, now)]
        if matched:
            # Highest priority wins; ties break by declaration order, so the outcome does not
            # depend on dict ordering or on a clock.
            best = sorted(matched, key=lambda r: (-r.priority, self.rules.index(r)))[0]
            return self._fire(best, now, question_open=question_open)
        return NO_ACTION

    def _speakable(self, rule, now: float) -> bool:
        last = self._last_spoken.get(rule.id)
        if last is not None and now - last < rule.cooldown:
            return False
        pending = self.pending(now)
        # Do not re-ask a question we are already waiting on an answer to.
        return not (pending is not None and pending.scene == rule.id)

    def _fire(self, rule, now: float, *, question_open: bool) -> SceneOutcome:
        if rule.proposes is None:
            self._last_spoken[rule.id] = now
            return SceneOutcome("notify", rule.id, _sentence(speech_for(rule.intent)),
                                None, "rule", "rule matched")
        # Validate BEFORE asking. Discovering after the driver says 好 that the call was never
        # usable is the proactive form of a falsely-affirmative reply.
        tc, _ = validate_tool_call(rule.proposes.name, dict(rule.proposes.parameters),
                                   self.cards, [rule.proposes.name])
        if tc is None:
            return NO_ACTION
        if question_open:
            # At most one open question across both systems, or 好 becomes ambiguous.
            return NO_ACTION
        self._last_spoken[rule.id] = now
        self._pending = PendingConsent(rule.id, tc, asked_at=now, expires_after=self.consent_ttl)
        return SceneOutcome("ask", rule.id, _sentence(speech_for(rule.intent)),
                            tc, "rule", "rule matched")

    # --- consent ----------------------------------------------------------------------
    def pending(self, now: float) -> Optional[PendingConsent]:
        if self._pending is not None and not self._pending.is_live(now):
            self._pending = None
        return self._pending

    def resolve(self, utterance: str, now: float) -> ConsentResult:
        pending = self.pending(now)
        if pending is None:
            return ConsentResult(answered=False)
        answer = classify(utterance)
        if answer is Answer.NOT_AN_ANSWER:
            # Abandon the question rather than hold it open: a driver who said something else
            # has moved on, and a stale question would make the NEXT 好 ambiguous.
            self._pending = None
            return ConsentResult(answered=False)
        self._pending = None
        if answer is Answer.NO:
            return ConsentResult(answered=True, speech=speech_for("ack_declined"))
        return self._execute(pending.proposal)

    def _execute(self, proposal: ToolCall) -> ConsentResult:
        """Consent authorises an action, not an outcome.

        The car may have changed between the question and the answer, so the call is
        re-validated and re-dispatched here rather than trusted from ask time.
        """
        tc, _ = validate_tool_call(proposal.name, dict(proposal.parameters),
                                   self.cards, [proposal.name])
        if tc is None:
            return ConsentResult(answered=True, speech=_FAILURE)
        res = self.executor.execute(tc)
        if not res.ok:
            return ConsentResult(answered=True, speech=_sentence(res.detail or "") or _FAILURE,
                                 tool_call=tc)
        return ConsentResult(answered=True, executed=True, tool_call=tc,
                             speech=_sentence(render_response(self.cards[tc.name], tc)))
