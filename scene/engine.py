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
from .llm import UNMATCHED
from .rules import RULES, Verdict, evaluate
from .speech import SPEECH, speech_for

CONSENT_TTL = 30.0
FALLBACK_COOLDOWN = 30.0
_FAILURE = "抱歉，这个操作没能完成。"


# Frozen because NO_ACTION below is a single shared instance: one `out.speech = ...` on a
# returned no-action would poison every `== NO_ACTION` comparison in the process, including
# the ones the tests rely on.
@dataclass(frozen=True)
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
        self._last_fallback: Optional[float] = None

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
        return self._fallback(verdicts, now, question_open=question_open)

    def _speakable(self, rule, now: float) -> bool:
        last = self._last_spoken.get(rule.id)
        if last is not None and now - last < rule.cooldown:
            return False
        pending = self.pending(now)
        # Do not re-ask a question we are already waiting on an answer to.
        return not (pending is not None and pending.scene == rule.id)

    def _fire(self, rule, now: float, *, question_open: bool) -> SceneOutcome:
        if question_open:
            # Checked before anything is recorded, so a rule the router silenced does not
            # also burn its own cooldown. Notifications are covered too: a notify creates no
            # pending consent and so cannot make 好 ambiguous, but talking over a question
            # the driver is being asked is still the scene subsystem interrupting.
            return NO_ACTION
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
        self._last_spoken[rule.id] = now
        self._pending = PendingConsent(rule.id, tc, asked_at=now, expires_after=self.consent_ttl)
        return SceneOutcome("ask", rule.id, _sentence(speech_for(rule.intent)),
                            tc, "rule", "rule matched")

    # --- the constrained fallback -------------------------------------------------------
    def _fallback(self, verdicts, now: float, *, question_open: bool) -> SceneOutcome:
        """Reached only when no rule matched. Near-misses and observations no rule mentions.

        A REJECT is deliberately not a near-miss: the car has already settled the question, so
        spending a decode on it would be paying for an answer we have.
        """
        if self.llm is None:
            return NO_ACTION
        near = [r for r, v in verdicts if v is Verdict.NEAR_MISS]
        mentioned = {k for r in self.rules for k in r.observed_keys}
        live = self.context.live(now)
        unconsumed = [k for k in live if k not in mentioned]
        if not near and not unconsumed:
            return NO_ACTION
        if self._last_fallback is not None and now - self._last_fallback < FALLBACK_COOLDOWN:
            return NO_ACTION
        self._last_fallback = now
        snapshot = {k: {"value": o.value, "confidence": o.confidence, "source": o.source}
                    for k, o in live.items()}
        try:
            decision = self.llm.decide(snapshot, self.rules, SPEECH)
        except Exception:
            return NO_ACTION
        return self._from_decision(decision, now, question_open=question_open)

    def _from_decision(self, decision, now: float, *, question_open: bool) -> SceneOutcome:
        if not isinstance(decision, dict):
            return NO_ACTION
        if question_open:
            # Same rule as the deterministic path: the scene subsystem does not talk over a
            # question the router is waiting on.
            return NO_ACTION
        kind = decision.get("decision")
        scene = decision.get("scene", UNMATCHED)
        speech = _sentence(speech_for(decision.get("reply_intent", "")))
        reason = str(decision.get("reason", ""))[:200]
        if kind == "notify" and speech:
            return SceneOutcome("notify", scene, speech, None, "llm", reason)
        if kind == "ask":
            # An ask is legal ONLY when it names a real rule carrying a proposal: an unmatched
            # scene has nothing for consent to authorise, so asking would open a question no
            # answer could act on. The rule then owns what is said and what is proposed — the
            # model only chose whether to ask, which is why the outcome's source is "rule".
            rule = next((r for r in self.rules if r.id == scene and r.proposes), None)
            if rule is None or not speech:
                return NO_ACTION
            return self._fire(rule, now, question_open=question_open)
        return NO_ACTION

    # --- consent ----------------------------------------------------------------------
    def pending(self, now: float) -> Optional[PendingConsent]:
        if self._pending is not None and not self._pending.is_live(now):
            self._pending = None
        return self._pending

    def resolve(self, utterance: str, now: float) -> ConsentResult:
        """Never raises, for the same reason observe() does not — and more urgently.

        This is the path that actually touches the car. SqliteExecutor turns every *modelled*
        refusal into ExecResult(ok=False), so anything that escapes as an exception is an
        infrastructure fault — a locked database, a disk error — and those are exactly the
        moments when a traceback would kill the session mid-actuation.
        """
        try:
            return self._resolve(utterance, now)
        except Exception:
            self._pending = None
            return ConsentResult(answered=True, speech=_FAILURE)

    def _resolve(self, utterance: str, now: float) -> ConsentResult:
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
            # Stripped before the emptiness test, the way t2f/reply.py::_exec_failures already
            # does it. Without the strip a whitespace-only detail is truthy after _sentence
            # appends a terminator, and the driver hears a spoken full stop with no cause.
            detail = (res.detail or "").strip()
            return ConsentResult(answered=True, speech=_sentence(detail) or _FAILURE,
                                 tool_call=tc)
        return ConsentResult(answered=True, executed=True, tool_call=tc,
                             speech=_sentence(render_response(self.cards[tc.name], tc)))
