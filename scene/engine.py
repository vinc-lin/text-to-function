"""Perception in, at most one sentence out.

The engine may speak. It may not act. The car moves in exactly one place — `resolve()`, after
an explicit yes — and a contract test asserts that no rule match ever produces a ToolCall on
its own.

Everything degrades to silence: an exception, a missing model, a proposal that fails
validation. A system nobody asked to speak has silence as its safe default.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional

from t2f.respond import render_response
from t2f.types import ToolCall
from t2f.validate import validate_tool_call

from .consent import Answer, PendingConsent, classify
from .llm import UNMATCHED
from .rules import RULES, Verdict, evaluate_explained
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


# Deliberately NOT a field on SceneOutcome: NO_ACTION is one shared frozen instance, so a
# populated diagnostics field would stop `_evaluate` returning the singleton and break every
# `== NO_ACTION` comparison for the sake of a display. The engine records these instead.
@dataclass(frozen=True)
class RuleReport:
    rule_id: str
    verdict: str                     # the Verdict's value, or "error"
    reason: str
    suppressed_by: str = ""          # "" when nothing suppressed it


@dataclass
class ConsentResult:
    answered: bool                   # False -> caller routes the utterance normally
    speech: str = ""
    executed: bool = False
    tool_call: Optional[ToolCall] = None


NO_ACTION = SceneOutcome("no_action", "", "", None, "rule", "")


def _sentence(text: str) -> str:
    return text if (not text or text[-1] in "。！？") else text + "。"


def _decision_note(decision) -> str:
    """A decode was spent, so what came back is recorded whether or not it was usable."""
    if not isinstance(decision, dict):
        return f"model returned {type(decision).__name__}, not a decision"
    return f"{decision.get('decision')} · {str(decision.get('reason', ''))[:120]}"


class SceneEngine:
    def __init__(self, cards_by_name, world, executor, rules=RULES, llm=None,
                 consent_ttl: float = CONSENT_TTL, *, perception):
        """`world` is everything the rules READ; `perception` is the one store this engine
        WRITES, and the world must be a view over it.

        Two objects rather than one because the split is real: the engine is perception's only
        writer, and the world is deliberately read-through with no door to write through — a
        belief arriving through the hub would carry no source and no provenance, which is the
        thing the envelope exists to make impossible. The store is therefore passed in rather
        than created here: the world is a constructor argument, so it has to exist first, and a
        store created in this body is one no caller could have built a view over.

        Getting that wrong is silent — the engine writes where nothing reads, every rule sees an
        empty context, and the system is merely quiet — so it is checked below rather than
        documented. The check is duck-typed: a test may pass any object that answers `signal`,
        `signal_status` and `observation`, and only a real `WorldView` can answer `reads`.

        Required, with no default, since perception became a store: a `SceneContext` needs one,
        the schema for it lives in `sim` and the queries in `intake`, and both sit above `scene`
        in the layering — so there is no context this constructor could honestly build for a
        caller who forgot. A refusal at the door beats a store nobody else can reach.
        """
        self.cards = cards_by_name
        self.world = world
        self.executor = executor
        self.rules = tuple(rules)
        self.llm = llm
        self.consent_ttl = consent_ttl
        self.context = perception
        if hasattr(world, "reads") and not world.reads(self.context):
            raise ValueError(
                "the world must read the perception store this engine writes — pass "
                "perception=<the SceneContext the WorldView was built over>")
        self._pending: Optional[PendingConsent] = None
        self._last_spoken: dict[str, float] = {}
        self._last_fallback: Optional[float] = None
        self._last_reports: list = []
        self._last_fallback_note: str = ""

    def reset(self) -> None:
        """Forget the cabin. Called when the car underneath is replaced.

        Every piece of state here is about a specific vehicle at a specific moment: what
        perception believed, what question is outstanding, which rules have spoken recently,
        and the explanation of the last decision. Keeping any of them across a reset lets the
        driver answer a question that was asked about a car that no longer exists — and an
        explanation about the previous vehicle is the same class of mistake.
        """
        # Cleared in place, never rebound. A WorldView — or anything else — built over this
        # store keeps holding the same object across a reset; rebinding would leave every
        # reader pointed at the discarded instance, reading empty perception forever with
        # nothing raising.
        self.context.clear()
        self._pending = None
        self._last_spoken.clear()
        self._last_fallback = None
        self._last_reports = []
        self._last_fallback_note = ""

    # --- perception -------------------------------------------------------------------
    def observe(self, obs, now: float, *, question_open: bool = False) -> SceneOutcome:
        """Never raises. A traceback here would kill a session after the work is done."""
        # Cleared first, so an explanation of the previous observation can never be read as
        # an explanation of this one.
        self._last_reports = []
        self._last_fallback_note = ""
        try:
            self.context.update(obs)
            return self._evaluate(now, question_open=question_open)
        except Exception as exc:
            # An engine that fails silently and then explains nothing is the opacity this
            # recording exists to remove.
            self._last_reports = [RuleReport("—", "error", str(exc)[:120])]
            return NO_ACTION

    def explain(self) -> list:
        """What the LAST observation decided, per rule. Never re-evaluates.

        By the time anything asks, `observe()` has already written `_last_spoken` and possibly
        armed a pending consent, so a fresh run would describe a different world than the one
        that produced the outcome — it would report its own cooldown as the reason it was
        silent.
        """
        return self._last_reports

    def fallback_note(self) -> str:
        """What the constrained fallback did on the last observation, or why it was skipped."""
        return self._last_fallback_note

    def _evaluate(self, now: float, *, question_open: bool) -> SceneOutcome:
        explained = [(r, *evaluate_explained(r, self.world, now)) for r in self.rules]
        # Recorded BEFORE anything fires: afterwards the winner's own cooldown and pending
        # consent are in place, and it would report itself as suppressed.
        self._last_reports = [
            RuleReport(r.id, v.value, why,
                       self._suppressor(r, now, question_open=question_open)
                       if v is Verdict.MATCH else "")
            for r, v, why in explained]
        verdicts = [(r, v) for r, v, _ in explained]
        matched = [r for r, v in verdicts if v is Verdict.MATCH and self._speakable(r, now)]
        if matched:
            # Highest priority wins; ties break by declaration order, so the outcome does not
            # depend on dict ordering or on a clock.
            best = sorted(matched, key=lambda r: (-r.priority, self.rules.index(r)))[0]
            for loser in matched:
                if loser is not best:
                    self._amend(loser.id, f"outranked by {best.id}")
            outcome = self._fire(best, now, question_open=question_open)
            if outcome is NO_ACTION:
                # _fire declined after the reports were written. question_open was already
                # recorded by _suppressor, so the only cause left is a proposal that failed
                # validation — and "match" printed next to silence with no reason beside it
                # is precisely the display defect this recording exists to remove.
                self._amend(best.id, "proposal failed validation")
            return outcome
        return self._fallback(verdicts, now, question_open=question_open)

    def _amend(self, rule_id: str, suppressed_by: str) -> None:
        """Fill in a suppressor discovered after the reports were written.

        Fills an EMPTY one only. A reason already recorded by `_suppressor` is the earlier
        and more specific cause; overwriting it would report the second thing that would have
        stopped the rule rather than the first thing that did.
        """
        self._last_reports = [
            replace(r, suppressed_by=suppressed_by)
            if r.rule_id == rule_id and not r.suppressed_by else r
            for r in self._last_reports]

    def _suppressor(self, rule, now: float, *, question_open: bool) -> str:
        """Why a matched rule would not speak — "" when nothing stops it.

        The single source of truth: `_speakable` is this same question asked as a boolean. A
        second copy written for the display would drift from the one that decides, and the
        display would then confidently explain a silence that had another cause.
        """
        reasons = []
        last = self._last_spoken.get(rule.id)
        if last is not None and now - last < rule.cooldown:
            reasons.append(f"cooldown, {rule.cooldown - (now - last):.0f}s left")
        pending = self.pending(now)
        # Do not re-ask a question we are already waiting on an answer to.
        if pending is not None and pending.scene == rule.id:
            reasons.append("already asked, awaiting an answer")
        if question_open:
            reasons.append("router holds a question")
        return "; ".join(reasons)

    def _speakable(self, rule, now: float) -> bool:
        # question_open is excluded deliberately: _fire checks it later so that a rule the
        # router silenced does not also burn its own cooldown.
        return not self._suppressor(rule, now, question_open=False)

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
            self._last_fallback_note = "no model attached"
            return NO_ACTION
        if question_open:
            # Checked before the decode, not after, for the reason _fire states: a subsystem
            # silenced by someone else's question should not also pay for the answer or burn
            # its own window. Discovering this after the call spent a decode and consumed the
            # budget, then discarded the result.
            self._last_fallback_note = "router holds a question"
            return NO_ACTION
        near = [r for r, v in verdicts if v is Verdict.NEAR_MISS]
        mentioned = {k for r in self.rules for k in r.observed_keys}
        # `live_observations`, not `live_facts`: the flattened form drops confidence, which is
        # what describes a near-miss, and mixes in vehicle signals, which no rule's
        # `observed_keys` can mention — every one of them would read as unconsumed and the
        # fallback would fire on a parked car saying nothing.
        live = self.world.live_observations(now)
        unconsumed = [k for k in live if k not in mentioned]
        if not near and not unconsumed:
            self._last_fallback_note = "no near-miss or unconsumed observation"
            return NO_ACTION
        if self._last_fallback is not None and now - self._last_fallback < FALLBACK_COOLDOWN:
            remaining = FALLBACK_COOLDOWN - (now - self._last_fallback)
            self._last_fallback_note = f"budget: {remaining:.0f}s remaining"
            return NO_ACTION
        self._last_fallback = now
        snapshot = {k: {"value": o.value, "confidence": o.confidence, "source": o.source}
                    for k, o in live.items()}
        try:
            decision = self.llm.decide(snapshot, self.rules, SPEECH)
        except Exception as exc:
            self._last_fallback_note = f"model raised: {str(exc)[:80]}"
            return NO_ACTION
        self._last_fallback_note = _decision_note(decision)
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
        intent = decision.get("reply_intent", "")
        speech = _sentence(speech_for(intent))
        reason = str(decision.get("reason", ""))[:200]
        if kind == "notify" and speech:
            # A notify carrying an ask_* intent would put a question in the cabin with no
            # pending consent behind it, and the driver's 好 would answer into the void.
            # scene_decision_schema makes this ungrammatical; the check stays because a
            # scripted fake, a drifted schema version, or a future intent added to the wrong
            # prefix can all still produce it.
            if intent.startswith("ask_"):
                return NO_ACTION
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
