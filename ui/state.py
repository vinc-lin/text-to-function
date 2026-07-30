"""One `Session` in, one JSON-able dict out. The whole contract between the engine and
the page.

Pure: nothing here writes to the session, the car or the engine. `GET /state` is polled
several times a second, and a snapshot that mutated anything would make the instrument a
participant in what it is meant to be observing.

Nothing is stored for the page's benefit either, with one labelled exception — the
conversation transcript, which is presentation history and has nowhere else to live; see
`_conversation`.
"""
from __future__ import annotations
import functools

from scene.speech import speech_for

LOG_LIMIT = 20
CONVERSATION_LIMIT = 60


def _pane(empty):
    """A pane that cannot be built yields its empty shape instead of propagating.

    The page polls continuously, including mid-reset and mid-rebuild. One exception
    anywhere replaces the entire instrument with a 500 exactly when something interesting
    is happening, so a pane that fails costs that pane and nothing else.

    `empty` is a factory, not a value: a shared `[]` handed out on every failure would be
    a mutable default that a caller could poison for every later reader.
    """
    def wrap(build):
        @functools.wraps(build)
        def pane(session):
            try:
                return build(session)
            except Exception:
                return empty()
        return pane
    return wrap


@_pane(lambda: "—")
def _mode(session) -> str:
    return session.mode_label()


@_pane(lambda: False)
def _scene_llm(session) -> bool:
    return session.scene.llm is not None


@_pane(lambda: 0.0)
def _clock_offset(session) -> float:
    return float(session.clock_offset)


@_pane(list)
def _perception(session) -> list:
    """`context_rows()` is exactly what the CLI's /context prints, plus the one thing a
    draining bar needs and a row does not carry: the ttl the observation was created with,
    which is what `expires_in` is a fraction OF.

    Read from the observation rather than recomputed as `age + expires_in`: that sum is
    the ttl only up to float error, and the page divides by it.
    """
    ttls = {key: obs.ttl for key, obs in session.scene.context.live(session._now()).items()}
    return [{"key": r.key, "value": r.value, "confidence": r.confidence, "source": r.source,
             "age": r.age, "expires_in": r.expires_in,
             # A row present in context_rows() but not in `ttls` can only be one observed in
             # the microseconds between the two clock reads; its own remaining time is the
             # honest denominator, and it is never zero.
             "ttl": ttls.get(r.key, r.expires_in)}
            for r in session.context_rows()]


@_pane(list)
def _car(session) -> list:
    """Only signals that differ from the seeded car — the same answer /car gives.

    Sorted, because the page flashes a row when its value changes and dict iteration order
    would make every unrelated write look like movement.
    """
    return [{"entity": e, "attribute": a, "value": v}
            for e, a, v in sorted(session.changed_signals())]


@_pane(list)
def _rules(session) -> list:
    """What the LAST observation decided, per rule. `explain()` never re-evaluates.

    `threshold`, `floor` and `observed_keys` come from the rule itself rather than from the
    reason string. The page draws a near-miss as a POSITION between the floor and the
    threshold, and the alternative was regexing `reason` — which scene/rules.py documents as
    diagnostics for a developer at a terminal, and which is therefore free to change wording
    without warning. Three fields off an object already in hand is cheaper than a parser that
    breaks silently.
    """
    by_id = {r.id: r for r in session.scene.rules}
    out = []
    for report in session.scene.explain():
        rule = by_id.get(report.rule_id)
        out.append({
            "rule_id": report.rule_id, "verdict": report.verdict, "reason": report.reason,
            "suppressed_by": report.suppressed_by,
            # A report with no matching rule is the "—" row observe() records when the engine
            # itself raised; it has no bands to draw and must not invent any.
            "threshold": rule.threshold if rule else None,
            "floor": rule.floor if rule else None,
            "observed_keys": list(rule.observed_keys) if rule else [],
        })
    return out


@_pane(str)
def _fallback(session) -> str:
    return session.scene.fallback_note()


@_pane(lambda: None)
def _pending(session):
    """The outstanding question, its sentence, and how long it has left.

    `PendingConsent` carries the scene id and the proposal, never the sentence — it is
    recovered here from the rule's intent, so there is exactly one copy of what the car
    said and it cannot disagree with what the driver heard. The engine's own rule tuple is
    consulted rather than `scene.RULES`, since the engine may have been built with another.
    """
    now = session._now()
    pending = session.scene.pending(now)
    if pending is None:
        return None
    rule = next((r for r in session.scene.rules if r.id == pending.scene), None)
    return {"scene": pending.scene,
            "question": speech_for(rule.intent) if rule is not None else "",
            "expires_in": pending.asked_at + pending.expires_after - now}


@_pane(list)
def _conversation(session) -> list:
    """The transcript, which is presentation history and nothing else.

    The session has no notion of one — the CLI's transcript is the terminal scrollback —
    so `ui/actions.py` keeps it on the session object under this name and this reads it.
    Deliberately not a `Session` field: nothing that decides anything may consult it, and
    a field on Session would invite exactly that.
    """
    return list(getattr(session, "ui_conversation", []))[-CONVERSATION_LIMIT:]


@_pane(list)
def _log(session) -> list:
    """Most recent first, as `recent_operations` returns them."""
    return [{"function": r["function"], "outcome": r["outcome"],
             "error": r["error"], "detail": r["detail"]}
            for r in session.car.recent_operations(LOG_LIMIT)]


def snapshot(session) -> dict:
    return {
        "mode": _mode(session),
        # Structured, not parsed back out of `mode`. The label joins its parts with " · " and
        # a page testing it for "S_llm" would also match "S" — the toggle would read as on the
        # moment the scene fallback was off.
        "scene_llm": _scene_llm(session),
        "clock_offset": _clock_offset(session),
        "perception": _perception(session),
        "car": _car(session),
        "rules": _rules(session),
        "fallback": _fallback(session),
        "pending": _pending(session),
        "conversation": _conversation(session),
        "log": _log(session),
    }
