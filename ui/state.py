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
# Fewer than the log and the transcript keep, and deliberately. A record entry is four lines
# and the pane is the page's history, not its scrollback — the question it answers is "why did
# it just do that", which the last few turns answer and forty do not.
STORE_LIMIT = 8


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
def _sensed(session) -> list:
    """Every signal the car senses, with its live value, its unit and its limits.

    A separate key from `car`, which means "changed from seeded" and has to keep meaning
    that. A speed resting at 0.0 is not a change and would never appear there — yet it is the
    entire answer to "why did the animal rule say nothing", so the page needs it visible
    whatever its value. Always-visible and appears-when-it-moves are different questions and
    one list cannot answer both.

    The limits ride along because the page draws a bounded control from them, and they are
    the same ones `Session.set_signal` enforces.

    `age` and `stale` ride along for the same reason one level up: they are what the WORLD
    says about this signal, so the pane and the rules cannot disagree about whether a value is
    still true. Computing staleness here — an age from the car, a max from the declaration, a
    comparison in a display — would be a second belief about one bus, and a row reading "45
    kph" beside a rule rejecting it as stale is the failure this whole discipline removes.
    """
    return session.sensed_rows()


@_pane(lambda: False)
def _bus(session) -> bool:
    """Whether the sensed-signal publisher is running.

    Not derivable from the rows, which is why it is its own field: a stopped bus whose values
    are still inside their max age looks exactly like a running one, and the toggle has to know
    which it is before the first signal goes stale.

    False when it cannot be asked, and that direction is deliberate. "Running" is the claim
    that everything on the Vehicle pane is fresh; a pane that could not reach the session must
    not make it.
    """
    return session.bus_publishing()


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


@_pane(list)
def _store(session) -> list:
    """What the store wrote down — the history behind every other pane.

    Every other pane is the system NOW: what perception believes, what the car holds, what the
    last observation decided. `explain()` is cleared by the next event and the transcript is
    presentation history that nothing may consult. This is the only pane reading the thing that
    survives the session, and it is the only one that can answer a question about a turn that
    has already scrolled past.

    `Session.recent_turns` exactly as the terminal's `/store` prints it, unreshaped: two doors
    onto one session must not be able to show two histories of it. Read-only, and there is no
    action or control behind it — the record is written by the door, never by the page.
    """
    return session.recent_turns(STORE_LIMIT)


def snapshot(session) -> dict:
    return {
        "mode": _mode(session),
        # Structured, not parsed back out of `mode`. The label joins its parts with " · " and
        # a page testing it for "S_llm" would also match "S" — the toggle would read as on the
        # moment the scene fallback was off.
        "scene_llm": _scene_llm(session),
        "clock_offset": _clock_offset(session),
        "perception": _perception(session),
        "sensed": _sensed(session),
        # Beside `sensed` because it is what those rows MEAN: with the bus stopped, every age
        # in them is climbing toward the moment no rule reads that signal any more.
        "bus": _bus(session),
        "car": _car(session),
        "rules": _rules(session),
        "fallback": _fallback(session),
        "pending": _pending(session),
        "conversation": _conversation(session),
        "log": _log(session),
        # Last, because it is the only key that is not a fact about now. Everything above is
        # the state of the running system; this is what it recorded on the way here.
        "store": _store(session),
    }
