"""The five actions and the one control, each one call the CLI already makes.

`ACTIONS` and `CONTROLS` together are the whole surface reachable from the page, and they are
two tables rather than one because they are two different things — see the comment above
`CONTROLS`. Every entry in either is a one-liner onto a `Session` method the terminal also
calls: the Yes button submits 好 through `handle()` exactly as typing it does, and nothing
here touches the executor or the engine directly. A button wired past the session would be a
second route to the car with its own rules, and the point of the consent design is that there
is only one.

An entry that could not be written as an existing session call is a signal to stop and ask,
not to add a shortcut here.
"""
from __future__ import annotations

CONVERSATION_LIMIT = 60


def _remember(session, who: str, text: str) -> None:
    """Append to the transcript the page renders.

    Kept on the session object rather than in a `Session` field, and read back by
    `ui/state.py:_conversation`: this is presentation history, nothing may decide anything
    from it, and a field on Session would invite exactly that. The terminal's equivalent is
    the scrollback, which the CLI also does not store.
    """
    if not text:
        return
    log = getattr(session, "ui_conversation", None)
    if log is None:
        log = session.ui_conversation = []
    log.append({"who": who, "text": text})
    del log[:-CONVERSATION_LIMIT]


def _observe(session, payload) -> dict:
    """One perception event, as /scene types it."""
    turn = session.observe(payload["key"], payload["value"],
                           float(payload.get("confidence", 0.9)),
                           payload.get("source", "cabin_cam"),
                           # None, not a default: Session.OBSERVATION_TTL is the session's
                           # answer to "how long", and repeating the number here would give
                           # the page a second opinion that could drift from it.
                           ttl=None if payload.get("ttl") is None else float(payload["ttl"]))
    _remember(session, "scene", turn.reply)
    return {"reply": turn.reply}


def _say(session, payload) -> dict:
    """The driver's words — a command, or 好 answering a question the car asked."""
    utterance = str(payload["utterance"])
    _remember(session, "driver", utterance)
    turn = session.handle(utterance)
    _remember(session, "car", turn.reply or turn.error or "")
    return {"reply": turn.reply}


def _clock(session, payload) -> dict:
    return {"reply": f"clock offset {session.advance_clock(float(payload['seconds'])):+.0f}s"}


def _reset(session, payload) -> dict:
    session.reset()
    # The transcript goes with the car. It is a record of what was said to a vehicle that no
    # longer exists, and leaving it would show consent given for signals now back at seeded.
    session.ui_conversation = []
    return {"reply": "fresh car"}


def _scene_llm(session, payload) -> dict:
    """Attach or detach the scene fallback — the same construction path as /scene-llm on.

    Refusal under --fake is copied from `cli/__main__.py:_build_scene_llm` on purpose: a
    scripted stand-in would fabricate the very thing the fallback is attached to observe, so
    the decision on the page would be one this code wrote rather than one a model made.
    """
    if not payload.get("on"):
        session.attach_scene_llm(None)
        return {"reply": session.mode_label()}
    if session.fake:
        raise ValueError("the scene fallback needs a real model — restart without --fake")
    from scene.llm import TransformersSceneLLM
    # Imported and constructed here rather than at module import: it loads a second copy of
    # Qwen3-0.6B, and a page that is only ever going to watch the rules must not pay for it.
    session.attach_scene_llm(TransformersSceneLLM())
    return {"reply": session.mode_label()}


ACTIONS = {
    "observe": _observe,
    "say": _say,
    "clock": _clock,
    "reset": _reset,
    "scene_llm": _scene_llm,
}


def perform(session, name: str, payload: dict) -> dict:
    """Run one named action. An unknown name raises KeyError — it is never guessed.

    The lookup is the enforcement: a name absent from the table has no route to the car at
    all, so a new button cannot reach the vehicle without appearing in this file.
    """
    return ACTIONS[name](session, payload)


# --- simulator controls, which are not actions --------------------------------------------

def _set_signal(session, payload) -> dict:
    """Tell the simulator what the car is doing, as /signal types it.

    `Session.set_signal` does the refusing — of an actuated signal, and of a value outside the
    declared limits — so the page cannot be a laxer door onto the car than the terminal is.
    """
    value = session.set_signal(payload["entity"], payload["attribute"], payload["value"])
    return {"reply": f"{payload['entity']}/{payload['attribute']} = {value}"}


# A SEPARATE table from ACTIONS, on purpose, and the two must stay disjoint.
#
# An action is something the Central Model performs: the driver says it, the pipeline routes
# it, the executor applies it under every check the executor exists to apply. Setting a sensed
# signal is none of that — it is the WORLD changing, the same category as the camera seeing a
# child or /reset re-seeding the vehicle, and nothing in the catalog can produce it.
#
# A sixth entry in ACTIONS would have made the page's route to the car look uniform when it is
# not. Two tables with different names and different justifications means anyone later asking
# "how does the page reach the car" finds the distinction before they find a shortcut.
CONTROLS = {
    "set_signal": _set_signal,
}


def perform_control(session, name: str, payload: dict) -> dict:
    """Run one named control. Mirrors `perform`, over the other table, for the same reason:
    a name absent from it has no route to the simulator at all."""
    return CONTROLS[name](session, payload)
