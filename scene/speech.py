"""Every sentence this subsystem can utter.

A table, not generation. The model selects an intent from these keys; it never authors what
the car says. Two prior fixes in this repo (e433e32, 70bfeb5) exist because generated or
internal text reached the driver, and a 0.6B model writing unreviewed sentences into a cabin
is a larger claim than this subsystem makes.

The confirmation AFTER consent is deliberately not here: it comes from render_response on the
executed card, so a scene-initiated action confirms exactly as a driver-initiated one does.
"""
from __future__ import annotations

SPEECH: dict[str, str] = {
    "ask_rear_child_lock":   "后排有小孩，要打开儿童锁吗？",
    "notify_animal_ahead":   "前方有动物，请注意。",
    "notify_driver_fatigue": "您看起来有些疲劳，请注意休息。",
    "ack_declined":          "好的。",
}


def speech_for(intent: str) -> str:
    """'' for an unknown intent. An unsayable intent is silence, never an exception."""
    return SPEECH.get(intent, "")
