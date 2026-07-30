# tests/scene/test_speech.py
"""Every sentence this subsystem can utter is in one table, and the table is checkable."""
import re

from scene.speech import SPEECH, speech_for


def test_every_intent_resolves_to_a_sentence():
    assert SPEECH and all(v.strip() for v in SPEECH.values())


def test_no_template_speaks_developer_text_to_the_driver():
    """This repo has twice shipped developer text into the cabin (e433e32, 70bfeb5).
    A table makes the check trivial, so there is no excuse for a third time.

    Brackets are in the class alongside letters because an unrendered `{state}` placeholder
    is the most likely way developer text reaches a driver now that render_response injects
    one — it would be spoken verbatim, braces and all. Digits are allowed: 三档 and 3档 are
    both things a car legitimately says."""
    for intent, text in SPEECH.items():
        assert not re.search(r"[A-Za-z_{}\[\]<>]", text), f"{intent}: {text}"


def test_every_template_ends_in_a_terminator():
    assert all(t[-1] in "。！？" for t in SPEECH.values())


def test_an_unknown_intent_is_silence_not_a_traceback():
    """An unsayable intent must degrade to no speech; raising here would kill a session
    after the work of the turn is already done."""
    assert speech_for("no_such_intent") == ""
