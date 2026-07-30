"""The fallback picks from what exists. It cannot invent a scene, a sentence, or an action."""
import pytest

from scene.llm import FakeSceneLLM, scene_decision_schema
from scene.rules import RULES
from scene.speech import SPEECH


def test_the_schema_offers_no_execute_decision():
    """The model cannot ask for the car to move. The most it can do is propose a question,
    and that still needs consent."""
    schema = scene_decision_schema(RULES, SPEECH)
    assert set(schema["properties"]["decision"]["enum"]) == {"notify", "ask", "no_action"}


def test_the_scene_enum_is_the_rule_ids_plus_unmatched():
    schema = scene_decision_schema(RULES, SPEECH)
    assert set(schema["properties"]["scene"]["enum"]) == {r.id for r in RULES} | {"unmatched"}


def test_the_intent_enum_is_exactly_the_speech_table():
    """If these drift apart the model can pick an intent that resolves to silence."""
    schema = scene_decision_schema(RULES, SPEECH)
    assert set(schema["properties"]["reply_intent"]["enum"]) == set(SPEECH)


def test_every_field_is_required_and_nothing_else_is_allowed():
    schema = scene_decision_schema(RULES, SPEECH)
    assert set(schema["required"]) == {"decision", "scene", "reason", "reply_intent"}
    assert schema["additionalProperties"] is False


def test_the_fake_returns_its_script():
    llm = FakeSceneLLM([{"decision": "notify", "scene": "unmatched",
                         "reason": "r", "reply_intent": "notify_driver_fatigue"}])
    assert llm.decide({}, RULES, SPEECH)["decision"] == "notify"


def test_the_fake_returns_none_when_the_script_runs_out():
    """Exhausted script means no decision, which the engine must read as silence."""
    assert FakeSceneLLM([]).decide({}, RULES, SPEECH) is None
