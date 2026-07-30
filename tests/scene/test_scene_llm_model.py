"""The real decoder against the real grammar. Deselected by default (`-m model`).

The design promised this test, and it earns its place: every other fallback test scripts a
FakeSceneLLM, so nothing else would notice if xgrammar's HF integration shifted names again
(t2f/llm/client.py:62 carries that caveat, and scene/llm.py now carries a second copy of the
same version-sensitive line). Arm S_llm would simply report silence, which is indistinguishable
from the model correctly declining to speak.
"""
import pytest

from scene.llm import TransformersSceneLLM, scene_decision_schema
from scene.rules import RULES
from scene.speech import SPEECH


@pytest.mark.model
def test_the_real_model_emits_a_decision_the_grammar_permits():
    llm = TransformersSceneLLM()
    snapshot = {"inside.rear_occupant": {"value": "child", "confidence": 0.62,
                                         "source": "cabin_cam"}}
    decision = llm.decide(snapshot, RULES, SPEECH)
    assert isinstance(decision, dict), "constrained decoding produced unparseable output"

    branches = {b["properties"]["decision"]["const"]: b
                for b in scene_decision_schema(RULES, SPEECH)["oneOf"]}
    branch = branches.get(decision["decision"])
    assert branch is not None, decision

    for field, spec in branch["properties"].items():
        assert field in decision, f"{field} missing from {decision}"
        if "enum" in spec:
            assert decision[field] in spec["enum"], f"{field}={decision[field]!r}"
    assert set(decision) <= set(branch["properties"]), f"extra keys in {decision}"
    assert llm.calls == 1


@pytest.mark.model
def test_the_real_model_cannot_ask_the_car_to_act():
    """The safety property that comes from the grammar rather than from a check. Whatever the
    model decides, `execute` is not in the vocabulary it can emit."""
    llm = TransformersSceneLLM()
    snapshot = {"inside.rear_occupant": {"value": "child", "confidence": 0.62,
                                         "source": "cabin_cam"}}
    decision = llm.decide(snapshot, RULES, SPEECH)
    assert decision["decision"] in {"notify", "ask", "no_action"}
