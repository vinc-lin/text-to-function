"""The fallback picks from what exists. It cannot invent a scene, a sentence, or an action."""
import pytest

from scene.llm import FakeSceneLLM, scene_decision_schema
from scene.rules import RULES
from scene.speech import SPEECH


def _branches(rules=RULES, speech=SPEECH):
    return {b["properties"]["decision"]["const"]: b
            for b in scene_decision_schema(rules, speech)["oneOf"]}


def test_the_schema_offers_no_execute_decision():
    """The model cannot ask for the car to move. The most it can do is propose a question,
    and that still needs consent."""
    assert set(_branches()) == {"notify", "ask", "no_action"}


def test_a_notify_cannot_carry_a_question():
    """The hole a flat schema left open: decision=notify with an ask_* intent puts a question
    in the cabin with no pending consent behind it, and the driver answers into the void.
    Branching on the decision makes it ungrammatical rather than merely checked."""
    notify = _branches()["notify"]["properties"]["reply_intent"]["enum"]
    assert notify and all(not i.startswith("ask_") for i in notify)


def test_only_a_rule_that_can_be_acted_on_may_be_asked_about():
    """An ask names a rule whose consent has something to authorise. A rule with no proposal
    is not askable, and that is now a property of the grammar rather than an if-statement."""
    askable = set(_branches()["ask"]["properties"]["scene"]["enum"])
    assert askable == {r.id for r in RULES if r.proposes is not None}
    assert "unmatched" not in askable
    # The converse, which only became testable with a notify-only rule in the set: a warning
    # proposes nothing, so asking about it would open a question no answer could act on.
    assert "animal_ahead" not in askable


def test_a_notify_only_rule_is_notifiable_and_is_not_dead_grammar():
    """`notifiable` held nothing but `unmatched` until the first notify-only rule shipped, so
    the branch could name a scene the model could never usefully select. A rule that warns
    must be reachable by name, or its speech is unsayable through the fallback."""
    notifiable = set(_branches()["notify"]["properties"]["scene"]["enum"])
    assert notifiable == {r.id for r in RULES if r.proposes is None} | {"unmatched"}
    assert "animal_ahead" in notifiable


def test_no_action_needs_no_intent_because_it_says_nothing():
    branch = _branches()["no_action"]
    assert "reply_intent" not in branch["properties"]
    assert set(branch["required"]) == {"decision", "scene", "reason"}


def test_every_intent_in_the_table_is_reachable_from_some_branch():
    """If an intent exists that no branch can select, it is dead speech — and if a branch can
    select one that is not in the table, it resolves to silence. Both are drift."""
    reachable = {i for b in _branches().values()
                 for i in b["properties"].get("reply_intent", {}).get("enum", [])}
    assert reachable == {k for k in SPEECH if not k.startswith("ack_")}


def test_no_branch_allows_a_field_it_did_not_name():
    for kind, branch in _branches().items():
        assert branch["additionalProperties"] is False, kind
        assert set(branch["required"]) <= set(branch["properties"]), kind


def test_the_reason_is_bounded_by_the_grammar():
    """reason shares a token budget with the fields that decide the outcome; a string cut off
    mid-generation makes the whole object unparseable and discards the decision."""
    assert all(b["properties"]["reason"]["maxLength"] == 80 for b in _branches().values())


def test_a_rule_set_with_nothing_askable_drops_the_ask_branch():
    """An empty enum is not valid JSON schema, so the branch must disappear rather than be
    emitted in a form no decoder can satisfy.

    Named explicitly rather than taken as `RULES[0]`: the set is ordered by priority now, so
    index 0 is `animal_ahead`, which already proposes nothing — stripping a proposal it never
    had would leave this asserting on a rule that was never askable in the first place."""
    import dataclasses
    from scene.rules import REAR_CHILD_WINDOW_LOCK
    assert REAR_CHILD_WINDOW_LOCK.proposes is not None, "the premise of this test"
    notify_only = dataclasses.replace(REAR_CHILD_WINDOW_LOCK, proposes=None)
    assert "ask" not in _branches(rules=(notify_only,))


def test_the_fake_returns_its_script():
    llm = FakeSceneLLM([{"decision": "notify", "scene": "unmatched",
                         "reason": "r", "reply_intent": "notify_driver_fatigue"}])
    assert llm.decide({}, RULES, SPEECH)["decision"] == "notify"


def test_the_fake_returns_none_when_the_script_runs_out():
    """Exhausted script means no decision, which the engine must read as silence."""
    assert FakeSceneLLM([]).decide({}, RULES, SPEECH) is None
