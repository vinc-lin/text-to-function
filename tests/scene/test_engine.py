"""One observation in, at most one sentence out — and the car moves only after a yes."""
import pytest

from intake.hub import WorldView
from scene.context import Observation
from scene.engine import SceneEngine, NO_ACTION
from scene.rules import RULES
from t2f.cards import load_catalog
from t2f.types import ExecResult, ToolCall
from tests.perception import perception_store


class SpyCar:
    """Stands in for SqliteVehicle: the child lock the test asks for, and nothing else.

    Every signal it holds is freshly written (`signal_age` of 0.0), so these tests measure the
    engine rather than the freshness discipline — tests/scene/test_rules.py and
    tests/sim/test_staleness.py own that.
    """

    def __init__(self, lock=False):
        self._s = {("window.all", "window_child_lock"): lock}
        self.writes = []

    def get_signal(self, entity, attribute):
        return self._s.get((entity, attribute))

    def signal_age(self, entity, attribute, now):
        return 0.0 if (entity, attribute) in self._s else None

    def set_signal(self, *a, **k):
        self.writes.append(a)


class RecordingExecutor:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or ExecResult(ok=True)

    def execute(self, tool_call):
        self.calls.append(tool_call)
        return self.result


@pytest.fixture(scope="module")
def cards():
    return {c.name: c for c in load_catalog("data/catalog")}


def _engine(cards, *, lock=False, world=None, executor=None, rules=RULES, llm=None):
    """One store, written by the engine and read by the world.

    Built here rather than inside the engine because the world is a constructor argument and
    has to exist first. Two SceneContexts would not raise — the engine would write one and
    every rule would read the other, empty — so the engine refuses the pairing instead, and
    this helper is the one place the pairing is made.
    """
    perception = perception_store()
    return SceneEngine(cards_by_name=cards,
                       world=WorldView(perception, SpyCar(lock)) if world is None else world,
                       executor=executor or RecordingExecutor(), rules=rules, llm=llm,
                       perception=perception)


def _child(confidence=0.9, at=100.0):
    return Observation("inside.rear_occupant", "child", confidence, "cabin_cam", at, ttl=300.0)


def _report(eng, rule_id="rear_child_window_lock"):
    """The report FOR a rule, never `explain()[0]`. These tests drive the child-lock rule, and
    positional access silently started describing a different rule the moment a second one
    shipped — reading the animal warning's REJECT as the child rule's verdict."""
    return next(r for r in eng.explain() if r.rule_id == rule_id)


def test_a_matching_rule_asks_and_touches_nothing(cards):
    ex = RecordingExecutor()
    out = _engine(cards, executor=ex).observe(_child(), now=100.0)
    assert out.kind == "ask" and out.source == "rule"
    assert out.speech == "后排有小孩，要打开儿童锁吗？"
    assert out.proposal == ToolCall("set_window_child_lock", {"enabled": True})
    assert ex.calls == [], "a rule match must never reach the car on its own"


def test_an_already_locked_car_says_nothing(cards):
    out = _engine(cards, lock=True).observe(_child(), now=100.0)
    assert out == NO_ACTION


def test_consent_executes_the_proposal(cards):
    ex = RecordingExecutor()
    eng = _engine(cards, executor=ex)
    eng.observe(_child(), now=100.0)
    res = eng.resolve("好", now=105.0)
    assert res.answered and res.executed
    assert ex.calls == [ToolCall("set_window_child_lock", {"enabled": True})]
    assert res.speech == "已为您打开车窗儿童锁。"


def test_declining_executes_nothing_and_acknowledges(cards):
    ex = RecordingExecutor()
    eng = _engine(cards, executor=ex)
    eng.observe(_child(), now=100.0)
    res = eng.resolve("不用", now=105.0)
    assert res.answered and not res.executed and ex.calls == []
    assert res.speech == "好的。"


def test_a_command_drops_the_question_and_is_not_an_answer(cards):
    """The driver ignored us and said something else. The pending question is abandoned and
    the caller routes the utterance normally — it must never be read as consent."""
    ex = RecordingExecutor()
    eng = _engine(cards, executor=ex)
    eng.observe(_child(), now=100.0)
    res = eng.resolve("把窗户关上", now=105.0)
    assert not res.answered and ex.calls == []
    assert eng.pending(now=106.0) is None


def test_an_expired_question_cannot_be_answered(cards):
    ex = RecordingExecutor()
    eng = _engine(cards, executor=ex)
    eng.observe(_child(), now=100.0)
    res = eng.resolve("好", now=100.0 + eng.consent_ttl + 1.0)
    assert not res.answered and ex.calls == []


def test_the_vehicle_refusal_is_spoken_with_its_cause(cards):
    ex = RecordingExecutor(ExecResult(ok=False, error="device_unavailable", detail="车窗控制模块离线"))
    eng = _engine(cards, executor=ex)
    eng.observe(_child(), now=100.0)
    res = eng.resolve("好", now=105.0)
    assert res.answered and not res.executed
    assert res.speech == "车窗控制模块离线。"


def test_a_rule_in_cooldown_does_not_speak_twice(cards):
    eng = _engine(cards)
    assert eng.observe(_child(at=100.0), now=100.0).kind == "ask"
    eng.resolve("不用", now=101.0)
    assert eng.observe(_child(at=102.0), now=102.0) == NO_ACTION


def test_the_same_question_is_not_asked_while_one_is_pending(cards):
    eng = _engine(cards)
    assert eng.observe(_child(at=100.0), now=100.0).kind == "ask"
    assert eng.observe(_child(at=101.0), now=101.0) == NO_ACTION


def test_a_scene_stays_silent_while_the_router_holds_a_question(cards):
    """At most one open question across both systems, or 好 becomes ambiguous about which
    one it answers — and the whole consent design rests on 好 being unambiguous."""
    out = _engine(cards).observe(_child(), now=100.0, question_open=True)
    assert out == NO_ACTION


def test_an_engine_exception_degrades_to_silence(cards):
    class Exploding:
        def signal(self, *a):
            raise RuntimeError("camera bus fell over")
    out = _engine(cards, world=Exploding()).observe(_child(), now=100.0)
    assert out == NO_ACTION


# --- fixes from the adversarial pass on this module ---------------------------------------

def test_resolve_survives_an_executor_that_raises(cards):
    """The consent path is the one that touches the car. SqliteExecutor turns every modelled
    refusal into ExecResult(ok=False), so an exception here is an infrastructure fault — a
    locked database, a disk error — and that is the worst possible moment to kill a session
    with a traceback."""
    class Exploding:
        def execute(self, tool_call):
            raise RuntimeError("CAN bus fell over")
    eng = _engine(cards, executor=Exploding())
    eng.observe(_child(), now=100.0)
    res = eng.resolve("好", now=105.0)
    assert res.answered and not res.executed
    assert res.speech == "抱歉，这个操作没能完成。"
    assert eng.pending(now=106.0) is None


def test_a_blank_refusal_detail_falls_back_to_the_generic_line(cards):
    """A whitespace-only detail is truthy once a terminator is appended, so without a strip
    the driver hears a spoken full stop and no cause at all."""
    ex = RecordingExecutor(ExecResult(ok=False, error="exec_failed", detail="   "))
    eng = _engine(cards, executor=ex)
    eng.observe(_child(), now=100.0)
    assert eng.resolve("好", now=105.0).speech == "抱歉，这个操作没能完成。"


def test_no_action_cannot_be_mutated_by_a_caller(cards):
    """It is one shared instance. A single assignment on a returned no-action would poison
    every `== NO_ACTION` comparison in the process, including the ones above."""
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        NO_ACTION.speech = "毁了"


def test_a_notification_also_yields_to_an_open_router_question(cards):
    """A notify creates no pending consent, so it cannot make 好 ambiguous — but talking over
    a question the driver is being asked is still the scene subsystem interrupting."""
    import dataclasses
    from scene.rules import REAR_CHILD_WINDOW_LOCK
    notify_only = dataclasses.replace(REAR_CHILD_WINDOW_LOCK, proposes=None,
                                      intent="notify_driver_fatigue")
    eng = _engine(cards, rules=(notify_only,))
    assert eng.observe(_child(), now=100.0, question_open=True) == NO_ACTION
    # and it did not burn its cooldown while being silenced by someone else's question
    assert eng.observe(_child(at=101.0), now=101.0).kind == "notify"


# --- the LLM fallback ---------------------------------------------------------------------

from scene.llm import FakeSceneLLM


def _obs(key, value, confidence=0.9, at=100.0):
    return Observation(key, value, confidence, "cabin_cam", at, ttl=300.0)


def test_a_near_miss_reaches_the_fallback(cards):
    llm = FakeSceneLLM([{"decision": "ask", "scene": "rear_child_window_lock",
                         "reason": "低置信但语境明确", "reply_intent": "ask_rear_child_lock"}])
    eng = _engine(cards, llm=llm)
    out = eng.observe(_child(confidence=0.62), now=100.0)
    assert out.kind == "ask" and out.source == "rule"
    assert out.proposal == ToolCall("set_window_child_lock", {"enabled": True})
    assert llm.calls == 1


def test_an_unconsumed_observation_reaches_the_fallback(cards):
    """Perception reported something no rule anticipated. Silence is the alternative."""
    llm = FakeSceneLLM([{"decision": "notify", "scene": "unmatched",
                         "reason": "驾驶员疲劳", "reply_intent": "notify_driver_fatigue"}])
    eng = _engine(cards, llm=llm)
    out = eng.observe(_obs("inside.driver_attention", "drowsy"), now=100.0)
    assert out.kind == "notify" and out.source == "llm"
    assert out.speech == "您看起来有些疲劳，请注意休息。"


def test_an_unmatched_scene_may_not_ask(cards):
    """An ask needs a proposal and an unmatched scene has none, so there would be nothing for
    consent to authorise. It degrades to silence rather than asking an empty question."""
    llm = FakeSceneLLM([{"decision": "ask", "scene": "unmatched",
                         "reason": "x", "reply_intent": "ask_rear_child_lock"}])
    eng = _engine(cards, llm=llm)
    assert eng.observe(_obs("inside.driver_attention", "drowsy"), now=100.0) == NO_ACTION


def test_the_fallback_still_cannot_reach_the_car(cards):
    """The model decided to ask. The car is still only reachable through consent."""
    ex = RecordingExecutor()
    llm = FakeSceneLLM([{"decision": "ask", "scene": "rear_child_window_lock",
                         "reason": "x", "reply_intent": "ask_rear_child_lock"}])
    eng = _engine(cards, executor=ex, llm=llm)
    eng.observe(_child(confidence=0.62), now=100.0)
    assert ex.calls == []


def test_a_below_floor_observation_never_reaches_the_fallback(cards):
    llm = FakeSceneLLM([{"decision": "ask", "scene": "rear_child_window_lock",
                         "reason": "x", "reply_intent": "ask_rear_child_lock"}])
    eng = _engine(cards, llm=llm)
    assert eng.observe(_child(confidence=0.30), now=100.0) == NO_ACTION
    assert llm.calls == 0


def test_a_clear_rule_match_never_consults_the_model(cards):
    """Arbitration order is what enforces 'the LLM never overrides the rules'."""
    llm = FakeSceneLLM([{"decision": "no_action", "scene": "unmatched",
                         "reason": "x", "reply_intent": "ack_declined"}])
    eng = _engine(cards, llm=llm)
    assert eng.observe(_child(confidence=0.95), now=100.0).source == "rule"
    assert llm.calls == 0


def test_a_rejected_rule_never_consults_the_model(cards):
    """The lock is already on. A settled question must not cost a decode."""
    llm = FakeSceneLLM([{"decision": "notify", "scene": "unmatched",
                         "reason": "x", "reply_intent": "notify_driver_fatigue"}])
    eng = _engine(cards, lock=True, llm=llm)
    assert eng.observe(_child(confidence=0.62), now=100.0) == NO_ACTION
    assert llm.calls == 0


def test_the_fallback_budget_holds(cards):
    llm = FakeSceneLLM([{"decision": "notify", "scene": "unmatched", "reason": "x",
                         "reply_intent": "notify_driver_fatigue"}] * 5)
    eng = _engine(cards, llm=llm)
    eng.observe(_obs("inside.driver_attention", "drowsy", at=100.0), now=100.0)
    eng.observe(_obs("inside.driver_attention", "drowsy", at=101.0), now=101.0)
    assert llm.calls == 1, "one call per FALLBACK_COOLDOWN window"


def test_the_fallback_yields_to_an_open_router_question(cards):
    llm = FakeSceneLLM([{"decision": "notify", "scene": "unmatched", "reason": "x",
                         "reply_intent": "notify_driver_fatigue"}])
    eng = _engine(cards, llm=llm)
    assert eng.observe(_obs("inside.driver_attention", "drowsy"), now=100.0,
                       question_open=True) == NO_ACTION


def test_a_model_that_raises_degrades_to_silence(cards):
    class Exploding:
        def decide(self, *a, **k):
            raise RuntimeError("decode failed")
    eng = _engine(cards, llm=Exploding())
    assert eng.observe(_child(confidence=0.62), now=100.0) == NO_ACTION


def test_junk_from_the_model_degrades_to_silence(cards):
    for junk in (None, "not a dict", {}, {"decision": "execute", "scene": "unmatched"}):
        llm = FakeSceneLLM([junk])
        eng = _engine(cards, llm=llm)
        assert eng.observe(_child(confidence=0.62), now=100.0) == NO_ACTION, junk


def test_no_model_attached_is_simply_silence(cards):
    eng = _engine(cards, llm=None)
    assert eng.observe(_child(confidence=0.62), now=100.0) == NO_ACTION


def test_a_notify_carrying_a_question_is_refused(cards):
    """Ungrammatical under scene_decision_schema, but a scripted fake or a drifted schema
    version can still produce it — and it would put a question in the cabin that no consent
    is waiting behind."""
    llm = FakeSceneLLM([{"decision": "notify", "scene": "unmatched",
                         "reason": "x", "reply_intent": "ask_rear_child_lock"}])
    eng = _engine(cards, llm=llm)
    assert eng.observe(_obs("inside.driver_attention", "drowsy"), now=100.0) == NO_ACTION


def test_an_open_router_question_costs_no_decode(cards):
    """Silenced by someone else's question, so it should not pay for the answer either — nor
    burn the fallback window, which would swallow the next real observation."""
    llm = FakeSceneLLM([{"decision": "notify", "scene": "unmatched", "reason": "x",
                         "reply_intent": "notify_driver_fatigue"}])
    eng = _engine(cards, llm=llm)
    assert eng.observe(_obs("inside.driver_attention", "drowsy", at=100.0), now=100.0,
                       question_open=True) == NO_ACTION
    assert llm.calls == 0
    # the window is intact, so the next observation still gets its chance
    out = eng.observe(_obs("inside.driver_attention", "drowsy", at=101.0), now=101.0)
    assert out.kind == "notify" and llm.calls == 1


# --- what the engine records about its own silence ----------------------------------------

def test_explain_reports_the_verdict_of_every_rule(cards):
    eng = _engine(cards)
    eng.observe(_child(confidence=0.62), now=100.0)
    # Every rule, not just the interesting one: a rule missing from the display is a silence
    # with no explanation beside it, which is exactly what this recording exists to remove.
    assert [r.rule_id for r in eng.explain()] == [r.id for r in RULES]
    row = _report(eng)
    assert row.verdict == "near_miss" and "0.62" in row.reason


def test_explain_reports_a_cooldown_as_the_suppressor(cards):
    """The rule MATCHED and still said nothing. Without this the display would show
    'match' next to silence and read as a bug."""
    eng = _engine(cards)
    eng.observe(_child(at=100.0), now=100.0)
    eng.resolve("不用", now=101.0)
    eng.observe(_child(at=102.0), now=102.0)
    row = _report(eng)
    assert row.verdict == "match"
    assert "cooldown" in row.suppressed_by and "118" in row.suppressed_by


def test_explain_reports_a_pending_question_as_the_suppressor(cards):
    eng = _engine(cards)
    eng.observe(_child(at=100.0), now=100.0)
    eng.observe(_child(at=101.0), now=101.0)
    assert "already asked" in _report(eng).suppressed_by


def test_explain_reports_the_router_holding_a_question(cards):
    eng = _engine(cards)
    eng.observe(_child(), now=100.0, question_open=True)
    assert "router" in _report(eng).suppressed_by


def test_explain_says_nothing_suppressed_a_rule_that_spoke(cards):
    eng = _engine(cards)
    eng.observe(_child(), now=100.0)
    assert _report(eng).suppressed_by == ""


def test_explain_before_any_observation_is_empty_not_an_error(cards):
    assert _engine(cards).explain() == []


def test_explain_survives_an_engine_that_blew_up(cards):
    """An engine that fails silently and then explains nothing is the exact opacity this
    work exists to remove."""
    class Exploding:
        def signal(self, *a):
            raise RuntimeError("camera bus fell over")
    eng = _engine(cards, world=Exploding())
    assert eng.observe(_child(), now=100.0) == NO_ACTION
    assert eng.explain() and eng.explain()[0].verdict == "error"
    assert "camera bus" in eng.explain()[0].reason


def test_explain_records_the_fallback_when_it_ran(cards):
    llm = FakeSceneLLM([{"decision": "no_action", "scene": "unmatched",
                         "reason": "不确定", "reply_intent": "ack_declined"}])
    eng = _engine(cards, llm=llm)
    eng.observe(_child(confidence=0.62), now=100.0)
    assert "no_action" in eng.fallback_note() and "不确定" in eng.fallback_note()


def test_the_fallback_note_says_why_it_was_not_consulted(cards):
    eng = _engine(cards)
    eng.observe(_child(confidence=0.62), now=100.0)
    assert "no model attached" in eng.fallback_note()


def test_the_fallback_note_says_when_the_budget_blocked_it(cards):
    llm = FakeSceneLLM([{"decision": "no_action", "scene": "unmatched", "reason": "x",
                         "reply_intent": "ack_declined"}] * 3)
    eng = _engine(cards, llm=llm)
    eng.observe(_child(confidence=0.62, at=100.0), now=100.0)
    eng.observe(_child(confidence=0.62, at=101.0), now=101.0)
    assert "budget" in eng.fallback_note()


def test_a_rule_that_lost_arbitration_says_who_beat_it(cards):
    """A MATCH sitting next to silence with no reason beside it reads as a bug in the car.
    Losing to a higher-priority rule is a perfectly good reason and must be said."""
    import dataclasses
    from scene.rules import REAR_CHILD_WINDOW_LOCK
    loud = dataclasses.replace(REAR_CHILD_WINDOW_LOCK, id="loud", priority=99)
    quiet = dataclasses.replace(REAR_CHILD_WINDOW_LOCK, id="quiet", priority=1)
    eng = _engine(cards, rules=(quiet, loud))
    eng.observe(_child(), now=100.0)
    by_id = {r.rule_id: r for r in eng.explain()}
    assert by_id["loud"].suppressed_by == ""
    assert by_id["quiet"].suppressed_by == "outranked by loud"


def test_a_rule_whose_proposal_will_not_validate_says_so(cards):
    """The contract sweep keeps every shipped rule's proposal valid, so this is unreachable
    today — but it is the one remaining path where a MATCH produces silence, and someone
    editing a rule by hand is exactly who needs to be told."""
    import dataclasses
    from scene.rules import REAR_CHILD_WINDOW_LOCK
    broken = dataclasses.replace(
        REAR_CHILD_WINDOW_LOCK,
        proposes=ToolCall("set_window_child_lock", {"enabled": "yes"}))
    eng = _engine(cards, rules=(broken,))
    assert eng.observe(_child(), now=100.0) == NO_ACTION
    assert eng.explain()[0].suppressed_by == "proposal failed validation"


def test_an_earlier_suppressor_is_not_overwritten_by_a_later_one(cards):
    """question_open is recorded first and is the reason the rule was silent. Reporting the
    validation problem instead would name the second thing that would have stopped it rather
    than the first thing that did."""
    import dataclasses
    from scene.rules import REAR_CHILD_WINDOW_LOCK
    broken = dataclasses.replace(
        REAR_CHILD_WINDOW_LOCK,
        proposes=ToolCall("set_window_child_lock", {"enabled": "yes"}))
    eng = _engine(cards, rules=(broken,))
    eng.observe(_child(), now=100.0, question_open=True)
    assert "router" in eng.explain()[0].suppressed_by


def test_a_reset_keeps_the_same_perception_store(cards):
    """The engine's reset must not swap the store out from under a reader. WorldView binds
    this object's methods, so a rebind would strand it on the discarded instance."""
    eng = _engine(cards)
    before = eng.context
    eng.observe(_child(), now=100.0)
    eng.reset()
    assert eng.context is before
    assert eng.context.live(now=100.0) == {}


def test_a_reset_is_visible_through_the_world(cards):
    """The reader's half of the property above. Clearing in place is only correct if the world
    sees the emptiness; a rebind would leave it reading a discarded store that still holds the
    child, and the engine would keep asking about a car that no longer exists."""
    eng = _engine(cards)
    eng.observe(_child(), now=100.0)
    assert eng.world.observation("inside.rear_occupant", now=100.0) is not None
    eng.reset()
    assert eng.world.observation("inside.rear_occupant", now=100.0) is None


# --- one store, two roles -----------------------------------------------------------------

def test_the_engine_writes_the_store_its_world_reads(cards):
    """The pairing the whole migration rests on. The engine is perception's only writer and
    the world is a reader over it; they must be over the same object."""
    eng = _engine(cards)
    eng.observe(_child(), now=100.0)
    assert eng.world.observation("inside.rear_occupant", now=100.0).value == "child"


def test_an_engine_refuses_a_world_over_a_different_store(cards):
    """A mismatch is silent: the engine writes one context, every rule reads another, empty
    one, and the system is merely quiet with nothing raising. Refused at construction instead,
    which is the only moment the mistake is still visible."""
    with pytest.raises(ValueError, match="perception"):
        SceneEngine(cards_by_name=cards, world=WorldView(perception_store(), SpyCar()),
                    executor=RecordingExecutor(), perception=perception_store())


def test_a_world_that_cannot_answer_the_question_is_still_allowed(cards):
    """The check is duck-typed on purpose. A test stub world answering only `signal` is a
    legitimate thing to hand the engine, and demanding `reads` would force every one of them to
    become a real WorldView over a real car for no property gained."""
    class Blind:
        def signal(self, *a):
            raise RuntimeError("no")
    assert _engine(cards, world=Blind()).observe(_child(), now=100.0) == NO_ACTION
