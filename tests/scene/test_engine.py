"""One observation in, at most one sentence out — and the car moves only after a yes."""
import pytest

from scene.context import Observation
from scene.engine import SceneEngine, NO_ACTION
from scene.rules import RULES
from t2f.cards import load_catalog
from t2f.types import ExecResult, ToolCall


class FakeFacts:
    def __init__(self, lock=False):
        self.lock = lock

    def signal(self, entity, attribute):
        return self.lock if (entity, attribute) == ("window.all", "window_child_lock") else None


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


def _engine(cards, facts=None, executor=None):
    return SceneEngine(cards_by_name=cards, facts=facts or FakeFacts(),
                       executor=executor or RecordingExecutor(), rules=RULES)


def _child(confidence=0.9, at=100.0):
    return Observation("inside.rear_occupant", "child", confidence, "cabin_cam", at, ttl=300.0)


def test_a_matching_rule_asks_and_touches_nothing(cards):
    ex = RecordingExecutor()
    out = _engine(cards, executor=ex).observe(_child(), now=100.0)
    assert out.kind == "ask" and out.source == "rule"
    assert out.speech == "后排有小孩，要打开儿童锁吗？"
    assert out.proposal == ToolCall("set_window_child_lock", {"enabled": True})
    assert ex.calls == [], "a rule match must never reach the car on its own"


def test_an_already_locked_car_says_nothing(cards):
    out = _engine(cards, facts=FakeFacts(lock=True)).observe(_child(), now=100.0)
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
    out = _engine(cards, facts=Exploding()).observe(_child(), now=100.0)
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
    eng = SceneEngine(cards_by_name=cards, facts=FakeFacts(), executor=RecordingExecutor(),
                      rules=(notify_only,))
    assert eng.observe(_child(), now=100.0, question_open=True) == NO_ACTION
    # and it did not burn its cooldown while being silenced by someone else's question
    assert eng.observe(_child(at=101.0), now=101.0).kind == "notify"
