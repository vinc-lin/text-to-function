"""The whole chain, against a real simulated car."""
import pytest

from cli.session import Session


@pytest.fixture
def session():
    return Session.build(fake=True, llm=False, gate="permissive")


def test_a_scene_locks_the_windows_and_the_car_then_refuses_to_open_them(session):
    """The interaction this slice exists to demonstrate. A proactive action changes what a
    later driver-initiated command is allowed to do, and the refusal is explained — both
    entry points, all four workflow steps, one car.

    The precondition is already in the seeded car: open_window requires
    window_child_lock == False and refuses with 车窗儿童锁已开启 (sim/seed.py:38).
    """
    assert session.observe("rear_occupant", "child", 0.9).reply == "后排有小孩，要打开儿童锁吗？"
    assert session.handle("好").reply == "已为您打开车窗儿童锁。"
    assert ("window.all", "window_child_lock", True) in session.changed_signals()
    assert "车窗儿童锁已开启" in session.handle("开车窗").reply


def test_declining_leaves_the_car_untouched(session):
    session.observe("rear_occupant", "child", 0.9)
    assert session.handle("不用").reply == "好的。"
    assert session.changed_signals() == []


def test_a_second_event_after_consent_says_nothing(session):
    """The car stays quiet after the thing was done — silence, not a repeat.

    Deliberately does NOT claim which mechanism produced the quiet: the Signal condition now
    rejects AND the 120s cooldown is still running, so mutating either one alone leaves this
    passing. That is fine here — what a driver cares about is not being asked twice, whatever
    suppressed it. The Signal mechanism is isolated properly in test_contract_sweep.py's
    test_a_rule_stays_silent_when_its_signal_condition_already_holds, where removing it is
    the only thing that fails.
    """
    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    assert session.observe("rear_occupant", "child", 0.9).reply == ""


def test_a_weak_detection_says_nothing_without_a_model(session):
    assert session.observe("rear_occupant", "child", 0.55).reply == ""


def test_the_operation_log_records_who_asked(session):
    """A scene-initiated action is logged exactly like a driver-initiated one — the log is
    the only place 'we tried and the car said no' can be recovered from."""
    session.observe("rear_occupant", "child", 0.9)
    session.handle("好")
    recent = session.car.recent_operations(limit=5)
    assert any(op["function"] == "set_window_child_lock" and op["outcome"] == "executed"
               for op in recent)
