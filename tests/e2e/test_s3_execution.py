"""S3 — execution. What actually gets dispatched, and what the barrier refuses to dispatch."""
import pytest

from .conftest import build_pipeline
from .doubles import FailingExecutor

WINDOW = "已为您调整当前区域车窗状态。"
TEMP25 = "已将当前区域温度设置为25°C。"


def test_s3_01_single_valid_dispatches_exactly_once():
    pipe, ex = build_pipeline()
    pipe.route("把空调调到25度")
    assert ex.dispatched == [("set_temperature", {"temperature": 25.0})]


def test_s3_02_multi_intent_dispatches_in_order():
    pipe, ex = build_pipeline()
    pipe.route("开车窗,温度调到25度")
    assert ex.dispatched == [("open_window", {"is_open": True}),
                             ("set_temperature", {"temperature": 25.0})]


def test_s3_03_barrier_executes_only_the_valid_subset():
    """One action is out of range. The barrier must dispatch the valid one and ONLY that."""
    pipe, ex = build_pipeline()
    pipe.route("开车窗,把温度调到99度")
    assert ex.dispatched == [("open_window", {"is_open": True})]


def test_s3_04_barrier_names_the_unexecuted_action_in_the_reply():
    """The refused action must not vanish — the driver has to hear that it did not happen."""
    pipe, ex = build_pipeline()
    result = pipe.route("开车窗,把温度调到99度")
    assert result.reply == WINDOW + "关于「把温度调到99度」我还需要确认一下，请补充信息。"
    assert "把温度调到99度" in result.reply


# S3-06 (an out-of-range value alone dispatches nothing) is owned by
# test_s4b_failure_cause.py::test_s4b_invalid_value_dispatches_nothing, which covers the
# same property across three utterances. Not restated here.

GAP1 = ("gap 1: execute()'s return value is discarded at all four call sites "
        "(t2f/plan.py:43, t2f/pipeline.py:64,104, t2f/dialog.py:42) and no dataclass "
        "carries a vehicle-reported outcome, so a failed actuation is spoken as success.")


def test_s3_08_failed_actuation_is_still_attempted():
    """GREEN — safety half, kept OUT of the xfail body below.

    xfail(strict) reports the same result whichever assertion trips, so pairing this
    currently-true check with the currently-false one would absorb a future regression
    here into the expected failure. Same split as test_s4b_06/07.
    """
    pipe, ex = build_pipeline(executor=FailingExecutor())
    pipe.route("把空调调到25度")
    assert ex.dispatched == [("set_temperature", {"temperature": 25.0})]


@pytest.mark.xfail(strict=True, reason=GAP1)
def test_s3_08_failed_actuation_is_not_confirmed_as_success():
    """The single most important case in the suite: the day a real vehicle adapter is
    attached, this is the bug that would tell the driver the car did something it did not."""
    pipe, _ = build_pipeline(executor=FailingExecutor())
    assert pipe.route("把空调调到25度").reply != "已将当前区域温度设置为25°C。"


@pytest.mark.xfail(strict=True, reason=GAP1)
def test_s3_09_failed_action_does_not_commit_vehicle_state():
    """t2f/plan.py:44-48 writes the confirmed state layer before knowing the call succeeded."""
    pipe, ex = build_pipeline(executor=FailingExecutor())
    pipe.route("开车窗,温度调到25度")
    assert pipe.state.get("set_temperature") is None
