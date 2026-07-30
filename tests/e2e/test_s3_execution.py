"""S3 — execution. What actually gets dispatched, and what the barrier refuses to dispatch."""
from .conftest import build_pipeline
from .doubles import FailingExecutor

WINDOW = "已为您打开当前区域车窗。"
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
    assert result.reply == WINDOW + "目标温度只能设置在16到32度之间。"
    # The refused action is named by its CAUSE now, not by echoing the span back.
    # Echoing told the driver only that we were unsure; the cause tells them what to say.


# S3-06 (an out-of-range value alone dispatches nothing) is owned by
# test_s4b_failure_cause.py::test_s4b_invalid_value_dispatches_nothing, which covers the
# same property across three utterances. Not restated here.

def test_s3_08_failed_actuation_is_still_attempted():
    """GREEN — safety half, kept separate from the confirmation half below.

    The two properties fail for different reasons and must be able to go red
    independently: this one guards the dispatch, the next one guards the reply.
    Same split as test_s4b_06/07.
    """
    pipe, ex = build_pipeline(executor=FailingExecutor())
    pipe.route("把空调调到25度")
    assert ex.dispatched == [("set_temperature", {"temperature": 25.0})]


def test_s3_08_failed_actuation_is_not_confirmed_as_success():
    """The single most important case in the suite, and now GUARANTEED: the router reads
    ExecResult, so a refused operation cannot be rendered as a confirmation. The day a real
    vehicle adapter is attached, this is the bug that would otherwise tell the driver the
    car did something it did not.

    Legacy (single-clause) path — DeterministicResolver.resolve.
    """
    pipe, _ = build_pipeline(executor=FailingExecutor())
    assert pipe.route("把空调调到25度").reply != "已将当前区域温度设置为25°C。"


def test_s3_08_the_vehicles_own_words_reach_the_driver():
    """Requirement 4b's third branch, wired end-to-end: the refusal detail travels
    executor → ExecResult.detail → ClauseResult.exec_error → result.reply.

    tests/test_reply_exec.py proves compose_reply speaks a detail it is handed; only a real
    route() proves it is handed one at all. Asserting against `ex.detail` — the double's own
    string, not a constant restated here — means no hard-coded line in reply.py could pass.
    """
    ex = FailingExecutor(error="precondition_failed", detail="空调尚未开启")
    pipe, _ = build_pipeline(executor=ex)
    result = pipe.route("把空调调到25度")
    assert ex.dispatched == [("set_temperature", {"temperature": 25.0})]   # it really was tried
    assert result.reply == ex.detail + "。"                                 # ...and refused, out loud
    assert TEMP25 not in result.reply       # never the confirmation for the action that failed
    assert "没能完成" not in result.reply    # the specific cause REPLACES the generic line


def test_s3_09_failed_action_does_not_commit_vehicle_state():
    """GUARANTEED: PlanExecutor phase 2 commits the confirmed state layer only AFTER the
    vehicle reports success, so a refusal leaves the state a later relative command
    ("再高一点") resolves against untouched rather than believing a fiction."""
    pipe, ex = build_pipeline(executor=FailingExecutor())
    pipe.route("开车窗,温度调到25度")
    assert pipe.state.get("set_temperature") is None


def test_s3_10_failed_action_on_the_plan_path_is_not_confirmed_as_success():
    """The plan path is a SEPARATE call site from the legacy one (PlanExecutor, not
    DeterministicResolver), so it needs its own proof that a refusal is not spoken as success.

    Also pins the `_route_plan` branch ORDER: a `failed` action must be read as an
    exec_error, not fall through to the generic clarification branch below it.
    """
    pipe, ex = build_pipeline(executor=FailingExecutor())
    result = pipe.route("开车窗,温度调到25度")
    # both actions were dispatched and both were refused
    assert ex.dispatched == [("open_window", {"is_open": True}),
                             ("set_temperature", {"temperature": 25.0})]
    assert [a.status for a in result.plan.actions] == ["failed", "failed"]
    # nothing the car did not do is spoken
    assert WINDOW not in result.reply and TEMP25 not in result.reply
    assert all(cr.response is None for cr in result.clauses)
    # a refusal is an exec_error, NOT a clarification (guards the branch order)
    assert all(cr.exec_error is not None for cr in result.clauses)
    assert all(cr.clarification is None for cr in result.clauses)
    # ...and no state was committed for either refused action
    assert pipe.state.get("set_temperature") is None
    assert pipe.state.get("open_window") is None


def test_s3_11_a_refusal_beside_an_invalid_action_is_not_reported_as_a_clarification():
    """The sharp edge of the `_route_plan` branch order, in the one shape where getting it
    wrong actively misinforms the driver.

    温度调到99度 is out of range, so the plan carries a real consolidated clarification.
    开车窗 validates, is dispatched, and the car refuses it. If the `failed` branch sat BELOW
    the generic `elif a is not None:` branch, the refused window action would be handed that
    OTHER action's clarification question — the driver would be asked to clarify a request
    that was understood perfectly and rejected by the vehicle, and the refusal itself would
    be erased before the reply layer ever saw it.
    """
    pipe, ex = build_pipeline(executor=FailingExecutor())
    result = pipe.route("开车窗,把温度调到99度")
    assert ex.dispatched == [("open_window", {"is_open": True})]   # only the valid one is tried
    window, temperature = result.plan.actions
    assert (window.status, temperature.status) == ("failed", "invalid")
    window_clause = result.clauses[0]
    assert window_clause.exec_error is not None
    assert window_clause.exec_error.code == "device_unavailable"
    assert window_clause.clarification is None       # NOT the 99度 question
    assert window_clause.response is None            # and never a confirmation
    assert WINDOW not in result.reply
