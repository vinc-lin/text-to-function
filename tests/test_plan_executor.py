from t2f.plan import PlanExecutor
from t2f.state import VehicleState
from t2f.types import ActionPlan, PlannedAction, RelativeSpec
from t2f.execute import MockExecutor
from t2f.cards import load_catalog

CARDS = {c.name: c for c in load_catalog("data/catalog")}
STEPS = {"by_unit": {"percent": 10, "celsius": 1, "level": 1},
         "amount_multiplier": {"small": 1, "medium": 2, "large": 3}}

def _exec():
    return PlanExecutor(CARDS, VehicleState(), MockExecutor(), STEPS)

def test_all_valid_execute():
    plan = ActionPlan(actions=[
        PlannedAction(span="把车窗锁打开", function="set_window_child_lock",
                      parameters={"enabled": True}),
        PlannedAction(span="天窗开到一半", function="set_sunroof_position",
                      parameters={"percent": 50}),
    ])
    pe = _exec(); executed, clar = pe.finalize(plan)
    assert [a.function for a in executed] == ["set_window_child_lock", "set_sunroof_position"]
    assert clar is None
    assert all(a.status == "executed" for a in plan.actions)

def test_partial_failure_executes_valid_and_clarifies_rest():
    # relative window with NO seeded state -> clarify; the other two are valid
    plan = ActionPlan(actions=[
        PlannedAction(span="把车窗锁打开", function="set_window_child_lock",
                      parameters={"enabled": True}),
        PlannedAction(span="再开一点", function="set_window_position",
                      parameters={"position": "driver"},
                      relative=RelativeSpec("increase", "small")),
        PlannedAction(span="天窗开到一半", function="set_sunroof_position",
                      parameters={"percent": 50}),
    ])
    pe = _exec(); executed, clar = pe.finalize(plan)
    assert {a.function for a in executed} == {"set_window_child_lock", "set_sunroof_position"}
    assert clar is not None
    assert plan.actions[1].status == "clarify"

def test_nothing_executes_before_validation():
    # an invalid action must not prevent the valid ones, but must never execute itself
    plan = ActionPlan(actions=[
        PlannedAction(span="bad", function="set_sunroof_position",
                      parameters={"percent": 999}),  # out of range
    ])
    pe = _exec(); executed, clar = pe.finalize(plan)
    assert executed == [] and clar is not None
    assert plan.actions[0].status == "invalid"
