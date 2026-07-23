from t2f.types import SpanRole, Span, RelativeSpec, PlannedAction, ActionPlan, RouteResult, LexFeatures

def test_span_defaults():
    s = Span(text="开空调", role=SpanRole.ACTION)
    assert s.attached_context == []

def test_planned_action_and_plan():
    a = PlannedAction(span="天窗开到一半", function="set_sunroof_position",
                      parameters={"percent": 50})
    assert a.status == "pending" and a.relative is None and a.tool_call is None
    rel = RelativeSpec(operation="increase", amount="small")
    b = PlannedAction(span="再开一点", function="set_window_position",
                      parameters={"position": "driver"}, relative=rel)
    plan = ActionPlan(actions=[a, b], source="deterministic")
    assert len(plan.actions) == 2 and plan.source == "deterministic"

def test_routeresult_has_plan_and_lexfeatures_amount():
    rr = RouteResult(utterance="x")
    assert rr.plan is None and rr.clauses == []
    assert LexFeatures().amount is None
