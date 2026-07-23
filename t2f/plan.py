from __future__ import annotations
from .types import ActionPlan, PlannedAction, FunctionCard
from .validate import validate_tool_call
from .state import VehicleState, StateResolver, state_key, primary_numeric_param
from .respond import build_plan_clarification


class PlanExecutor:
    """Validate the WHOLE plan, then execute only the valid subset; one consolidated
    clarification for the rest. Nothing executes until every action has been resolved+validated."""

    def __init__(self, cards_by_name: dict[str, FunctionCard], state: VehicleState,
                 executor, relative_steps: dict):
        self.cards = cards_by_name
        self.state = state
        self.executor = executor
        self.resolver = StateResolver(relative_steps)

    def finalize(self, plan: ActionPlan):
        # Phase 1: resolve + validate (NO execution)
        for a in plan.actions:
            if a.function is None or a.function not in self.cards:
                a.status = "reject"
                continue
            if a.relative is not None:
                a, err = self.resolver.resolve(a, self.state, self.cards)
                if err:
                    a.status = "clarify" if err == "missing_state" else "invalid"
                    a.error = err
                    continue
            tc, errs = validate_tool_call(a.function, a.parameters, self.cards, [a.function])
            if tc is None:
                a.status = "clarify" if any(e.code == "missing_required" for e in errs) else "invalid"
                a.error = ";".join(e.code for e in errs)
            else:
                a.tool_call = tc
                a.status = "valid"

        # Phase 2: barrier passed — execute the valid subset in order
        executed = []
        for a in plan.actions:
            if a.status == "valid":
                self.executor.execute(a.tool_call)
                p = primary_numeric_param(self.cards[a.function])
                if p is not None and p.name in a.tool_call.parameters:
                    self.state.set(state_key(a.function, a.tool_call.parameters),
                                   a.tool_call.parameters[p.name], layer="confirmed")
                a.status = "executed"
                executed.append(a)

        # Phase 3: consolidated clarification for the actionable remainder
        pending = [a for a in plan.actions if a.status in ("clarify", "invalid")]
        clar = build_plan_clarification(pending) if pending else None
        return executed, clar
