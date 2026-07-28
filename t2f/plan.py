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
                # Keep the errors themselves, not just their codes. The driver-facing detail
                # lives on the ValidationError, and joining codes into a string threw away the
                # only thing 4b can say — so a bad value inside a multi-intent utterance used
                # to degrade to "I need to check about 「…」" while the same value alone
                # explained itself.
                a.validation_errors = list(errs)
            else:
                a.tool_call = tc
                a.status = "valid"

        # Phase 2: barrier passed — execute the valid subset in order.
        # The vehicle gets the last word: a refused action is NOT executed, so it neither
        # commits the confirmed state layer nor reaches the reply layer as a confirmation.
        executed = []
        for a in plan.actions:
            if a.status == "valid":
                res = self.executor.execute(a.tool_call)
                if not res.ok:
                    a.status = "failed"
                    a.error = res.error or "exec_failed"
                    a.detail = res.detail
                    continue
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
