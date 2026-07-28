"""The executor seam, backed by the simulated car.

Order matters: availability, then preconditions, then physical limits, then write. A refusal
must leave the car exactly as it was, and every attempt is logged either way -- the log is the
only place from which "we tried and the car said no" can be recovered.

The physical-limit check is the reason this class can say something validation cannot. The
catalog's range is what a *function* accepts; the signal's range is what *this* actuator can
reach. They are allowed to disagree, and when they do the car wins.
"""
from __future__ import annotations

from t2f.types import ExecResult, FunctionCard, ToolCall
from t2f.state import primary_numeric_param, state_key
from t2f.phrase import limit_phrase
from sim.mapping import resolve_writes, signal_for_function
# One definition of "the positions this card can be called with", shared with the seeder:
# if the snapshot invented positions the seeder never created it would read empty rows, and
# if it dropped one the live state layer would be missing a signal the car really has.
from sim.seed import _positions
from sim.vehicle import SqliteVehicle


def _fmt(value: float) -> str:
    """Limits are stored as REAL; 60.0 must be spoken as 60, 22.5 as 22.5."""
    return str(int(value)) if float(value).is_integer() else str(value)




class SqliteExecutor:
    def __init__(self, car: SqliteVehicle, cards_by_name: dict[str, FunctionCard]):
        self.car = car
        self.cards = cards_by_name

    def execute(self, tool_call: ToolCall) -> ExecResult:
        card = self.cards.get(tool_call.name)
        if card is None:
            return self._refuse(tool_call, "unknown_function", f"{tool_call.name} 不在功能表中")

        writes = resolve_writes(card, tool_call)
        if not writes:
            # Nothing addressable (lock_doors, pause_playback, ...): a no-op still succeeds,
            # and is still logged, because it was still attempted.
            self.car.log(tool_call.name, tool_call.parameters, "executed", None, "")
            return ExecResult(ok=True)

        entity = writes[0][0]
        available, reason = self.car.is_available(entity)
        if not available:
            return self._refuse(tool_call, "device_unavailable", reason or f"{entity} 当前不可用")

        for pre in self.car.preconditions_for(card.name):
            if self.car.get_signal(pre["entity"], pre["attribute"]) != pre["equals"]:
                return self._refuse(tool_call, "precondition_failed", pre["detail"])

        for ent, attr, value in writes:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            lo, hi = self.car.limits_of(ent, attr)
            if lo is not None and value < lo:
                return self._refuse(tool_call, "out_of_range",
                                    limit_phrase(card, primary_numeric_param(card), lo, "最低",
                                                 low=lo, high=hi))
            if hi is not None and value > hi:
                return self._refuse(tool_call, "out_of_range",
                                    limit_phrase(card, primary_numeric_param(card), hi, "最高",
                                                 low=lo, high=hi))

        self.car.write_many(writes)
        self.car.log(tool_call.name, tool_call.parameters, "executed", None, "")
        return ExecResult(ok=True)

    def _refuse(self, tool_call: ToolCall, error: str, detail: str) -> ExecResult:
        self.car.log(tool_call.name, tool_call.parameters, "refused", error, detail)
        return ExecResult(ok=False, error=error, detail=detail)

    def snapshot(self) -> dict:
        """Current car state keyed the way VehicleState/StateResolver expect."""
        out: dict = {}
        for card in self.cards.values():
            if primary_numeric_param(card) is None:
                continue                      # only numeric signals answer "再开一点"
            for position in _positions(card):
                sig = signal_for_function(card, position)
                if sig is None:
                    continue
                value = self.car.get_signal(*sig)
                if value is None:
                    continue
                params = {"position": position} if position else {}
                out[state_key(card.name, params)] = value
        return out
