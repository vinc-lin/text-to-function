from __future__ import annotations
from typing import Optional
from .types import FunctionCard, PlannedAction

_REL_UNITS = ("percent", "celsius", "level")


def state_key(function: str, params: dict) -> str:
    pos = params.get("position")
    return f"{function}/{pos}" if pos else function


def primary_numeric_param(card: FunctionCard):
    for p in card.params:
        if p.unit in _REL_UNITS:
            return p
    for p in card.params:
        if p.type in ("integer", "number"):
            return p
    return None


class VehicleState:
    """Injectable mock store with layered priority: live > confirmed > session default."""
    def __init__(self):
        self._layers = {"live": {}, "confirmed": {}, "session": {}}

    def reset(self, live: Optional[dict] = None):
        self._layers = {"live": dict(live or {}), "confirmed": {}, "session": {}}

    def set(self, key: str, value, layer: str = "live"):
        self._layers[layer][key] = value

    def get(self, key: str):
        for layer in ("live", "confirmed", "session"):
            if key in self._layers[layer]:
                return self._layers[layer][key]
        return None


class StateResolver:
    def __init__(self, relative_steps: dict):
        self.by_unit = relative_steps.get("by_unit", {})
        self.amount_multiplier = relative_steps.get("amount_multiplier",
                                                    {"small": 1, "medium": 2, "large": 3})

    def resolve(self, action: PlannedAction, state: VehicleState,
                cards_by_name: dict[str, FunctionCard]) -> tuple[PlannedAction, Optional[str]]:
        """Fill the absolute numeric param from current state +/- step. Returns (action, error).
        error='missing_state' -> clarify; error='no_numeric_param' -> cannot apply relative."""
        card = cards_by_name[action.function]
        p = primary_numeric_param(card)
        if p is None:
            return action, "no_numeric_param"
        key = state_key(action.function, action.parameters)
        current = state.get(key)
        if current is None:
            return action, "missing_state"
        step = self.by_unit.get(p.unit, 10) * self.amount_multiplier.get(action.relative.amount, 1)
        delta = step if action.relative.operation == "increase" else -step
        val = current + delta
        if p.minimum is not None:
            val = max(val, p.minimum)
        if p.maximum is not None:
            val = min(val, p.maximum)
        action.parameters[p.name] = int(round(val)) if p.type == "integer" else val
        return action, None
