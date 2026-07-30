"""Read-only access to the car, for rules that condition on vehicle state.

Deliberately read-only. The engine may ask the car what is true and may not write: every
write goes through executor.execute so it gets validation, preconditions, physical limits and
an operation-log entry. A second write path would be a second set of rules about what the car
allows, and the car is the authority on that.
"""
from __future__ import annotations
from typing import Any, Optional


class VehicleFacts:
    def __init__(self, car):
        # Bind the reader, not the car. Holding the whole SqliteVehicle would leave
        # `set_signal` one attribute access away from any rule, which is the write path the
        # docstring above says must not exist — and a docstring is not an enforcement
        # mechanism. There is nothing here to write through.
        self._read = car.get_signal

    def signal(self, entity: str, attribute: str) -> Optional[Any]:
        return self._read(entity, attribute)
