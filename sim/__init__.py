"""The simulated car: a peer of `t2f/`, injected through the existing `execute()` seam and
swapped for a real bus adapter on a vehicle. `t2f/` imports nothing from here."""
from sim.executor import SqliteExecutor
from sim.vehicle import SqliteVehicle

__all__ = ["SqliteVehicle", "SqliteExecutor"]
