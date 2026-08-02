"""A perception store for tests: a real `SceneContext` over a real database.

`SceneContext(store)` needs a store, the queries live in `intake` and the schema in `sim`, so
the honest way to build one is the car that owns the connection — which is exactly what
`cli/session.py` does. A hand-written fake store would be a second implementation of
`newest_perception`, i.e. a second answer to the newest-then-liveness question that §5 of the
store design exists to protect, held by the tests rather than by the code, and free to be right
while the code is wrong.

Fresh and in memory per call. A shared one would carry beliefs between tests — the same
contamination `eval/run_scene_eval.py` builds a fresh car per row to avoid.
"""
from intake.store import Store
from scene.context import SceneContext
from sim.vehicle import SqliteVehicle


def perception_store() -> SceneContext:
    car = SqliteVehicle(":memory:")
    car.init_schema()
    # The car is not returned and nothing else holds it. That is fine, and only because
    # `SqliteVehicle` has no `__del__`: an in-memory database lives as long as its connection
    # is open, the store holds that connection, and closing happens only on an explicit
    # `car.close()` that nothing here calls.
    return SceneContext(Store(car.conn))
