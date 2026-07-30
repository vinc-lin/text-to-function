"""The engine may ask the car what is true. It may not tell the car anything."""
from scene.facts import VehicleFacts


class SpyCar:
    """Stands in for SqliteVehicle, and records any write it is asked to make."""

    def __init__(self, **signals):
        self._signals = signals
        self.writes = []

    def get_signal(self, entity, attribute):
        return self._signals.get((entity, attribute))

    def set_signal(self, entity, attribute, value, **kw):
        self.writes.append((entity, attribute, value))

    def write_many(self, writes):
        self.writes.extend(writes)


def test_a_known_signal_reads_through():
    car = SpyCar(**{})
    car._signals[("window.all", "window_child_lock")] = False
    assert VehicleFacts(car).signal("window.all", "window_child_lock") is False


def test_an_unknown_signal_reads_as_absent():
    """A typo'd entity must not raise — but see test_contract_sweep for why it must also not
    pass silently into a rule."""
    assert VehicleFacts(SpyCar()).signal("nope.all", "nope") is None


def test_the_car_is_not_reachable_through_the_port():
    """The docstring says read-only; this is what makes it true.

    Holding the whole SqliteVehicle would leave set_signal one attribute access away from
    any rule, and a second write path is a second set of beliefs about what the car allows —
    bypassing the availability, precondition, physical-limit and logging checks that only
    executor.execute performs.
    """
    facts = VehicleFacts(SpyCar())
    reachable = [v for v in vars(facts).values() if hasattr(v, "set_signal")]
    assert reachable == [], "a writable car is reachable from VehicleFacts"


def test_nothing_the_port_offers_can_write():
    car = SpyCar()
    facts = VehicleFacts(car)
    facts.signal("a", "b")
    assert car.writes == []
