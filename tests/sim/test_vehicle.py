import json, pytest
from sim.vehicle import SqliteVehicle


@pytest.fixture
def car():
    v = SqliteVehicle(":memory:")
    v.init_schema()
    return v


def test_set_and_get_signal(car):
    car.set_signal("window.driver", "position", 40, unit="percent", limits=(0, 100))
    assert car.get_signal("window.driver", "position") == 40


def test_value_keeps_its_type(car):
    car.set_signal("climate.all", "ac_power", True)
    assert car.get_signal("climate.all", "ac_power") is True
    car.set_signal("climate.driver", "temperature", 22.5, unit="celsius")
    assert car.get_signal("climate.driver", "temperature") == 22.5


def test_missing_signal_is_none(car):
    assert car.get_signal("nope.all", "nothing") is None


def test_limits_are_readable(car):
    car.set_signal("window.driver", "position", 40, unit="percent", limits=(0, 60))
    assert car.limits_of("window.driver", "position") == (0.0, 60.0)


def test_write_many_is_atomic(car):
    car.set_signal("window.driver", "position", 10, limits=(0, 100))
    with pytest.raises(ValueError):
        car.write_many([("window.driver", "position", 50), ("bad", None, 1)])
    assert car.get_signal("window.driver", "position") == 10      # rolled back


def test_log_records_both_outcomes(car):
    car.log("open_window", {"is_open": True}, "executed", None, "")
    car.log("set_temperature", {"temperature": 25}, "refused", "precondition_failed", "空调未开启")
    rows = car.recent_operations()
    assert [r["outcome"] for r in rows] == ["refused", "executed"]   # newest first
    assert rows[0]["error"] == "precondition_failed"


def test_device_availability(car):
    assert car.is_available("window.driver") == (True, None)
    car.set_device("window.driver", False, "执行器无响应")
    assert car.is_available("window.driver") == (False, "执行器无响应")
