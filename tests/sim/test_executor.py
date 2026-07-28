"""The executor seam, backed by the simulated car.

The point of these cases is the one thing validation cannot do: **refuse**. A card says a
window takes 0-100; the window in front of the driver may be jammed at 60, the A/C may be off,
the actuator may not answer. Each refusal must leave the car exactly as it was and still land
in the operation log, because a refusal the router cannot see is spoken as a success.

Signal addresses here are the ones `sim.mapping.resolve_writes` really produces
(`window_position`, not `position` -- the sunroof lives in the `window` domain too).
"""
import pytest

from t2f.cards import load_catalog
from t2f.types import ToolCall
from t2f.validate import validate_tool_call
from t2f.state import primary_numeric_param
from sim.vehicle import SqliteVehicle
from sim.seed import seed_from_catalog
from sim.executor import SqliteExecutor

CARDS = load_catalog("data/catalog")
BY = {c.name: c for c in CARDS}


@pytest.fixture
def ex():
    car = SqliteVehicle(":memory:")
    car.init_schema()
    seed_from_catalog(car, CARDS)
    return SqliteExecutor(car, BY)


def _jam(car, entity, attribute, lo, hi):
    """Give one signal a physical range tighter than any card's.

    Not `set_signal(..., limits=...)`: the row already exists and set_signal's ON CONFLICT
    clause deliberately keeps the first writer's limits (see sim/seed.py -- an aliased
    boolean card must not erase the numeric card's range). A jammed window is a property of
    the hardware, so it is written straight onto the row.
    """
    car.conn.execute(
        "UPDATE signal SET min_value=?, max_value=? WHERE entity=? AND attribute=?",
        (lo, hi, entity, attribute))
    car.conn.commit()


# --- an operation actually moves the car -------------------------------------------------

def test_operation_changes_state(ex):
    r = ex.execute(ToolCall("set_temperature", {"temperature": 25, "position": "driver"}))
    assert r.ok
    assert ex.car.get_signal("climate.driver", "temperature") == 25


def test_aliasing_functions_move_the_same_window(ex):
    ex.execute(ToolCall("set_window_position", {"percent": 40, "position": "driver"}))
    assert ex.car.get_signal("window.driver", "window_position") == 40
    ex.execute(ToolCall("open_window", {"is_open": False, "position": "driver"}))
    assert ex.car.get_signal("window.driver", "window_position") == 0


# --- the three refusals -------------------------------------------------------------------

def test_precondition_refusal(ex):
    ex.car.set_signal("climate.all", "ac_power", False)
    r = ex.execute(ToolCall("set_temperature", {"temperature": 25, "position": "driver"}))
    assert not r.ok and r.error == "precondition_failed" and "空调" in r.detail


def test_refusal_does_not_change_state(ex):
    before = ex.car.get_signal("climate.driver", "temperature")
    ex.car.set_signal("climate.all", "ac_power", False)
    ex.execute(ToolCall("set_temperature", {"temperature": 25, "position": "driver"}))
    assert ex.car.get_signal("climate.driver", "temperature") == before


def test_device_unavailable_refusal(ex):
    ex.car.set_device("window.driver", False, "执行器无响应")
    r = ex.execute(ToolCall("open_window", {"is_open": True, "position": "driver"}))
    assert not r.ok and r.error == "device_unavailable"
    assert r.detail == "执行器无响应"


def test_physical_limit_tighter_than_the_card(ex):
    """The card says 0-100. This window is jammed at 60. Validation cannot see that."""
    _jam(ex.car, "window.driver", "window_position", 0, 60)
    ex.car.set_signal("window.driver", "window_position", 10)

    call = ToolCall("set_window_position", {"percent": 90, "position": "driver"})
    valid, errs = validate_tool_call(call.name, call.parameters, BY, [call.name])
    assert valid is not None and errs == [], "the catalog accepts 90; only the car can refuse it"

    r = ex.execute(call)
    assert not r.ok and r.error == "out_of_range"
    # The WHOLE sentence, not just the number. Asserting `"60" in r.detail` was what let
    # `window_position 最高只能到 60` — an internal signal address — reach a driver's ears.
    # The whole permitted range, not just the bound that was broken: a driver who asked
    # for 90 can act on "0 to 60" and has to guess after "at most 60".
    assert r.detail == "车窗开度只能设置在0到60%之间"
    assert ex.car.get_signal("window.driver", "window_position") == 10


def test_no_refusal_ever_speaks_an_internal_signal_name(ex):
    """The durable guard. A refusal detail is spoken verbatim to the driver, so it may never
    contain a signal attribute (`window_position`), an entity (`climate.driver`), or a raw
    function name — those are addresses, not words. Sweeps every refusable numeric param in
    the real catalog rather than the one case that happened to be noticed.
    """
    from sim.mapping import resolve_writes
    leaks = []
    for card in BY.values():
        param = primary_numeric_param(card)
        if param is None or param.maximum is None:
            continue
        params = {"position": "driver"} if card.param("position") else {}
        params[param.name] = param.maximum + 1_000          # certain to exceed any limit
        writes = resolve_writes(card, ToolCall(card.name, params))
        if not writes:
            continue
        entity, attribute, _ = writes[0]
        _jam(ex.car, entity, attribute, param.minimum or 0, param.maximum or 1)
        r = ex.execute(ToolCall(card.name, params))
        if not r.ok and r.error == "out_of_range":
            for address in (attribute, entity, card.name, param.name):
                if address in r.detail:
                    leaks.append((card.name, address, r.detail))
    assert leaks == [], f"internal addresses spoken to the driver: {leaks}"


# --- refusals are whole: no state, and a log row anyway -----------------------------------

def test_every_attempt_is_logged(ex):
    ex.execute(ToolCall("set_temperature", {"temperature": 25, "position": "driver"}))
    ex.car.set_device("window.driver", False, "x")
    ex.execute(ToolCall("open_window", {"is_open": True, "position": "driver"}))
    outcomes = [r["outcome"] for r in ex.car.recent_operations()]
    assert outcomes == ["refused", "executed"]


def test_a_refusal_skips_the_write_and_still_logs(ex):
    """Both halves at once. A refusal that logs but writes anyway, or holds the state back
    but logs nothing, passes the two cases above separately and is still broken."""
    before = ex.car.get_signal("climate.driver", "temperature")
    ex.car.set_signal("climate.all", "ac_power", False)

    r = ex.execute(ToolCall("set_temperature", {"temperature": 25, "position": "driver"}))

    assert not r.ok
    assert ex.car.get_signal("climate.driver", "temperature") == before
    rows = ex.car.recent_operations()
    assert [(x["outcome"], x["error"], x["function"]) for x in rows] == [
        ("refused", "precondition_failed", "set_temperature")]


# --- the live state layer finally has a producer ------------------------------------------

def test_snapshot_feeds_state_resolver(ex):
    ex.execute(ToolCall("set_temperature", {"temperature": 25, "position": "driver"}))
    assert ex.snapshot()["set_temperature/driver"] == 25


def test_snapshot_keys_are_only_ones_a_lookup_can_ask_for(ex):
    """Positions come from the card, not from a fixed driver/passenger/rear/all list. A card
    with no position parameter is only ever addressed as itself, so 'set_volume/all' would be
    a key in the live layer that no StateResolver lookup can ever match."""
    snap = ex.snapshot()
    assert snap["set_volume"] is not None
    assert "set_volume/all" not in snap
    assert "set_temperature/all" in snap        # this card really does accept position=all
