from t2f.state import VehicleState, StateResolver, state_key, primary_numeric_param
from t2f.types import PlannedAction, RelativeSpec
from t2f.cards import load_catalog

CARDS = {c.name: c for c in load_catalog("data/catalog")}
STEPS = {"by_unit": {"percent": 10, "celsius": 1, "level": 1},
         "amount_multiplier": {"small": 1, "medium": 2, "large": 3}}

def test_relative_increase_uses_state_and_clamps():
    st = VehicleState(); st.set("set_window_position/driver", 30)
    r = StateResolver(STEPS)
    a = PlannedAction(span="再开一点", function="set_window_position",
                      parameters={"position": "driver"},
                      relative=RelativeSpec("increase", "small"))
    a2, err = r.resolve(a, st, CARDS)
    assert err is None and a2.parameters["percent"] == 40

def test_clamp_at_max():
    st = VehicleState(); st.set("set_window_position/driver", 95)
    r = StateResolver(STEPS)
    a = PlannedAction(span="再开一点", function="set_window_position",
                      parameters={"position": "driver"},
                      relative=RelativeSpec("increase", "small"))
    a2, err = r.resolve(a, st, CARDS)
    assert a2.parameters["percent"] == 100

def test_missing_state_returns_clarify():
    st = VehicleState()
    r = StateResolver(STEPS)
    a = PlannedAction(span="再开一点", function="set_window_position",
                      parameters={"position": "driver"},
                      relative=RelativeSpec("increase", "small"))
    _, err = r.resolve(a, st, CARDS)
    assert err == "missing_state"

def test_state_priority_live_over_confirmed():
    st = VehicleState()
    st.set("set_volume", 3, layer="confirmed")
    st.set("set_volume", 5, layer="live")
    assert st.get("set_volume") == 5
