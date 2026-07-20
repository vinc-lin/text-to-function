# tests/test_validate.py
from t2f.types import FunctionCard, ParamSpec
from t2f.validate import validate_tool_call

CARDS = {"set_temperature": FunctionCard("set_temperature", "climate", "温度",
    params=[ParamSpec("temperature", "number", required=True, minimum=16, maximum=32),
            ParamSpec("position", "enum", enum=["driver", "passenger"])])}
CAND = ["set_temperature"]

def test_valid():
    tc, errs = validate_tool_call("set_temperature", {"temperature": 25, "position": "driver"}, CARDS, CAND)
    assert errs == [] and tc.parameters["temperature"] == 25

def test_not_in_candidates():
    tc, errs = validate_tool_call("open_window", {}, CARDS, CAND)
    assert tc is None and any(e.code == "not_in_candidates" for e in errs)

def test_range_violation():
    tc, errs = validate_tool_call("set_temperature", {"temperature": 99}, CARDS, CAND)
    assert tc is None and any(e.code == "out_of_range" for e in errs)

def test_unknown_param():
    tc, errs = validate_tool_call("set_temperature", {"temperature": 25, "foo": 1}, CARDS, CAND)
    assert tc is None and any(e.code == "unknown_param" for e in errs)

def test_missing_required():
    tc, errs = validate_tool_call("set_temperature", {"position": "driver"}, CARDS, CAND)
    assert tc is None and any(e.code == "missing_required" for e in errs)

def test_bad_enum():
    tc, errs = validate_tool_call("set_temperature", {"temperature": 25, "position": "trunk"}, CARDS, CAND)
    assert tc is None and any(e.code == "bad_enum" for e in errs)
