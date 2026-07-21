from t2f.types import FunctionCard, ParamSpec
from t2f.llm.schema import candidates_to_json_schema

def _cards():
    return [
        FunctionCard("set_temperature", "climate", "温度",
            params=[ParamSpec("temperature", "number", required=True, minimum=16, maximum=32),
                    ParamSpec("position", "enum", enum=["driver", "passenger"])]),
        FunctionCard("set_fan_speed", "climate", "风速",
            params=[ParamSpec("level", "integer", required=True, minimum=1, maximum=7)]),
    ]

def test_oneof_over_candidates():
    s = candidates_to_json_schema(_cards())
    assert "oneOf" in s and len(s["oneOf"]) == 2
    opt = next(o for o in s["oneOf"] if o["properties"]["name"]["const"] == "set_temperature")
    props = opt["properties"]["parameters"]["properties"]
    assert props["temperature"] == {"type": "number", "minimum": 16, "maximum": 32}
    assert props["position"] == {"enum": ["driver", "passenger"]}
    assert opt["properties"]["parameters"]["required"] == ["temperature"]
    assert opt["properties"]["parameters"]["additionalProperties"] is False

def test_single_card_no_oneof():
    s = candidates_to_json_schema(_cards()[:1])
    assert "oneOf" not in s and s["properties"]["name"]["const"] == "set_temperature"
