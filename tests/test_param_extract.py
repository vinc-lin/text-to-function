# tests/test_param_extract.py
from t2f.types import FunctionCard, ParamSpec
from t2f.lexical import extract_features
from t2f.params.extract import ParameterExtractor

def test_extract_temperature_and_missing_position():
    card = FunctionCard("set_temperature", "climate", "温度",
        params=[ParamSpec("temperature", "number", required=True, unit="celsius", minimum=16, maximum=32),
                ParamSpec("position", "enum", required=True, enum=["driver", "passenger"])])
    params, missing = ParameterExtractor().extract("把空调调到25度", extract_features("把空调调到25度"), card)
    assert params["temperature"] == 25
    assert missing == ["position"]

def test_extract_full():
    card = FunctionCard("set_temperature", "climate", "温度",
        params=[ParamSpec("temperature", "number", required=True, unit="celsius"),
                ParamSpec("position", "enum", required=True, enum=["driver", "passenger"])])
    params, missing = ParameterExtractor().extract("副驾调到22度", extract_features("副驾调到22度"), card)
    assert params == {"temperature": 22, "position": "passenger"} and missing == []
