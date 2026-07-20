# tests/test_param_extractors.py
from t2f.types import ParamSpec
from t2f.lexical import extract_features
from t2f.params.extractors import (extract_temperature, extract_level,
                                    extract_position, extract_boolean)

def test_temperature():
    f = extract_features("把空调调到25度")
    assert extract_temperature("把空调调到25度", f,
        ParamSpec("temperature", "number", unit="celsius", minimum=16, maximum=32)) == 25

def test_level():
    f = extract_features("风速调到三档")
    assert extract_level("风速调到三档", f, ParamSpec("level", "integer", unit="level")) == 3

def test_position_maps_to_enum():
    f = extract_features("副驾这边")
    assert extract_position("副驾这边", f,
        ParamSpec("position", "enum", enum=["driver", "passenger", "rear"])) == "passenger"

def test_boolean():
    f = extract_features("打开车窗")
    assert extract_boolean("打开车窗", f, ParamSpec("on", "boolean")) is True
