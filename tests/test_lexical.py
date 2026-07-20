# tests/test_lexical.py
from t2f.lexical import extract_features

def test_temperature():
    f = extract_features("把空调调到25度")
    assert 25.0 in f.temperatures

def test_level_and_position():
    f = extract_features("后排风速调到三档")
    assert 3 in f.levels
    assert "rear" in f.positions

def test_percentage():
    f = extract_features("车窗开到百分之五十")
    assert 50.0 in f.percentages

def test_on_off_and_operation():
    assert extract_features("打开车窗").on_off is True
    assert extract_features("关闭空调").on_off is False
    assert extract_features("温度调高一点").operation == "increase"
    assert extract_features("风速开到最大").operation == "max"

def test_position_driver_aliases():
    assert "driver" in extract_features("主驾这边热").positions
    assert "passenger" in extract_features("副驾驶座位").positions
