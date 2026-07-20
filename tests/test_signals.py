from t2f.types import FunctionCard, ParamSpec
from t2f.lexical import extract_features
from t2f.signals.keyword_alias import keyword_alias_score
from t2f.signals.param_compat import param_compat_score

def _temp_card():
    return FunctionCard("set_temperature", "climate", "设置温度",
                        params=[ParamSpec("temperature", "number", unit="celsius"),
                                ParamSpec("position", "enum", enum=["driver", "passenger"])],
                        aliases=["温度", "空调温度"])

def _fan_card():
    return FunctionCard("set_fan_speed", "climate", "风速",
                        params=[ParamSpec("level", "integer", unit="level")], aliases=["风速"])

def test_keyword_alias():
    assert keyword_alias_score("把温度调到25度", _temp_card()) > 0
    assert keyword_alias_score("打开车窗", _temp_card()) == 0

def test_param_compat_favors_matching_function():
    f = extract_features("把空调调到25度")
    assert param_compat_score(f, _temp_card()) > param_compat_score(f, _fan_card())

def test_param_compat_level_favors_fan():
    f = extract_features("风速调到三档")
    assert param_compat_score(f, _fan_card()) > param_compat_score(f, _temp_card())
