# tests/test_respond.py
from t2f.types import FunctionCard, ToolCall, ParamSpec
from t2f.respond import render_response, build_clarification
from t2f.execute import MockExecutor

def test_render_fills_template():
    card = FunctionCard("set_temperature", "climate", "温度",
        params=[ParamSpec("temperature", "number"), ParamSpec("position", "enum", enum=["driver"])],
        response_template="已将{position}温度设置为{temperature}°C。")
    r = render_response(card, ToolCall("set_temperature", {"temperature": 25, "position": "driver"}))
    assert "25" in r and "°C" in r

def test_clarification_for_missing_position():
    card = FunctionCard("set_temperature", "climate", "温度",
        params=[ParamSpec("position", "enum", required=True, enum=["driver", "passenger"])])
    c = build_clarification(card, ["position"])
    assert c.pending.pending_function == "set_temperature"
    assert "区域" in c.question or "位置" in c.question

def test_mock_executor():
    assert MockExecutor().execute(ToolCall("x", {"a": 1}))["ok"] is True
