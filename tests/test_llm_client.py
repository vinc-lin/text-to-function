from t2f.types import FunctionCard, ToolCall, LLMResult
from t2f.llm.client import FakeLLMClient

def test_fake_scripts_by_substring():
    c = FakeLLMClient(scripts={
        "温度": LLMResult(tool_call=ToolCall("set_temperature", {"temperature": 25, "position": "passenger"})),
        "音量": LLMResult(clarification="您想把音量调到多少？"),
    })
    r = c.complete_tool_call("把副驾温度调到25度", [], {})
    assert r.tool_call.name == "set_temperature" and r.tool_call.parameters["position"] == "passenger"
    assert c.complete_tool_call("调音量", [], {}).clarification is not None

def test_fake_default_when_no_match():
    c = FakeLLMClient(scripts={}, default=LLMResult(error="no_match"))
    assert c.complete_tool_call("随便", [], {}).error == "no_match"
