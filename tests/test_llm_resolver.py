from t2f.types import FunctionCard, ParamSpec, Candidate, Decision, Band, LexFeatures, ToolCall, LLMResult
from t2f.llm.client import FakeLLMClient
from t2f.pipeline import LLMResolver
from t2f.lexical import extract_features

CARD = FunctionCard("set_temperature", "climate", "温度",
    params=[ParamSpec("temperature", "number", required=True, minimum=16, maximum=32),
            ParamSpec("position", "enum", required=True, enum=["driver", "passenger"])])
CARDS = {"set_temperature": CARD}

def _decision():
    return Decision(Band.MEDIUM, "set_temperature",
                    [Candidate("set_temperature", 0.5), Candidate("set_fan_speed", 0.4)])

def test_llm_resolver_executes_valid_call():
    client = FakeLLMClient(scripts={"温度": LLMResult(
        tool_call=ToolCall("set_temperature", {"temperature": 25, "position": "passenger"}))})
    r = LLMResolver(client).resolve("把副驾温度调到25度", extract_features("把副驾温度调到25度"),
                                    _decision(), CARDS, executor=None)
    assert r.tool_call is not None and r.response is not None and r.needs_llm is True

def test_llm_resolver_clarifies_on_missing_param():
    client = FakeLLMClient(scripts={"温度": LLMResult(
        tool_call=ToolCall("set_temperature", {"temperature": 25}))})  # position missing
    r = LLMResolver(client).resolve("温度调到25度", extract_features("温度调到25度"),
                                    _decision(), CARDS, executor=None)
    assert r.tool_call is None and r.clarification is not None

def test_llm_resolver_rejects_invalid_function():
    client = FakeLLMClient(scripts={"x": LLMResult(tool_call=ToolCall("not_a_candidate", {}))},
                           default=LLMResult(tool_call=ToolCall("not_a_candidate", {})))
    r = LLMResolver(client, max_retries=0).resolve("x", extract_features("x"), _decision(), CARDS, executor=None)
    assert r.tool_call is None and r.validation_errors  # never executes a non-candidate
