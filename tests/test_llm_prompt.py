from t2f.types import FunctionCard, ParamSpec
from t2f.llm.prompt import build_prompt, compact_schema

def _card():
    return FunctionCard("set_temperature", "climate", "设置温度",
        params=[ParamSpec("temperature", "number", required=True, minimum=16, maximum=32),
                ParamSpec("position", "enum", enum=["driver", "passenger"])])

def test_compact_schema_lists_params():
    s = compact_schema(_card())
    assert "set_temperature" in s and "temperature" in s and "position" in s and "driver" in s

def test_build_prompt_contains_only_allowed_content():
    msgs = build_prompt("把温度调到25度", [_card()], {"temperature": 25})
    text = " ".join(m["content"] for m in msgs)
    assert any(m["role"] == "system" for m in msgs)
    assert "把温度调到25度" in text          # original clause
    assert "set_temperature" in text          # candidate name
    assert "25" in text                       # extracted params
    assert "climate" not in text              # NO domain name leaked (req 7)
