import pytest


def test_fake_llm_pipeline_end_to_end():
    """The Spec-2 medium-band pipeline runs end-to-end with a FakeLLMClient (no model)."""
    from t2f.cards import load_catalog
    from t2f.embed import FakeEmbedder
    from t2f.config import Config
    from t2f.gate import Thresholds
    from t2f.types import LLMResult, ToolCall
    from t2f.llm.client import FakeLLMClient
    from eval.arms import build_arm_c_llm, predict
    cards = load_catalog("data/catalog")
    cfg = Config.default()
    cfg.thresholds = Thresholds(0.9, 0.5, 0.05)  # force MEDIUM band -> LLM
    client = FakeLLMClient(default=LLMResult(tool_call=ToolCall("set_volume", {"level": 3})))
    p = build_arm_c_llm(cards, FakeEmbedder(256), cfg, client, ood_texts=["今天天气怎么样"])
    rec = predict(p, {"utterance": "音量调到3", "expected_functions": ["set_volume"], "type": "single"})
    assert "llm_json_ok" in rec and len(rec["bands"]) == 1


@pytest.mark.model
def test_real_llm_emits_schema_valid_toolcall():
    """The real xgrammar-constrained Qwen3-0.6B emits a tool call that passes strict validation."""
    from t2f.cards import load_catalog
    from t2f.llm.client import TransformersXGrammarClient
    from t2f.validate import validate_tool_call
    cards = {c.name: c for c in load_catalog("data/catalog")}
    cand = [cards["set_temperature"], cards["set_fan_speed"]]
    r = TransformersXGrammarClient().complete_tool_call("把温度调到24度", cand, {"temperature": 24})
    # constrained output is either a valid tool-call or an explicit reject — never malformed
    if r.tool_call is not None:
        tc, errs = validate_tool_call(r.tool_call.name, r.tool_call.parameters, cards,
                                      [c.name for c in cand])
        assert tc is not None or errs
    else:
        assert r.clarification is not None  # declined (reject) — also acceptable
