from pathlib import Path
from t2f.cards import load_catalog
from t2f.embed import FakeEmbedder
from t2f.config import Config
from t2f.gate import Thresholds
from t2f.types import LLMResult, ToolCall
from t2f.llm.client import FakeLLMClient
from eval.arms import build_arm_c_llm, predict

FIX = Path(__file__).parent / "fixtures" / "catalog"

def _cfg():
    c = Config.default(); c.thresholds = Thresholds(0.9, 0.5, 0.05); return c  # force MEDIUM band

def test_arm_c_llm_resolves_medium_via_fake_llm():
    cards = load_catalog(FIX)
    client = FakeLLMClient(default=LLMResult(tool_call=ToolCall("set_temperature", {"temperature": 25})))
    p = build_arm_c_llm(cards, FakeEmbedder(256), _cfg(), client)
    rec = predict(p, {"utterance": "把空调调到25度", "expected_functions": ["set_temperature"],
                      "expected_params": {"set_temperature": {"temperature": 25}}, "type": "single"})
    assert "llm_json_ok" in rec and rec["needs_llm"][0] is True
