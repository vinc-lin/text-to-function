from pathlib import Path
from t2f.cards import load_catalog
from t2f.embed import FakeEmbedder
from t2f.config import Config
from t2f.gate import PERMISSIVE
from eval.arms import build_arm_c, build_arm_c_baseline, predict

FIX = Path(__file__).parent / "fixtures" / "catalog"

def _cfg():
    c = Config.default(); c.thresholds = PERMISSIVE; return c

def test_arm_c_predict_record_shape():
    cards = load_catalog(FIX)
    p = build_arm_c(cards, FakeEmbedder(256), _cfg())
    rec = predict(p, {"utterance": "把空调调到25度", "expected_functions": ["set_temperature"],
                      "expected_params": {"set_temperature": {"temperature": 25}}, "type": "single"})
    assert rec["ranked_per_clause"] and rec["predicted_functions"][0] == "set_temperature"
    assert len(rec["bands"]) == 1 and "exec_correct" in rec

def test_baseline_builds():
    cards = load_catalog(FIX)
    p = build_arm_c_baseline(cards, FakeEmbedder(256), _cfg())
    assert predict(p, {"utterance": "开车窗", "expected_functions": ["open_window"],
                       "type": "single"})["predicted_functions"]
