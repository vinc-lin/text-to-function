from pathlib import Path
from t2f.classify.train import build_training_pairs

def test_build_training_pairs_filters_to_single_label(tmp_path):
    ds = tmp_path / "d.jsonl"
    ds.write_text("\n".join([
        '{"utterance": "开窗", "expected_functions": ["open_window"], "type": "single"}',
        '{"utterance": "开窗并调温", "expected_functions": ["open_window","set_temperature"], "type": "multi_intent"}',
        '{"utterance": "天气", "expected_functions": [], "type": "ood"}',
    ]), encoding="utf-8")
    pairs = build_training_pairs([str(ds)])
    assert pairs == [("开窗", "open_window")]   # multi + ood excluded
