from pathlib import Path
from eval.run_eval import run

FIX = Path(__file__).parent / "fixtures" / "catalog"

def test_run_eval_fake_produces_metrics(tmp_path):
    ds = tmp_path / "mini.jsonl"
    ds.write_text('\n'.join([
        '{"utterance": "把空调调到25度", "expected_functions": ["set_temperature"], "type": "single", "split": "test"}',
        '{"utterance": "风速调到三档", "expected_functions": ["set_fan_speed"], "type": "single", "split": "test"}',
        '{"utterance": "今天天气怎么样", "expected_functions": [], "type": "ood", "split": "test"}',
    ]), encoding="utf-8")
    report = run(arm="C", dataset=str(ds), catalog=str(FIX), fake=True, permissive=True)
    assert "recall@1" in report["metrics"]
    assert 0.0 <= report["metrics"]["recall@1"] <= 1.0
    assert "p95_latency_ms" in report["metrics"]
