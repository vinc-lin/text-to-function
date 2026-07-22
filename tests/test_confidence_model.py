from t2f.safety.confidence import ExecutionConfidence, ConfidenceThresholds, build_confidence_dataset
from t2f.safety.features import FEATURE_ORDER
from t2f.types import Candidate, LexFeatures

def _feat(top1_score, has_req):
    d = {k: 0.0 for k in FEATURE_ORDER}
    d["top1_score"] = top1_score; d["margin"] = top1_score / 2; d["has_required_params"] = has_req
    return d

def test_fit_predict_separates():
    feats = [_feat(0.9, 1.0) for _ in range(8)] + [_feat(0.2, 0.0) for _ in range(8)]
    labels = [1] * 8 + [0] * 8
    m = ExecutionConfidence().fit(feats, labels)
    assert m.predict_proba(_feat(0.9, 1.0)) > m.predict_proba(_feat(0.2, 0.0))
    assert 0.0 <= m.predict_proba(_feat(0.5, 1.0)) <= 1.0

def test_build_dataset_labels():
    rows = [{"utterance": "a", "expected_functions": ["f1"], "type": "single"},
            {"utterance": "b", "expected_functions": [], "type": "ood"}]
    def route(u):
        fn = "f1" if u == "a" else "f9"
        return ([Candidate(fn, 0.7)], LexFeatures(raw=u))
    feats, labels = build_confidence_dataset(rows, route, {}, {})
    assert labels == [1, 0] and len(feats) == 2

def test_save_load_roundtrip(tmp_path):
    feats = [_feat(0.9, 1.0)] * 6 + [_feat(0.2, 0.0)] * 6
    m = ExecutionConfidence().fit(feats, [1]*6 + [0]*6)
    p = tmp_path / "c.joblib"; m.save(str(p))
    assert ExecutionConfidence.load(str(p)).predict_proba(_feat(0.9, 1.0)) > 0.5

def test_thresholds_defaults():
    t = ConfidenceThresholds()
    assert t.tau_low == 0.3 and t.tau_high == 0.7
