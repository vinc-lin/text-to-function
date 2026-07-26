from t2f.types import Candidate, FunctionCard, ParamSpec, LexFeatures
from t2f.retrieve import OOD_MARKER
from research.safety.features import confidence_features, FEATURE_ORDER
from t2f.lexical import extract_features

CARDS = {"set_temperature": FunctionCard("set_temperature", "climate", "温度",
    params=[ParamSpec("temperature", "number", required=True, unit="celsius")])}
DK = {"climate": ["空调", "温度"]}

def test_feature_keys_and_values():
    lex = extract_features("把空调调到25度")
    cands = [Candidate("set_temperature", 0.8, signal_scores={"param_compat": 1.0}),
             Candidate("set_fan_speed", 0.6),
             Candidate(OOD_MARKER, 0.3)]
    f = confidence_features(cands, lex, CARDS, DK)
    assert set(f) == set(FEATURE_ORDER)
    assert f["top1_score"] == 0.8 and abs(f["margin"] - 0.2) < 1e-9
    assert f["ood_marker_sim"] == 0.3
    assert f["top1_param_compat"] == 1.0
    assert f["has_required_params"] == 1.0        # temperature is extractable from the clause
    assert f["domain_kw_hit"] == 1.0

def test_empty_candidates_all_zero():
    f = confidence_features([], LexFeatures(raw="x"), CARDS, DK)
    assert all(f[k] == 0.0 for k in FEATURE_ORDER)
