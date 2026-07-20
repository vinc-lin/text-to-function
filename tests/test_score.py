from t2f.types import FunctionCard, ParamSpec, Candidate
from t2f.lexical import extract_features
from t2f.score import Scorer, EmbeddingOnlyScorer

def _cards():
    return {
        "set_temperature": FunctionCard("set_temperature", "climate", "温度",
            params=[ParamSpec("temperature", "number", unit="celsius")], aliases=["温度"]),
        "set_fan_speed": FunctionCard("set_fan_speed", "climate", "风速",
            params=[ParamSpec("level", "integer", unit="level")], aliases=["风速"]),
    }

def test_hybrid_promotes_param_compatible_function():
    cards = _cards()
    # embedding slightly favors fan, but the query clearly sets a temperature
    cands = [Candidate("set_fan_speed", 0.61, embedding_score=0.61),
             Candidate("set_temperature", 0.60, embedding_score=0.60)]
    f = extract_features("把温度调到25度")
    sc = Scorer(weights={"embedding": 0.5, "keyword_alias": 0.2, "param_compat": 0.25, "domain_prior": 0.05},
                domain_keywords={"climate": ["空调", "温度", "风"]})
    out = sc.rescore("把温度调到25度", f, cands, cards)
    assert out[0].function == "set_temperature"
    assert "param_compat" in out[0].signal_scores

def test_baseline_preserves_order():
    cands = [Candidate("a", 0.9, embedding_score=0.9), Candidate("b", 0.8, embedding_score=0.8)]
    out = EmbeddingOnlyScorer().rescore("x", extract_features("x"), cands, {})
    assert [c.function for c in out] == ["a", "b"]
