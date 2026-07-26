def test_confidence_gate_pipeline_fake():
    from t2f.cards import load_catalog
    from t2f.embed import FakeEmbedder
    from t2f.config import Config
    from t2f.score import Scorer
    from t2f.pipeline import Pipeline
    from research.safety.features import FEATURE_ORDER
    from research.safety.confidence import ExecutionConfidence, ConfidenceThresholds
    from research.safety.gate import ConfidenceModelGate
    cards = load_catalog("data/catalog")
    cfg = Config.default()

    def f(s):
        d = {k: 0.0 for k in FEATURE_ORDER}
        d["top1_score"] = s
        return d

    model = ExecutionConfidence().fit([f(0.9)] * 8 + [f(0.1)] * 8, [1] * 8 + [0] * 8)
    gate = ConfidenceModelGate(model, ConfidenceThresholds(0.3, 0.6), cfg.domain_keywords)
    pipe = Pipeline(cards, FakeEmbedder(256), Scorer(cfg.weights, cfg.domain_keywords), gate, cfg)
    res = pipe.route("把空调调到25度")
    assert len(res.clauses) == 1 and res.clauses[0].decision.band is not None
