from t2f.pipeline import Pipeline
from t2f.embed import FakeEmbedder
from t2f.score import Scorer
from t2f.gate import ConfidenceGate
from t2f.config import Config
from t2f.cards import load_catalog
from t2f.types import SpanRole

def _pipe():
    cfg = Config.load("config.yaml")
    cards = load_catalog("data/catalog")
    return Pipeline(cards, FakeEmbedder(256), Scorer(cfg.weights, cfg.domain_keywords),
                    ConfidenceGate(cfg.thresholds), cfg)

def test_single_action_uses_legacy_path():
    rr = _pipe().route("把温度调到22度")
    assert rr.plan is None and len(rr.clauses) == 1

def test_context_bearing_utterance_builds_plan_and_suppresses_context():
    rr = _pipe().route("后排小孩老去按车窗，把车窗锁打开")
    assert rr.plan is not None
    assert all("后排小孩" not in c.clause for c in rr.clauses)

def test_zero_action_utterance_falls_back_to_legacy():
    rr = _pipe().route("导航去公司")
    assert rr.plan is None and len(rr.clauses) >= 1
