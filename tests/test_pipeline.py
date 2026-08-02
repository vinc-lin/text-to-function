# tests/test_pipeline.py
from pathlib import Path
from t2f.cards import load_catalog
from t2f.embed import FakeEmbedder
from t2f.retrieve import PrototypeStore, Retriever
from t2f.score import Scorer
from t2f.gate import ConfidenceGate, PERMISSIVE
from t2f.pipeline import Pipeline
from t2f.config import Config

FIX = Path(__file__).parent / "fixtures" / "catalog"

def _pipeline():
    cards = load_catalog(FIX)
    emb = FakeEmbedder(256)
    cfg = Config.default()
    cfg.thresholds = PERMISSIVE  # permissive for fake emb
    return Pipeline(cards, emb, Scorer(cfg.weights, cfg.domain_keywords),
                    ConfidenceGate(cfg.thresholds), cfg)

def test_route_single_intent_produces_toolcall():
    res = _pipeline().route("把空调调到25度")
    assert len(res.clauses) == 1
    cl = res.clauses[0]
    assert cl.decision.chosen == "set_temperature"
    if cl.tool_call:  # high band
        assert cl.tool_call.parameters.get("temperature") == 25

def test_route_multi_intent_splits():
    # two clauses that are both recognizable actions in the reduced fixture (车窗 + 温度 aliases);
    # the new architecture gates clause creation on actionability, so both must be real actions.
    res = _pipeline().route("开车窗,温度调到25度")
    assert len(res.clauses) == 2

def test_latency_recorded():
    res = _pipeline().route("把空调调到25度")
    assert res.clauses[0].latency_ms >= 0
