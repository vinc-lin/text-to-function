from t2f.types import Candidate, LexFeatures
from t2f.gate import ConfidenceGate, Thresholds, Band

T = Thresholds(high_top1=0.6, high_margin=0.08, low_top1=0.35)

def test_high_confidence():
    cands = [Candidate("a", 0.8, signal_scores={"param_compat": 1.0}), Candidate("b", 0.5)]
    d = ConfidenceGate(T).decide(cands, LexFeatures(), {})
    assert d.band == Band.HIGH and d.chosen == "a"

def test_medium_when_margin_small():
    cands = [Candidate("a", 0.7), Candidate("b", 0.69)]
    assert ConfidenceGate(T).decide(cands, LexFeatures(), {}).band == Band.MEDIUM

def test_low_when_top1_weak():
    cands = [Candidate("a", 0.2), Candidate("b", 0.1)]
    d = ConfidenceGate(T).decide(cands, LexFeatures(), {})
    assert d.band == Band.LOW and d.chosen is None
