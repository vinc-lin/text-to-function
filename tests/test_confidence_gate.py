from t2f.types import Candidate, LexFeatures, Band
from t2f.retrieve import OOD_MARKER
from research.safety.confidence import ExecutionConfidence, ConfidenceThresholds
from research.safety.features import FEATURE_ORDER
from research.safety.gate import ConfidenceModelGate

def _model():
    def f(s):
        d = {k: 0.0 for k in FEATURE_ORDER}; d["top1_score"] = s; d["margin"] = s / 2; return d
    return ExecutionConfidence().fit([f(0.9)] * 8 + [f(0.1)] * 8, [1] * 8 + [0] * 8)

def test_bands_by_probability():
    g = ConfidenceModelGate(_model(), ConfidenceThresholds(tau_low=0.3, tau_high=0.6))
    hi = g.decide([Candidate("a", 0.95), Candidate("b", 0.4)], LexFeatures(raw="x"), {})
    lo = g.decide([Candidate("a", 0.05), Candidate("b", 0.02)], LexFeatures(raw="x"), {})
    assert hi.band == Band.HIGH and hi.chosen == "a"
    assert lo.band == Band.LOW and lo.chosen is None
    assert "p_correct" in hi.features

def test_ood_marker_rejected():
    g = ConfidenceModelGate(_model(), ConfidenceThresholds(0.3, 0.6))
    d = g.decide([Candidate(OOD_MARKER, 0.9), Candidate("a", 0.1)], LexFeatures(raw="x"), {})
    assert d.band == Band.LOW and d.chosen is None
