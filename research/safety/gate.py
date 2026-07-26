"""The Spec-3 learned execution-confidence gate.

Moved out of `t2f/gate.py`: no eval arm ever constructed it, and `t2f/gate.py` imported
`confidence_features` at module level, dragging this whole path into the runtime import
graph on every gate load. The measured frontier it produced is recorded in
docs/superpowers/RESULTS.md; this is the code behind those numbers.
"""
from __future__ import annotations
from t2f.types import Decision, Band
from t2f.retrieve import OOD_MARKER
from research.safety.features import confidence_features


class ConfidenceModelGate:
    """Bands on a learned P(top-1 correct) instead of a raw score threshold. Same decide() shape."""

    def __init__(self, model, thresholds, domain_keywords=None):
        self.model = model
        self.t = thresholds
        self.domain_keywords = domain_keywords or {}

    def decide(self, candidates, features, cards_by_name):
        if not candidates:
            return Decision(Band.LOW, None, [], ood_score=1.0, features={})
        if candidates[0].function == OOD_MARKER:
            return Decision(Band.LOW, None, candidates, ood_score=1.0, features={"ood_marker": 1.0})
        feat = confidence_features(candidates, features, cards_by_name, self.domain_keywords)
        p = self.model.predict_proba(feat)
        info = {"p_correct": p}
        if p < self.t.tau_low:
            return Decision(Band.LOW, None, candidates, ood_score=1.0 - p, features=info)
        if p >= self.t.tau_high:
            return Decision(Band.HIGH, candidates[0].function, candidates, ood_score=1.0 - p, features=info)
        return Decision(Band.MEDIUM, candidates[0].function, candidates, ood_score=1.0 - p, features=info)
