from __future__ import annotations
from dataclasses import dataclass
from .types import Candidate, Decision, Band, LexFeatures
from .retrieve import OOD_MARKER


@dataclass
class Thresholds:
    high_top1: float = 0.60
    high_margin: float = 0.08
    low_top1: float = 0.35


class ConfidenceGate:
    def __init__(self, thresholds: Thresholds | None = None):
        self.t = thresholds or Thresholds()

    def decide(self, candidates: list[Candidate], features: LexFeatures,
               cards_by_name: dict) -> Decision:
        if not candidates:
            return Decision(Band.LOW, None, [], ood_score=1.0, features={})
        if candidates[0].function == OOD_MARKER:
            # query matched an out-of-domain/chitchat prototype → reject, never execute
            return Decision(Band.LOW, None, candidates, ood_score=1.0,
                            features={"top1": candidates[0].score, "ood_marker": 1.0})
        top1 = candidates[0].score
        margin = top1 - (candidates[1].score if len(candidates) > 1 else 0.0)
        pc = candidates[0].signal_scores.get("param_compat", 0.0)
        feats = {"top1": top1, "margin": margin, "param_compat": pc}
        ood = 1.0 - top1
        if top1 < self.t.low_top1:
            return Decision(Band.LOW, None, candidates, ood_score=ood, features=feats)
        if top1 >= self.t.high_top1 and margin >= self.t.high_margin:
            return Decision(Band.HIGH, candidates[0].function, candidates, ood_score=ood, features=feats)
        return Decision(Band.MEDIUM, candidates[0].function, candidates, ood_score=ood, features=feats)
