from __future__ import annotations
from dataclasses import dataclass
from .types import Candidate, Decision, Band, LexFeatures


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


def calibrate_gate(dev_rows, route_top_candidates, target_high_precision: float = 0.98) -> Thresholds:
    """Grid-search thresholds on dev rows.
    dev_rows: list of dicts {utterance, expected_functions, type}.
    route_top_candidates(utterance) -> list[Candidate] (already hybrid-scored).
    Picks the thresholds giving the most HIGH-band coverage while HIGH-band top-1 precision
    >= target and OOD examples never land in HIGH.
    """
    best, best_cov = Thresholds(), -1.0
    for high_top1 in [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        for high_margin in [0.03, 0.05, 0.08, 0.12]:
            for low_top1 in [0.25, 0.3, 0.35, 0.4]:
                if low_top1 >= high_top1:
                    continue
                t = Thresholds(high_top1, high_margin, low_top1)
                gate = ConfidenceGate(t)
                high_total = high_correct = ood_in_high = 0
                for r in dev_rows:
                    cands = route_top_candidates(r["utterance"])
                    d = gate.decide(cands, LexFeatures(), {})
                    if d.band == Band.HIGH:
                        high_total += 1
                        if r.get("type") == "ood":
                            ood_in_high += 1
                        elif d.chosen in r.get("expected_functions", []):
                            high_correct += 1
                if high_total == 0 or ood_in_high > 0:
                    continue
                prec = high_correct / high_total
                if prec >= target_high_precision and high_total > best_cov:
                    best, best_cov = t, high_total
    return best
