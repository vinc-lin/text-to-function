from __future__ import annotations
from dataclasses import dataclass
from .types import Candidate, Decision, Band, LexFeatures
from .retrieve import OOD_MARKER


# Frozen because it is a value, and because PERMISSIVE below is now shared. `ConfidenceGate`
# keeps the Thresholds it is handed by reference, so before this a single `gate.t.high_top1 = x`
# anywhere reached into the module constant and retuned the gate for every session in the
# process — including ones built afterwards. Nothing in the tree does that, and now nothing can.
@dataclass(frozen=True)
class Thresholds:
    high_top1: float = 0.60
    high_margin: float = 0.08
    low_top1: float = 0.35


# The gate behind `--gate permissive` / `/gate permissive` (see cli/__main__.py): zeros
# high_margin, the shipped gate's binding constraint (config.yaml) — recognition usually finds
# the right function, but a close runner-up keeps it out of HIGH, and this stops blocking it.
# The two floors drop with it, which guts LOW and trades refusal for coverage (see
# docs/TRYING_IT.md). A shipped product mode, not a test fixture, so it lives beside Thresholds
# rather than in cli/session.py or scattered across test files.
PERMISSIVE = Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)


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
