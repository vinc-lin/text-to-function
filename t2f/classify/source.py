from __future__ import annotations
from ..types import Candidate


class ClassifierCandidateSource:
    def __init__(self, classifier, k: int = 3):
        self.classifier = classifier
        self.k = k

    def augment(self, candidates: list[Candidate], clause: str):
        by_name = {c.function: c for c in candidates}
        probs: dict[str, float] = {}
        for fn, p in self.classifier.predict_topk(clause, self.k):
            probs[fn] = p
            if fn not in by_name:
                cand = Candidate(function=fn, score=0.0, embedding_score=0.0)
                candidates.append(cand)
                by_name[fn] = cand
        return candidates, probs
