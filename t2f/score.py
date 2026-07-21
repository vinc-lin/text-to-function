from __future__ import annotations
from .types import Candidate, FunctionCard, LexFeatures
from .signals.keyword_alias import keyword_alias_score
from .signals.param_compat import param_compat_score
from .signals.domain_prior import domain_prior_score

DEFAULT_WEIGHTS = {"embedding": 0.55, "keyword_alias": 0.15, "param_compat": 0.25, "domain_prior": 0.05,
                   "classifier_prob": 0.0}


class Scorer:
    def __init__(self, weights: dict[str, float] | None = None,
                 domain_keywords: dict[str, list[str]] | None = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.domain_keywords = domain_keywords or {}

    def rescore(self, clause: str, features: LexFeatures, candidates: list[Candidate],
                cards_by_name: dict[str, FunctionCard],
                classifier_probs: dict[str, float] | None = None) -> list[Candidate]:
        w = self.weights
        classifier_probs = classifier_probs or {}
        for c in candidates:
            card = cards_by_name.get(c.function)
            if card is None:
                continue
            kw = keyword_alias_score(clause, card)
            pc = param_compat_score(features, card)
            dp = domain_prior_score(clause, card, self.domain_keywords)
            cp = classifier_probs.get(c.function, 0.0)
            c.signal_scores = {"keyword_alias": kw, "param_compat": pc, "domain_prior": dp,
                               "classifier_prob": cp}
            c.score = (w["embedding"] * c.embedding_score + w["keyword_alias"] * kw
                       + w["param_compat"] * pc + w["domain_prior"] * dp
                       + w.get("classifier_prob", 0.0) * cp)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates


class EmbeddingOnlyScorer:
    def rescore(self, clause, features, candidates, cards_by_name, classifier_probs=None):
        candidates.sort(key=lambda c: c.embedding_score, reverse=True)
        return candidates
