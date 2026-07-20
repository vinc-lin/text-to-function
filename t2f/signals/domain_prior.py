from ..types import FunctionCard


def domain_prior_score(clause: str, card: FunctionCard, domain_keywords: dict[str, list[str]]) -> float:
    kws = domain_keywords.get(card.domain, [])
    return 1.0 if any(k in clause for k in kws) else 0.0
