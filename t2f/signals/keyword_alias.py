from ..types import FunctionCard


def keyword_alias_score(clause: str, card: FunctionCard) -> float:
    if not card.aliases:
        base = 0.0
    else:
        hits = sum(1 for a in card.aliases if a and a in clause)
        base = min(1.0, hits / max(1, min(3, len(card.aliases))))
    name_tokens = [t for t in card.name.split("_") if len(t) > 2]
    bonus = 0.1 if any(t in clause for t in name_tokens) else 0.0
    return min(1.0, base + bonus)
