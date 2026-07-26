from __future__ import annotations
from t2f.retrieve import OOD_MARKER
from t2f.params.extract import ParameterExtractor

FEATURE_ORDER = ["top1_score", "margin", "top3_spread", "ood_marker_sim", "top1_param_compat",
                 "classifier_prob", "classifier_margin", "n_candidates", "query_len",
                 "has_required_params", "domain_kw_hit"]
_EX = ParameterExtractor()


def confidence_features(candidates, lex, cards_by_name, domain_keywords=None) -> dict:
    domain_keywords = domain_keywords or {}
    if not candidates:
        return {k: 0.0 for k in FEATURE_ORDER}
    c0 = candidates[0]
    s0 = c0.score
    s1 = candidates[1].score if len(candidates) > 1 else 0.0
    s2 = candidates[2].score if len(candidates) > 2 else s1
    ood_sim = next((c.score for c in candidates if c.function == OOD_MARKER), 0.0)
    cp0 = c0.signal_scores.get("classifier_prob", 0.0)
    cp1 = candidates[1].signal_scores.get("classifier_prob", 0.0) if len(candidates) > 1 else 0.0
    card = cards_by_name.get(c0.function)
    if card is None:
        has_req = dom_hit = 0.0
    else:
        _, missing = _EX.extract(lex.raw, lex, card)
        has_req = 1.0 if not missing else 0.0
        kws = domain_keywords.get(card.domain, [])
        dom_hit = 1.0 if any(k in lex.raw for k in kws) else 0.0
    return {
        "top1_score": s0, "margin": s0 - s1, "top3_spread": s0 - s2,
        "ood_marker_sim": ood_sim, "top1_param_compat": c0.signal_scores.get("param_compat", 0.0),
        "classifier_prob": cp0, "classifier_margin": cp0 - cp1,
        "n_candidates": float(len(candidates)), "query_len": float(len(lex.raw)),
        "has_required_params": has_req, "domain_kw_hit": dom_hit,
    }
