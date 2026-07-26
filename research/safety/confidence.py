from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from research.safety.features import FEATURE_ORDER, confidence_features


@dataclass
class ConfidenceThresholds:
    tau_low: float = 0.3
    tau_high: float = 0.7


def _vec(feat: dict) -> np.ndarray:
    return np.array([feat[k] for k in FEATURE_ORDER], dtype=float)


class ExecutionConfidence:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, feature_dicts, labels):
        self.clf.fit(np.vstack([_vec(f) for f in feature_dicts]), labels)
        return self

    def predict_proba(self, feat: dict) -> float:
        classes = list(self.clf.classes_)
        if 1 not in classes:
            return 0.0
        return float(self.clf.predict_proba(_vec(feat).reshape(1, -1))[0][classes.index(1)])

    def save(self, path):
        import joblib
        joblib.dump(self.clf, path)

    @classmethod
    def load(cls, path):
        import joblib
        obj = cls()
        obj.clf = joblib.load(path)
        return obj


def build_confidence_dataset(rows, route_fn, cards_by_name, domain_keywords):
    feats, labels = [], []
    for r in rows:
        cands, lex = route_fn(r["utterance"])
        feats.append(confidence_features(cands, lex, cards_by_name, domain_keywords))
        top1 = cands[0].function if cands else None
        is_ood = r.get("type") == "ood"
        labels.append(1 if (not is_ood and top1 in r.get("expected_functions", [])) else 0)
    return feats, labels


def calibrate_thresholds(points, target_error: float = 0.05,
                         ood_budget_high: float = 0.10, ood_budget_low: float = 0.30) -> ConfidenceThresholds:
    """Pick (tau_low, tau_high) for a two-band safety policy from dev points (p, correct, is_ood).

    tau_high — the DIRECT-execute (zero-LLM) floor: lowest cut where the executed set has both
      in-domain error <= target_error AND OOD fraction <= ood_budget_high, maximizing coverage.
      (Counting OOD-as-error alone forces tau_high implausibly high, killing the medium band, so
      in-domain precision and OOD leakage are budgeted separately.)
    tau_low — the REJECT floor below which we abstain: lowest cut whose above-set has OOD fraction
      <= ood_budget_low. The band [tau_low, tau_high) is the medium/LLM zone, where the LLM's reject
      option + OOD prototypes provide the residual OOD defense. Keeping tau_low < tau_high preserves
      a real medium band so the LLM can recover coverage.
    """
    if not points:
        return ConfidenceThresholds()
    cuts = sorted({round(p, 3) for p, _, _ in points})
    n_ood = sum(1 for _, _, o in points if o)

    def ood_frac_above(cut):
        above_ood = sum(1 for p, _, o in points if o and p >= cut)
        return above_ood / n_ood if n_ood else 0.0

    # tau_high: in-domain error <= target AND OOD leakage <= budget, max coverage
    best_high, best_cov, fallback_high, fallback_key = None, -1, cuts[-1], (1.1, 1.1)
    for cut in cuts:
        indom = [(p, c) for (p, c, o) in points if not o and p >= cut]
        cov = len(indom)
        err = sum(1 for _, c in indom if not c) / cov if cov else 1.0
        oodf = ood_frac_above(cut)
        if (oodf, err) < fallback_key:
            fallback_key, fallback_high = (oodf, err), cut
        if err <= target_error and oodf <= ood_budget_high and cov > best_cov:
            best_high, best_cov = cut, cov
    tau_high = best_high if best_high is not None else fallback_high

    # tau_low: lowest cut with OOD leakage <= ood_budget_low (the medium-band floor)
    lows = [c for c in cuts if c <= tau_high and ood_frac_above(c) <= ood_budget_low]
    tau_low = min(lows) if lows else 0.0
    tau_low = min(tau_low, tau_high)
    return ConfidenceThresholds(tau_low=float(tau_low), tau_high=float(tau_high))
