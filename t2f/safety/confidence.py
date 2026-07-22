from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .features import FEATURE_ORDER, confidence_features


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


def calibrate_thresholds(points, target_error: float = 0.05) -> ConfidenceThresholds:
    if not points:
        return ConfidenceThresholds()
    cuts = sorted({round(p, 3) for p, _, _ in points})
    # tau_high: lowest cut whose executed set (p>=cut) has error <= target, max coverage
    best_high, best_cov, fallback_high, fallback_err = None, -1, cuts[-1], 1.1
    for cut in cuts:
        ex = [(p, c, o) for (p, c, o) in points if p >= cut]
        if not ex:
            continue
        err = sum(1 for _, c, _ in ex if not c) / len(ex)
        if err < fallback_err:
            fallback_err, fallback_high = err, cut
        if err <= target_error and len(ex) > best_cov:
            best_high, best_cov = cut, len(ex)
    tau_high = best_high if best_high is not None else fallback_high
    # tau_low: lowest cut with no OOD point at/above it (all OOD rejected)
    ood_ps = [p for p, _, o in points if o]
    tau_low = 0.0 if not ood_ps else min([c for c in cuts if all(p < c for p in ood_ps)] or [tau_high])
    tau_low = min(tau_low, tau_high)
    return ConfidenceThresholds(tau_low=float(tau_low), tau_high=float(tau_high))
