# Accuracy & Safety Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive OOD false-execution (0.32) and incorrect-execution (0.34) toward ≈0 via a learned execution-confidence gate that abstains below a safety-calibrated threshold, plus hard-negative mining to lift recall@3.

**Architecture:** A pure feature extractor turns routing signals into a feature vector; a logistic-regression `ExecutionConfidence` model predicts `P(top-1 is correct)`; a `ConfidenceModelGate` (same `decide()` interface as the existing gate) bands on that probability (execute / LLM-fallback / abstain). A hard-negative mining tool surfaces confusable clusters for targeted prototype additions. Everything is LR-over-cheap-features (on-device-friendly) and tested on this x86 box against the gold set.

**Tech Stack:** Python 3.10, scikit-learn + joblib (already installed), numpy, plus the Spec 1/2 stack. Default tests never load a model; the real embedder/LLM run only in the hands-on integration task.

---

## Conventions (same as Specs 1–2)
- TDD per task: failing test → confirm FAIL → implement → confirm PASS → commit.
- `python3 -m pytest -q` from repo root. No `pip install` in tasks (deps present).
- New dataclasses/constants live with their module; reuse Spec 1/2 types.

## New / changed files
```
t2f/safety/__init__.py
t2f/safety/features.py        # FEATURE_ORDER + confidence_features()
t2f/safety/confidence.py      # ConfidenceThresholds, ExecutionConfidence, build_confidence_dataset()
t2f/gate.py                   # + ConfidenceModelGate (modify)
t2f/tools/__init__.py
t2f/tools/mine_hard_negatives.py   # mine_confusions() + report
eval/metrics.py               # + coverage(), executed_correct helpers (modify)
eval/run_eval.py              # + confidence-gate wiring / frontier (modify)
```

---

## Task 1: Confidence features (`t2f/safety/features.py`)

**Files:** Create `t2f/safety/__init__.py`, `t2f/safety/features.py`; Test `tests/test_confidence_features.py`

**Contract:** `confidence_features(candidates, lex, cards_by_name, domain_keywords=None) -> dict[str,float]` with exactly the keys in `FEATURE_ORDER`. `lex` is a `LexFeatures` (its `.raw` holds the clause text). Reads per-candidate `signal_scores` (`param_compat`, `classifier_prob`) populated by the scorer; the OOD marker's score becomes `ood_marker_sim`; `has_required_params` uses the deterministic extractor on the top-1 card. Empty candidate list → all-zero features.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_confidence_features.py
from t2f.types import Candidate, FunctionCard, ParamSpec, LexFeatures
from t2f.retrieve import OOD_MARKER
from t2f.safety.features import confidence_features, FEATURE_ORDER
from t2f.lexical import extract_features

CARDS = {"set_temperature": FunctionCard("set_temperature", "climate", "温度",
    params=[ParamSpec("temperature", "number", required=True, unit="celsius")])}
DK = {"climate": ["空调", "温度"]}

def test_feature_keys_and_values():
    lex = extract_features("把空调调到25度")
    cands = [Candidate("set_temperature", 0.8, signal_scores={"param_compat": 1.0}),
             Candidate("set_fan_speed", 0.6),
             Candidate(OOD_MARKER, 0.3)]
    f = confidence_features(cands, lex, CARDS, DK)
    assert set(f) == set(FEATURE_ORDER)
    assert f["top1_score"] == 0.8 and abs(f["margin"] - 0.2) < 1e-9
    assert f["ood_marker_sim"] == 0.3
    assert f["top1_param_compat"] == 1.0
    assert f["has_required_params"] == 1.0        # temperature is extractable from the clause
    assert f["domain_kw_hit"] == 1.0

def test_empty_candidates_all_zero():
    f = confidence_features([], LexFeatures(raw="x"), CARDS, DK)
    assert all(f[k] == 0.0 for k in FEATURE_ORDER)
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement**
```python
# t2f/safety/__init__.py
```
```python
# t2f/safety/features.py
from __future__ import annotations
from ..retrieve import OOD_MARKER
from ..params.extract import ParameterExtractor

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
```
- [ ] **Step 4: Run test** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: execution-confidence features"`

---

## Task 2: Confidence model + dataset builder (`t2f/safety/confidence.py`)

**Files:** Create `t2f/safety/confidence.py`; Test `tests/test_confidence_model.py`

**Contract:**
- `ConfidenceThresholds(tau_low=0.3, tau_high=0.7)` dataclass.
- `ExecutionConfidence`: `fit(feature_dicts, labels) -> self` (LR over `FEATURE_ORDER`), `predict_proba(feat) -> float` (P of class 1), `save(path)`, `load(path)`.
- `build_confidence_dataset(rows, route_fn, cards_by_name, domain_keywords) -> (feature_dicts, labels)`: `route_fn(utterance) -> (candidates, lex)`; label = 1 if `candidates[0].function` is in the row's `expected_functions` and the row isn't OOD, else 0.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_confidence_model.py
from t2f.safety.confidence import ExecutionConfidence, ConfidenceThresholds, build_confidence_dataset
from t2f.safety.features import FEATURE_ORDER
from t2f.types import Candidate, LexFeatures

def _feat(top1_score, has_req):
    d = {k: 0.0 for k in FEATURE_ORDER}
    d["top1_score"] = top1_score; d["margin"] = top1_score / 2; d["has_required_params"] = has_req
    return d

def test_fit_predict_separates():
    feats = [_feat(0.9, 1.0) for _ in range(8)] + [_feat(0.2, 0.0) for _ in range(8)]
    labels = [1] * 8 + [0] * 8
    m = ExecutionConfidence().fit(feats, labels)
    assert m.predict_proba(_feat(0.9, 1.0)) > m.predict_proba(_feat(0.2, 0.0))
    assert 0.0 <= m.predict_proba(_feat(0.5, 1.0)) <= 1.0

def test_build_dataset_labels():
    rows = [{"utterance": "a", "expected_functions": ["f1"], "type": "single"},
            {"utterance": "b", "expected_functions": [], "type": "ood"}]
    def route(u):
        fn = "f1" if u == "a" else "f9"
        return ([Candidate(fn, 0.7)], LexFeatures(raw=u))
    feats, labels = build_confidence_dataset(rows, route, {}, {})
    assert labels == [1, 0] and len(feats) == 2

def test_save_load_roundtrip(tmp_path):
    feats = [_feat(0.9, 1.0)] * 6 + [_feat(0.2, 0.0)] * 6
    m = ExecutionConfidence().fit(feats, [1]*6 + [0]*6)
    p = tmp_path / "c.joblib"; m.save(str(p))
    assert ExecutionConfidence.load(str(p)).predict_proba(_feat(0.9, 1.0)) > 0.5

def test_thresholds_defaults():
    t = ConfidenceThresholds()
    assert t.tau_low == 0.3 and t.tau_high == 0.7
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement**
```python
# t2f/safety/confidence.py
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
```
- [ ] **Step 4: Run test** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: execution-confidence model + dataset builder"`

---

## Task 3: Threshold calibration (`t2f/safety/confidence.py`)

**Files:** Modify `t2f/safety/confidence.py`; Test `tests/test_confidence_calibrate.py`

**Contract:** `calibrate_thresholds(points, target_error=0.05) -> ConfidenceThresholds`, where `points` is a list of `(p, correct: bool, is_ood: bool)`. `tau_high` = the lowest `p` cutoff whose *executed set* (points with `p ≥ cut`) has error rate (`not correct`, which includes OOD) ≤ `target_error`, maximizing executed coverage; if none qualifies, use the cutoff with the minimum error. `tau_low` = the lowest cutoff at which **no** OOD point remains above it (all OOD rejected), capped at `tau_high`. Pure function.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_confidence_calibrate.py
from t2f.safety.confidence import calibrate_thresholds

def test_calibrate_separates_and_rejects_ood():
    # in-domain correct at high P, wrong/ood at low P
    pts = [(0.9, True, False)] * 8 + [(0.85, True, False)] * 8 \
        + [(0.4, False, False)] * 4 + [(0.3, False, True)] * 6
    t = calibrate_thresholds(pts, target_error=0.05)
    assert t.tau_high <= 0.85 and t.tau_high > 0.4      # executes the clean high-P set
    # no OOD (max p 0.3) may sit at/above tau_low
    assert t.tau_low > 0.3
    assert t.tau_low <= t.tau_high

def test_calibrate_empty():
    from t2f.safety.confidence import ConfidenceThresholds
    assert isinstance(calibrate_thresholds([]), ConfidenceThresholds)
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement** — append to `t2f/safety/confidence.py`:
```python
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
```
- [ ] **Step 4: Run test** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: confidence threshold calibration (safety frontier)"`

---

## Task 4: Confidence-model gate (`t2f/gate.py`)

**Files:** Modify `t2f/gate.py`; Test `tests/test_confidence_gate.py`

**Contract:** `ConfidenceModelGate(model, thresholds, domain_keywords=None)` with `decide(candidates, features, cards_by_name) -> Decision` (same interface as `ConfidenceGate`). Compute `feat = confidence_features(candidates, features, cards_by_name, domain_keywords)`, `p = model.predict_proba(feat)`. Band: OOD marker top-1 → LOW; `p ≥ tau_high` → HIGH; `p < tau_low` → LOW (abstain); else MEDIUM. `chosen = candidates[0].function` for HIGH/MEDIUM, `None` for LOW. Store `p` in `Decision.features["p_correct"]`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_confidence_gate.py
from t2f.types import Candidate, LexFeatures, Band
from t2f.retrieve import OOD_MARKER
from t2f.safety.confidence import ExecutionConfidence, ConfidenceThresholds
from t2f.safety.features import FEATURE_ORDER
from t2f.gate import ConfidenceModelGate

def _model():
    def f(s):
        d = {k: 0.0 for k in FEATURE_ORDER}; d["top1_score"] = s; d["margin"] = s / 2; return d
    return ExecutionConfidence().fit([f(0.9)] * 8 + [f(0.1)] * 8, [1] * 8 + [0] * 8)

def test_bands_by_probability():
    g = ConfidenceModelGate(_model(), ConfidenceThresholds(tau_low=0.3, tau_high=0.6))
    hi = g.decide([Candidate("a", 0.95), Candidate("b", 0.4)], LexFeatures(raw="x"), {})
    lo = g.decide([Candidate("a", 0.05), Candidate("b", 0.02)], LexFeatures(raw="x"), {})
    assert hi.band == Band.HIGH and hi.chosen == "a"
    assert lo.band == Band.LOW and lo.chosen is None
    assert "p_correct" in hi.features

def test_ood_marker_rejected():
    g = ConfidenceModelGate(_model(), ConfidenceThresholds(0.3, 0.6))
    d = g.decide([Candidate(OOD_MARKER, 0.9), Candidate("a", 0.1)], LexFeatures(raw="x"), {})
    assert d.band == Band.LOW and d.chosen is None
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement** — append to `t2f/gate.py` (imports at top: `from .safety.features import confidence_features`; keep `OOD_MARKER` already imported):
```python
class ConfidenceModelGate:
    """Bands on a learned P(top-1 correct) instead of a raw score threshold. Same decide() shape."""

    def __init__(self, model, thresholds, domain_keywords=None):
        self.model = model
        self.t = thresholds
        self.domain_keywords = domain_keywords or {}

    def decide(self, candidates, features, cards_by_name):
        if not candidates:
            return Decision(Band.LOW, None, [], ood_score=1.0, features={})
        if candidates[0].function == OOD_MARKER:
            return Decision(Band.LOW, None, candidates, ood_score=1.0, features={"ood_marker": 1.0})
        feat = confidence_features(candidates, features, cards_by_name, self.domain_keywords)
        p = self.model.predict_proba(feat)
        info = {"p_correct": p}
        if p < self.t.tau_low:
            return Decision(Band.LOW, None, candidates, ood_score=1.0 - p, features=info)
        if p >= self.t.tau_high:
            return Decision(Band.HIGH, candidates[0].function, candidates, ood_score=1.0 - p, features=info)
        return Decision(Band.MEDIUM, candidates[0].function, candidates, ood_score=1.0 - p, features=info)
```
(Note: `t2f/gate.py` already imports `from .retrieve import OOD_MARKER` and `from .types import Candidate, Decision, Band, LexFeatures`. Add only the `confidence_features` import. This creates an import edge gate→safety.features→params/retrieve; safety.features does not import gate, so no cycle.)

- [ ] **Step 4: Run test** → PASS; run full `python3 -m pytest -q` (green)
- [ ] **Step 5: Commit** — `git commit -am "feat: confidence-model gate"`

---

## Task 5: Coverage metric + frontier (`eval/metrics.py`)

**Files:** Modify `eval/metrics.py`; Test `tests/test_metrics_spec3.py`

**Contract (append):**
- `coverage(records) -> float`: fraction of in-scope (single/multi_intent/ambiguous) rows where **every** clause executed (`executed[i]` all True).
- `frontier(records_by_tau) -> list[dict]`: given `{tau: records}`, return per-τ `{tau, ood_false_execution, incorrect_execution, e2e, coverage, clarification_rate}` using existing metric fns.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_metrics_spec3.py
from eval.metrics import coverage

def _rec(typ, executed):
    return {"row": {"type": typ}, "executed": executed, "bands": ["high"] * len(executed),
            "exec_correct": [True] * len(executed)}

def test_coverage_counts_fully_executed_rows():
    recs = [_rec("single", [True]), _rec("single", [False]),
            _rec("multi_intent", [True, False]), _rec("ood", [False])]
    # in-scope = 3 rows (single,single,multi); fully executed = 1
    assert abs(coverage(recs) - (1 / 3)) < 1e-9
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement** — append to `eval/metrics.py`:
```python
def coverage(records) -> float:
    rows = [r for r in records if r["row"].get("type") in ("single", "multi_intent", "ambiguous")]
    if not rows:
        return 0.0
    done = sum(1 for r in rows if r["executed"] and all(r["executed"]))
    return done / len(rows)


def frontier(records_by_tau) -> list:
    out = []
    for tau, recs in records_by_tau.items():
        out.append({
            "tau": tau,
            "ood_false_execution": ood_false_execution_rate(recs),
            "incorrect_execution": incorrect_execution_rate(recs),
            "e2e": e2e_executable_accuracy(recs, "deterministic"),
            "coverage": coverage(recs),
            "clarification_rate": clarification_rate(recs),
        })
    return sorted(out, key=lambda d: d["tau"])
```
- [ ] **Step 4: Run test** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: coverage metric + safety/coverage frontier"`

---

## Task 6: Hard-negative mining tool (`t2f/tools/mine_hard_negatives.py`)

**Files:** Create `t2f/tools/__init__.py`, `t2f/tools/mine_hard_negatives.py`; Test `tests/test_mine_hard_negatives.py`

**Contract:** `mine_confusions(rows, route_fn) -> list[dict]`: for each single/ambiguous row whose gold function is NOT the routed top-1 but appears later in the ranked candidates, record `{"gold": g, "distractor": top1, "utterance": u}`. `summarize(confusions) -> list[(pair, count)]` aggregates `(gold, distractor)` pairs by frequency, most-confused first. `route_fn(utterance) -> list[str]` returns ranked candidate function names.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_mine_hard_negatives.py
from t2f.tools.mine_hard_negatives import mine_confusions, summarize

def test_mine_and_summarize():
    rows = [{"utterance": "u1", "expected_functions": ["set_temperature"], "type": "single"},
            {"utterance": "u2", "expected_functions": ["set_temperature"], "type": "single"},
            {"utterance": "u3", "expected_functions": ["open_window"], "type": "single"}]
    ranked = {"u1": ["set_fan_speed", "set_temperature"],   # gold below distractor
              "u2": ["set_fan_speed", "set_temperature"],
              "u3": ["open_window"]}                          # correct top1, not a confusion
    conf = mine_confusions(rows, lambda u: ranked[u])
    assert len(conf) == 2 and all(c["distractor"] == "set_fan_speed" for c in conf)
    top = summarize(conf)
    assert top[0][0] == ("set_temperature", "set_fan_speed") and top[0][1] == 2
```

- [ ] **Step 2: Run test** → FAIL
- [ ] **Step 3: Implement**
```python
# t2f/tools/__init__.py
```
```python
# t2f/tools/mine_hard_negatives.py
from __future__ import annotations
from collections import Counter


def mine_confusions(rows, route_fn):
    out = []
    for r in rows:
        if r.get("type") not in ("single", "ambiguous"):
            continue
        gold = r.get("expected_functions", [])
        ranked = route_fn(r["utterance"])
        if not ranked or ranked[0] in gold:
            continue
        if any(g in ranked for g in gold):   # gold present but ranked below the distractor
            out.append({"gold": gold[0], "distractor": ranked[0], "utterance": r["utterance"]})
    return out


def summarize(confusions):
    c = Counter((x["gold"], x["distractor"]) for x in confusions)
    return c.most_common()
```
- [ ] **Step 4: Run test** → PASS
- [ ] **Step 5: Commit** — `git commit -am "feat: hard-negative mining tool"`

---

## Task 7: Integration — train confidence gate, frontier eval, mining, results

**Files:** Modify `docs/superpowers/RESULTS.md`, possibly small additions to `data/catalog/*.yaml`; Create `data/analysis/hard_negatives.md`; Test `tests/test_integration_spec3.py`

- [ ] **Step 1: Write the integration test (fake path, always runs)**
```python
# tests/test_integration_spec3.py
def test_confidence_gate_pipeline_fake():
    from t2f.cards import load_catalog
    from t2f.embed import FakeEmbedder
    from t2f.config import Config
    from t2f.score import Scorer
    from t2f.pipeline import Pipeline
    from t2f.safety.features import FEATURE_ORDER
    from t2f.safety.confidence import ExecutionConfidence, ConfidenceThresholds
    from t2f.gate import ConfidenceModelGate
    cards = load_catalog("data/catalog")
    cfg = Config.default()
    def f(s):
        d = {k: 0.0 for k in FEATURE_ORDER}; d["top1_score"] = s; return d
    model = ExecutionConfidence().fit([f(0.9)] * 8 + [f(0.1)] * 8, [1] * 8 + [0] * 8)
    gate = ConfidenceModelGate(model, ConfidenceThresholds(0.3, 0.6), cfg.domain_keywords)
    pipe = Pipeline(cards, FakeEmbedder(256), Scorer(cfg.weights, cfg.domain_keywords), gate, cfg)
    res = pipe.route("把空调调到25度")
    assert len(res.clauses) == 1 and res.clauses[0].decision.band is not None
```

- [ ] **Step 2: Run the fake test** → `python3 -m pytest tests/test_integration_spec3.py -q` PASS

- [ ] **Step 3: Train, calibrate, sweep the frontier (real embedder; reuse Spec-2 classifier + LLM)**

Write a script (in the session scratchpad, not committed) that:
1. Loads the real `TransformersEmbedder`, cards, OOD prototypes, and the trained char-ngram classifier; builds a Scorer.
2. Defines `route_fn(utt)` returning `(rescored_candidates, lex)` for the first clause (with classifier union + OOD prototypes, matching Arm D).
3. `build_confidence_dataset` over gold **dev** + the 96 OOD prototypes (as `type:"ood"` rows) → fit `ExecutionConfidence`; save to `models/confidence.joblib`.
4. Builds `(p, correct, is_ood)` points over dev; `calibrate_thresholds(points, target_error=0.05)`.
5. Builds Arm-D-style pipelines using `ConfidenceModelGate` at several τ operating points (the calibrated one plus a sweep, e.g. tau_high ∈ {0.5,0.6,0.7,0.8}); runs the real LLM medium band on **test**; computes `frontier(...)` (OOD false-exec, incorrect-exec, e2e, coverage, clarification-rate) at each.
6. Runs `mine_confusions`/`summarize` on dev; writes `data/analysis/hard_negatives.md` (top confusable pairs + counts). If clear confusable clusters emerge, add a few **discriminative** utterances/aliases to the relevant `data/catalog/*.yaml` (dev-guided only) and re-measure recall@1/@3 on test.

- [ ] **Step 4: Write the RESULTS "Spec 3" section** — add a frontier table (τ vs OOD-false-exec / incorrect-exec / e2e / coverage / clarification-rate), the recommended operating point, the confidence gate vs the Spec-2 heuristic-gate baseline (must not regress OOD/incorrect at matched coverage), and any recall@3 lift from hard-negative additions. For any target still missed at the recommended τ, state the gap + lever (do not silently pass). Run full `python3 -m pytest -q` (green) and `python3 -m pytest -m model -q` (if available).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "eval: spec3 confidence gate + safety/coverage frontier + hard-negative report"`

---

## Self-Review (completed by plan author)

**Spec coverage:** confidence features (T1), model + dataset (T2), calibration/frontier thresholds (T3), learned gate integration (T4), coverage + frontier metrics (T5), hard-negative mining (T6), train/calibrate/frontier eval + recall hardening + RESULTS (T7). Non-goals (on-device, new large models, new LLM/data) excluded. The per-LLM-pick confidence check mentioned in the spec's §3.3 is intentionally deferred — the gate's pre-LLM abstention is the primary control and is what the plan builds/measures; noted as future work in T7's write-up.

**Placeholder scan:** T7 (real training + frontier sweep + optional catalog additions) specifies the procedure + files + targets rather than inlining model outputs or a variable number of prototypes; all logic tasks (T1–T6) carry complete code.

**Type consistency:** `FEATURE_ORDER`, `confidence_features(candidates, lex, cards_by_name, domain_keywords)`, `ConfidenceThresholds(tau_low, tau_high)`, `ExecutionConfidence.fit/predict_proba/save/load`, `build_confidence_dataset(rows, route_fn, cards_by_name, domain_keywords)` with `route_fn->(candidates, lex)`, `calibrate_thresholds(points, target_error)`, `ConfidenceModelGate(model, thresholds, domain_keywords).decide(candidates, features, cards_by_name)`, and the mining `route_fn->list[str]` are used consistently across T1–T7. The gate reuses the existing `Decision(Band, chosen, candidates, ood_score, features)` shape, so `Pipeline`/`predict` consume it unchanged.
