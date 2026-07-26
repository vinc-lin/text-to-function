"""Offline threshold calibration — grid-search over a dev split.

Moved out of `t2f/gate.py`: it never runs in the vehicle (thresholds arrive pre-baked in
config.yaml), and the runtime package should contain only what `Pipeline.route()` can reach.
"""
from __future__ import annotations
from t2f.gate import Thresholds, ConfidenceGate
from t2f.types import Band, LexFeatures


def calibrate_gate(dev_rows, route_top_candidates, target_high_precision: float = 0.98,
                   execute_medium: bool = False) -> Thresholds:
    """Grid-search thresholds on dev rows.
    dev_rows: list of dicts {utterance, expected_functions, type}.
    route_top_candidates(utterance) -> list[Candidate] (already hybrid-scored).

    Default (execute_medium=False, Spec-1 zero-LLM arms): only the HIGH band executes, so the
    objective is to maximize HIGH-band coverage subject to HIGH-band top-1 precision >= target and
    no OOD in HIGH.

    execute_medium=True (Spec-2 LLM arms, where the MEDIUM band is executed by the LLM): OOD that
    reaches HIGH *or* MEDIUM would be executed, so the constraint tightens to "no OOD in any
    executing band" — i.e. OOD must fall to LOW (rejected). low_top1 becomes the OOD floor. The
    objective then maximizes in-domain rows routed into the executing bands (coverage) subject to
    that OOD-safety constraint and the HIGH precision constraint.

    The candidate list for an utterance is independent of the thresholds, so it is computed ONCE
    per dev row and cached — the grid search only re-bands the cached candidates.
    """
    cand_cache: dict[str, list] = {}
    for r in dev_rows:
        u = r["utterance"]
        if u not in cand_cache:
            cand_cache[u] = route_top_candidates(u)

    low_grid = ([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
                if execute_medium else [0.15, 0.20, 0.25, 0.30, 0.35])

    # Evaluate every threshold combo once, collecting the stats both objectives need.
    combos = []
    for high_top1 in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for high_margin in [0.02, 0.03, 0.05, 0.08, 0.12]:
            for low_top1 in low_grid:
                if low_top1 >= high_top1:
                    continue
                gate = ConfidenceGate(Thresholds(high_top1, high_margin, low_top1))
                high_total = high_correct = ood_in_high = ood_in_exec = ind_exec = 0
                for r in dev_rows:
                    d = gate.decide(cand_cache[r["utterance"]], LexFeatures(), {})
                    is_ood = r.get("type") == "ood"
                    executing = d.band in (Band.HIGH, Band.MEDIUM)
                    if d.band == Band.HIGH:
                        high_total += 1
                        if is_ood:
                            ood_in_high += 1
                        elif d.chosen in r.get("expected_functions", []):
                            high_correct += 1
                    if is_ood and executing:
                        ood_in_exec += 1
                    elif not is_ood and executing and r.get("expected_functions"):
                        ind_exec += 1
                if high_total == 0 or ood_in_high > 0:
                    continue
                combos.append({
                    "t": Thresholds(high_top1, high_margin, low_top1),
                    "prec": high_correct / high_total, "high_total": high_total,
                    "ood_in_exec": ood_in_exec, "ind_exec": ind_exec,
                })
    if not combos:
        return Thresholds()

    if execute_medium:
        # Among combos that meet HIGH precision and leak zero OOD into an executing band, pick a
        # CONSERVATIVE operating point: the highest low_top1 (largest OOD safety margin against the
        # dev/test gap) that still keeps in-domain coverage within 90% of the achievable maximum.
        # Biasing toward rejection over execution is the PRD's stated preference for the risky path.
        safe = [c for c in combos if c["prec"] >= target_high_precision and c["ood_in_exec"] == 0]
        if safe:
            max_cov = max(c["ind_exec"] for c in safe)
            near = [c for c in safe if c["ind_exec"] >= 0.9 * max_cov]
            best = max(near, key=lambda c: (c["t"].low_top1, c["ind_exec"]))
        else:  # fallback: least OOD leak, then most in-domain coverage
            best = min(combos, key=lambda c: (c["ood_in_exec"], -c["ind_exec"]))
        return best["t"]

    # zero-LLM arms: maximize HIGH coverage subject to precision; fallback = best precision/coverage
    ok = [c for c in combos if c["prec"] >= target_high_precision]
    pool = ok if ok else combos
    key = (lambda c: (c["high_total"], c["prec"])) if ok else (lambda c: (c["prec"], c["high_total"]))
    return max(pool, key=key)["t"]
