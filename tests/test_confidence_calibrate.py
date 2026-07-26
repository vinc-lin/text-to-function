from research.safety.confidence import calibrate_thresholds, ConfidenceThresholds


def test_calibrate_two_band_keeps_medium_zone():
    # in-domain correct at high p; in-domain wrong at mid p; OOD at low-mid p
    pts = ([(0.9, True, False)] * 10 + [(0.85, True, False)] * 10
           + [(0.5, False, False)] * 4 + [(0.45, False, True)] * 3 + [(0.2, False, True)] * 7)
    t = calibrate_thresholds(pts, target_error=0.05, ood_budget_high=0.10, ood_budget_low=0.30)
    assert t.tau_low < t.tau_high            # a real medium/LLM band survives
    assert t.tau_high >= 0.8                  # direct-execute floor excludes the wrong/OOD mid set
    assert 0.3 < t.tau_low < t.tau_high       # reject floor sits below the execute floor


def test_calibrate_empty():
    assert isinstance(calibrate_thresholds([]), ConfidenceThresholds)
