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
