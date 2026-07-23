from t2f.params.numerals import parse_fraction_percent

def test_common_fractions():
    assert parse_fraction_percent("天窗开到一半") == 50
    assert parse_fraction_percent("开个三分之一") == 33
    assert parse_fraction_percent("开四分之三") == 75
    assert parse_fraction_percent("留个缝") is None      # no fraction
    assert parse_fraction_percent("开到百分之三十") is None  # explicit % handled elsewhere
