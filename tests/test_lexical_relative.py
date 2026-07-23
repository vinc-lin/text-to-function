from t2f.lexical import extract_features


def test_fraction_becomes_percent():
    f = extract_features("天窗开到一半")
    assert 50 in f.percentages


def test_relative_open_a_bit():
    f = extract_features("主驾这边窗户再开一点")
    assert f.operation == "increase" and f.amount == "small"


def test_relative_lower_volume():
    f = extract_features("音量调小一点")
    assert f.operation == "decrease" and f.amount == "small"


def test_absolute_not_relative():
    f = extract_features("把温度调到22度")
    assert f.amount is None and 22 in f.temperatures
