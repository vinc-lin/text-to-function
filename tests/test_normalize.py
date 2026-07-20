# tests/test_normalize.py
from t2f.normalize import normalize

def test_fullwidth_and_punct():
    assert normalize("把空调调到２５度！") == "把空调调到25度!"

def test_whitespace_and_latin_lower():
    assert normalize("  AC  ON  ") == "ac on"

def test_fullwidth_comma_unified():
    assert normalize("开窗，开空调") == "开窗,开空调"
