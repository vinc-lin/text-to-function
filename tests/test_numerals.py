# tests/test_numerals.py
from t2f.params.numerals import parse_number, find_numbers

def test_arabic():
    assert parse_number("25") == 25.0
    assert parse_number("3.5") == 3.5

def test_chinese_basic():
    assert parse_number("二十五") == 25.0
    assert parse_number("两") == 2.0
    assert parse_number("十") == 10.0
    assert parse_number("一百二十") == 120.0
    assert parse_number("零") == 0.0

def test_parse_number_invalid():
    assert parse_number("abc") is None

def test_find_numbers_mixed():
    assert find_numbers("从20调到二十五度") == [20.0, 25.0]
    assert find_numbers("没有数字") == []
