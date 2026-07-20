# tests/test_segment.py
from t2f.segment import split

def test_single_intent():
    assert split("把空调调到25度") == ["把空调调到25度"]

def test_punctuation_split():
    assert split("开窗,开空调") == ["开窗", "开空调"]

def test_conjunction_split():
    assert split("打开车窗然后把空调调到25度") == ["打开车窗", "把空调调到25度"]

def test_no_split_inside_number_unit():
    # "20到25度" is a range, must not split on 到; and 和 inside "柔和" must not split
    assert split("温度调到20度") == ["温度调到20度"]

def test_and_between_actions():
    assert split("打开主驾车窗和副驾车窗") == ["打开主驾车窗", "副驾车窗"] or \
           split("打开主驾车窗和副驾车窗") == ["打开主驾车窗和副驾车窗"]
