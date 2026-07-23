"""Regression for the value-clobber bug: a clause with BOTH a relative verb (调高/调低) AND an
explicit value must be treated as ABSOLUTE, so the StateResolver never overwrites the stated value."""
from t2f.pipeline import Pipeline
from t2f.lexical import extract_features


def test_explicit_value_with_relative_verb_is_absolute():
    # 调高/调低 set feats.operation, but the explicit value makes the command absolute
    assert Pipeline._relative_spec(extract_features("温度调高到26度")) is None
    assert Pipeline._relative_spec(extract_features("亮度调低到30%")) is None
    assert Pipeline._relative_spec(extract_features("天窗开到一半")) is None  # 一半 -> percent 50


def test_pure_relative_without_value_stays_relative():
    r1 = Pipeline._relative_spec(extract_features("音量调小一点"))
    assert r1 is not None and r1.operation == "decrease"
    r2 = Pipeline._relative_spec(extract_features("主驾这边窗户再开一点"))
    assert r2 is not None and r2.operation == "increase"
