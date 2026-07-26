"""Validator rules for the new e2e row types (`invalid`, `asr_noise`).

The `expected_params` rule is the load-bearing one: `param_exact_match` is NOT
type-filtered (eval/metrics.py:33-43), so an `expected_params` on a new-type row would
contaminate a headline metric even though the rows live in a separate file.
"""
from eval.dataset import validate_against_catalog

NAMES = {"set_temperature"}


def test_invalid_row_requires_a_cause():
    rows = [{"utterance": "把温度调到99度", "type": "invalid",
             "expected_functions": ["set_temperature"], "expected_execution": False}]
    assert any("expected_cause" in p for p in validate_against_catalog(rows, NAMES))


def test_invalid_row_must_forbid_execution():
    rows = [{"utterance": "x", "type": "invalid", "expected_functions": ["set_temperature"],
             "expected_cause": "out_of_range"}]
    assert any("expected_execution" in p for p in validate_against_catalog(rows, NAMES))


def test_unknown_cause_is_rejected():
    rows = [{"utterance": "x", "type": "invalid", "expected_functions": ["set_temperature"],
             "expected_execution": False, "expected_cause": "banana"}]
    assert any("banana" in p for p in validate_against_catalog(rows, NAMES))


def test_new_types_must_not_carry_expected_params():
    rows = [{"utterance": "x", "type": "asr_noise", "expected_functions": ["set_temperature"],
             "source_utterance": "y", "expected_params": {"set_temperature": {"temperature": 25}}}]
    assert any("expected_params" in p for p in validate_against_catalog(rows, NAMES))


def test_asr_noise_requires_a_source():
    rows = [{"utterance": "x", "type": "asr_noise", "expected_functions": ["set_temperature"]}]
    assert any("source_utterance" in p for p in validate_against_catalog(rows, NAMES))


def test_a_well_formed_invalid_row_passes():
    rows = [{"utterance": "把温度调到99度", "type": "invalid",
             "expected_functions": ["set_temperature"], "expected_execution": False,
             "expected_cause": "out_of_range"}]
    assert validate_against_catalog(rows, NAMES) == []


def test_a_well_formed_asr_noise_row_passes():
    rows = [{"utterance": "把空调调到二十五都", "type": "asr_noise",
             "expected_functions": ["set_temperature"],
             "source_utterance": "把空调调到二十五度"}]
    assert validate_against_catalog(rows, NAMES) == []


def test_existing_row_types_are_unaffected():
    """Regression guard: the new rules must not reject the shapes gold.jsonl already uses."""
    rows = [{"utterance": "把空调调到25度", "type": "single",
             "expected_functions": ["set_temperature"],
             "expected_params": {"set_temperature": {"temperature": 25}}},
            {"utterance": "讲个笑话", "type": "ood", "expected_functions": []}]
    assert validate_against_catalog(rows, NAMES) == []
