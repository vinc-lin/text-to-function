"""Regression tests for issues found in final code review."""
from t2f.segment import split
from t2f.normalize import normalize
from t2f.types import FunctionCard, ToolCall, ParamSpec, Decision, Band, LexFeatures, ValidationError
from t2f.respond import render_response
from t2f.pipeline import DeterministicResolver
from eval.metrics import schema_valid_rate, e2e_executable_accuracy


# #4 — segmenter must not split decimals / frequencies
def test_segment_keeps_decimals():
    assert split(normalize("温度调到22.5度")) == ["温度调到22.5度"]
    assert split(normalize("收音机调到FM101.7")) == ["收音机调到fm101.7"]
    # a genuine sentence-final period (from CJK 。) between non-digits still splits
    assert split("开窗.开门") == ["开窗", "开门"]


# #5 — integral numeric params render without a trailing ".0"
def test_render_integral_number_no_trailing_zero():
    card = FunctionCard("set_temperature", "climate", "温度",
                        params=[ParamSpec("temperature", "number", unit="celsius")],
                        response_template="已将温度设置为{temperature}°C。")
    r = render_response(card, ToolCall("set_temperature", {"temperature": 25.0}))
    assert r == "已将温度设置为25°C。"


# #3 — LOW band produces a clarification (not silence)
def test_low_band_produces_clarification():
    res = DeterministicResolver({}).resolve("乱七八糟", LexFeatures(),
                                            Decision(Band.LOW, None, [], ood_score=1.0))
    assert res.tool_call is None
    assert res.clarification is not None and res.clarification.question
    assert res.needs_llm is False


# #1 — schema_valid_rate counts only validated calls (clarifications excluded)
def test_schema_valid_rate_excludes_clarifications():
    rec = {"bands": ["medium", "medium", "high"],
           "tool_calls": [ToolCall("a", {}), None, None],
           "val_errors": [[], [ValidationError("x", "y")], []]}  # valid, invalid, clarification
    # 1 valid of 2 validated -> 0.5; the third (no tc, no errs) is not a schema failure
    assert schema_valid_rate([rec]) == 0.5


# #2 — deterministic e2e requires actual execution, not merely a HIGH band
def test_e2e_deterministic_requires_execution():
    rec = {"row": {"type": "single", "expected_functions": ["a"]},
           "bands": ["high"], "executed": [False], "exec_correct": [True],
           "ranked_per_clause": [["a", "b"]]}
    assert e2e_executable_accuracy([rec], "deterministic") == 0.0   # high but not executed
    assert e2e_executable_accuracy([rec], "ceiling") == 1.0         # gold in top-3 -> LLM ceiling
