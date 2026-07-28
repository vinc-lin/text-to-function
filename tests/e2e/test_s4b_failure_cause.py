"""S4b — on failure, explain the SPECIFIC cause.

Today every hard failure collapses into one constant. These cases separate the two halves:
nothing-executed (green, already true) from cause-explained (red, gap 2).
"""
import pytest

from t2f.llm.client import FakeLLMClient
from t2f.types import LLMResult, ToolCall

from .conftest import build_pipeline

GENERIC = "抱歉，这个操作没能完成。"
GENERIC_QUESTION = "请补充更多信息。"
GAP2 = ("gap 2: t2f/reply.py never reads ClauseResult.validation_errors, so every cause "
        "collapses into one constant. The cause data exists and reaches the reply layer.")
GAP3 = ("gap 3: _CLARIFY (t2f/respond.py:7-9) knows 3 parameter names, covering 10 of the "
        "catalog's 76 required-parameter slots. `is_open` is not one of them.")

# (case id, utterance) — each fails validation with the cause named in the id
NO_EXECUTION = [
    ("S4B-03 out_of_range above max", "把温度调到99度"),
    ("S4B-04 out_of_range below min", "把温度调到5度"),
    ("S4B-05 out_of_range integer", "风速调到20档"),
]


@pytest.mark.parametrize("case_id,utterance", NO_EXECUTION, ids=[c[0] for c in NO_EXECUTION])
def test_s4b_invalid_value_dispatches_nothing(case_id, utterance):
    """GREEN — the safety half. An unusable value must never reach the vehicle."""
    pipe, ex = build_pipeline()
    pipe.route(utterance)
    assert ex.dispatched == []


@pytest.mark.parametrize("case_id,utterance", NO_EXECUTION, ids=[c[0] for c in NO_EXECUTION])
def test_s4b_invalid_value_explains_the_cause(case_id, utterance):
    """The explanation half — was red until the validation cause table landed."""
    pipe, _ = build_pipeline()
    assert pipe.route(utterance).reply != GENERIC


def test_s4b_03_out_of_range_names_the_limit():
    """The bounds are in the card (minimum 16, maximum 32) and never spoken."""
    pipe, _ = build_pipeline()
    reply = pipe.route("把温度调到99度").reply
    assert "16" in reply and "32" in reply


def test_s4b_02_missing_required_param_dispatches_nothing():
    """GREEN — a missing required parameter must not be guessed."""
    pipe, ex = build_pipeline()
    pipe.route("车窗")
    assert ex.dispatched == []


def test_s4b_02_missing_required_param_names_what_is_missing():
    """The question names the parameter — was red until build_clarification used the catalog."""
    pipe, _ = build_pipeline()
    assert pipe.route("车窗").reply != GENERIC_QUESTION


# --- causes that only the LLM path can produce -------------------------------------------
# bad_enum and type_mismatch never come out of the deterministic extractors, so these drive
# the medium band with a scripted fake client.


def _bad_enum_llm():
    return FakeLLMClient(default=LLMResult(
        tool_call=ToolCall("set_temperature", {"temperature": 25, "position": "trunk"})))


def test_s4b_06_bad_enum_dispatches_nothing():
    """GREEN — safety half, kept OUT of the xfail body.

    An assertion inside an xfail is unguarded: xfail(strict) reports the same result
    whichever line trips, so a regression here would be absorbed by the expected failure
    below. The safety half therefore lives in its own green test.
    """
    pipe, ex = build_pipeline(llm_client=_bad_enum_llm())
    pipe.route("温度调高一点")
    assert ex.dispatched == []


def test_s4b_06_bad_enum_explains_the_cause():
    """The card's enum list is now spoken, not just computed."""
    pipe, _ = build_pipeline(llm_client=_bad_enum_llm())
    assert pipe.route("温度调高一点").reply != GENERIC


def _type_mismatch_llm():
    return FakeLLMClient(default=LLMResult(
        tool_call=ToolCall("set_temperature", {"temperature": "warm"})))


def test_s4b_07_type_mismatch_dispatches_nothing():
    """GREEN — safety half, kept out of the xfail body (see test_s4b_06 above)."""
    pipe, ex = build_pipeline(llm_client=_type_mismatch_llm())
    pipe.route("温度调高一点")
    assert ex.dispatched == []


def test_s4b_07_type_mismatch_explains_the_cause():
    """RED — 'temperature must be numeric' is computed and never spoken."""
    pipe, _ = build_pipeline(llm_client=_type_mismatch_llm())
    assert pipe.route("温度调高一点").reply != GENERIC
