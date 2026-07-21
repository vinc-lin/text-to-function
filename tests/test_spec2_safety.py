"""Regression tests for the Spec-2 OOD-safety fix: reject escape hatch + OOD-aware gate."""
from t2f.types import FunctionCard, ParamSpec, Candidate, Decision, Band, LexFeatures, LLMResult
from t2f.llm.schema import candidates_to_json_schema, REJECT_NAME
from t2f.llm.client import FakeLLMClient, _parse_tool_call
from t2f.pipeline import LLMResolver
from t2f.gate import calibrate_gate, ConfidenceGate
from t2f.lexical import extract_features

CARD = FunctionCard("set_temperature", "climate", "温度",
    params=[ParamSpec("temperature", "number", required=True, minimum=16, maximum=32)])
CARDS = {"set_temperature": CARD}


def test_schema_includes_reject_option():
    s = candidates_to_json_schema([CARD], allow_reject=True)
    names = {o["properties"]["name"]["const"] for o in s["oneOf"]}
    assert REJECT_NAME in names and "set_temperature" in names
    # without the flag, no reject option
    assert "oneOf" not in candidates_to_json_schema([CARD], allow_reject=False)


def test_parse_reject_becomes_clarification_not_toolcall():
    r = _parse_tool_call('{"name": "__reject__"}')
    assert r.tool_call is None and r.clarification == REJECT_NAME


def test_resolver_reject_does_not_execute():
    client = FakeLLMClient(default=LLMResult(clarification=REJECT_NAME))
    decision = Decision(Band.MEDIUM, "set_temperature",
                        [Candidate("set_temperature", 0.4), Candidate("set_fan_speed", 0.3)])
    r = LLMResolver(client).resolve("今天天气怎么样", extract_features("今天天气怎么样"),
                                    decision, CARDS, executor=None)
    assert r.tool_call is None            # never executes an OOD request
    assert r.clarification is not None     # asks/declines instead


def test_calibrate_execute_medium_routes_ood_to_low():
    # in-domain rows score high; OOD rows score low. execute_medium must pick a low_top1 that
    # rejects the OOD (top1 below it) while keeping in-domain in an executing band.
    def make(fn, score):
        return [Candidate(fn, score), Candidate("other", score - 0.2)]
    dev = ([{"utterance": f"ind{i}", "type": "single", "expected_functions": ["set_temperature"]} for i in range(8)]
           + [{"utterance": f"ood{i}", "type": "ood", "expected_functions": []} for i in range(4)])
    cache = {}
    for i in range(8):
        cache[f"ind{i}"] = make("set_temperature", 0.7)
    for i in range(4):
        cache[f"ood{i}"] = make("set_temperature", 0.25)  # spurious low match
    thr = calibrate_gate(dev, lambda u: cache[u], execute_medium=True)
    gate = ConfidenceGate(thr)
    # every OOD row must be LOW (rejected), every in-domain row must execute (HIGH or MEDIUM)
    for i in range(4):
        assert gate.decide(cache[f"ood{i}"], LexFeatures(), {}).band == Band.LOW
    for i in range(8):
        assert gate.decide(cache[f"ind{i}"], LexFeatures(), {}).band in (Band.HIGH, Band.MEDIUM)
