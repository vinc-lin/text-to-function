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


def test_ood_marker_top1_is_rejected():
    """When an OOD prototype wins retrieval, the gate rejects (LOW) without executing."""
    from t2f.embed import FakeEmbedder
    from t2f.retrieve import PrototypeStore, Retriever, OOD_MARKER
    from t2f.gate import ConfidenceGate, Thresholds, Band
    from t2f.types import LexFeatures
    cards = [FunctionCard("open_window", "window", "打开车窗", utterances=["开车窗", "把窗户打开"])]
    emb = FakeEmbedder(256)
    store = PrototypeStore.build(cards, emb, ood_texts=["今天天气怎么样", "讲个笑话"])
    retr = Retriever(store)
    gate = ConfidenceGate(Thresholds(0.5, 0.05, 0.2))
    # a chitchat query matches the OOD prototype -> OOD_MARKER top1 -> rejected
    q = emb.encode(["今天天气怎么样"], is_query=True)[0]
    cands = retr.retrieve(q, top_k=3)
    assert cands[0].function == OOD_MARKER
    d = gate.decide(cands, LexFeatures(), {})
    assert d.band == Band.LOW and d.chosen is None


class _StubPipe:
    def __init__(self, clause_result):
        self._cr = clause_result
    def route(self, utterance):
        from t2f.types import RouteResult
        return RouteResult(utterance=utterance, clauses=[self._cr])


def test_predict_scores_executed_function_not_top1():
    """exec_correct/incorrect-execution must reflect the function the LLM EXECUTED, not retrieval top1."""
    from t2f.types import ClauseResult, ToolCall
    from eval.arms import predict
    from eval.metrics import incorrect_execution_rate
    # top1 is the WRONG function; the LLM executed the CORRECT non-top1 candidate -> should score right
    dec = Decision(Band.MEDIUM, "set_fan_speed",
                   [Candidate("set_fan_speed", 0.5), Candidate("set_temperature", 0.45)])
    cr = ClauseResult(clause="x", decision=dec,
                      tool_call=ToolCall("set_temperature", {"temperature": 25}),
                      response="ok", needs_llm=True)
    rec = predict(_StubPipe(cr), {"utterance": "x", "expected_functions": ["set_temperature"],
                                  "expected_params": {"set_temperature": {"temperature": 25}}, "type": "single"})
    assert rec["predicted_functions"][0] == "set_temperature"
    assert rec["exec_correct"][0] is True
    assert incorrect_execution_rate([rec]) == 0.0

    # top1 is the gold function but the LLM executed a WRONG candidate -> counts as incorrect execution
    dec2 = Decision(Band.MEDIUM, "set_temperature",
                    [Candidate("set_temperature", 0.5), Candidate("set_fan_speed", 0.45)])
    cr2 = ClauseResult(clause="x", decision=dec2, tool_call=ToolCall("set_fan_speed", {"level": 3}),
                       response="ok", needs_llm=True)
    rec2 = predict(_StubPipe(cr2), {"utterance": "x", "expected_functions": ["set_temperature"], "type": "single"})
    assert rec2["exec_correct"][0] is False
    assert incorrect_execution_rate([rec2]) == 1.0
