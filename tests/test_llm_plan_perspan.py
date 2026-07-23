from t2f.pipeline import Pipeline
from t2f.embed import FakeEmbedder
from t2f.score import Scorer
from t2f.gate import ConfidenceGate, Thresholds
from t2f.config import Config
from t2f.cards import load_catalog
from t2f.llm.client import FakeLLMClient
from t2f.types import LLMResult, ToolCall
from t2f.llm.schema import REJECT_NAME


def _plan_pipe(llm_client):
    cfg = Config.load("config.yaml")
    cards = load_catalog("data/catalog")
    # thresholds so no span is HIGH -> the plan path is taken and routed through the LLM
    gate = ConfidenceGate(Thresholds(high_top1=2.0, high_margin=2.0, low_top1=0.0))
    pipe = Pipeline(cards, FakeEmbedder(256), Scorer(cfg.weights, cfg.domain_keywords), gate, cfg)
    pipe.llm_client = llm_client
    return pipe


def test_llm_abstention_defers_every_span():
    # LLM abstains (__reject__) on every span -> nothing executes, all deferred (context/OOD safety)
    pipe = _plan_pipe(FakeLLMClient(default=LLMResult(clarification=REJECT_NAME)))
    rr = pipe.route("开车窗，把空调打开")
    assert rr.plan is not None and rr.plan.source == "llm"
    assert all(a.function is None for a in rr.plan.actions)
    assert not any(a.status == "executed" for a in rr.plan.actions)


def test_llm_confirmation_executes_span():
    # LLM confirms a concrete tool call -> that span executes via the barrier
    client = FakeLLMClient(default=LLMResult(tool_call=ToolCall(name="set_ac_power", parameters={"enabled": True})))
    pipe = _plan_pipe(client)
    rr = pipe.route("开车窗，把空调打开")
    executed = [a.function for a in rr.plan.actions if a.status == "executed"]
    assert "set_ac_power" in executed
