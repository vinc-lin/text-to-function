"""One factory assembles the product, so the session and the eval harness cannot drift."""
from pathlib import Path

from t2f.build import build_pipeline
from t2f.cards import load_catalog
from t2f.config import Config
from t2f.embed import FakeEmbedder
from t2f.gate import PERMISSIVE
from t2f.execute import MockExecutor
from t2f.llm.client import FakeLLMClient
from t2f.pipeline import NullMediumResolver, LLMResolver
from t2f.types import LLMResult, ToolCall

FIX = Path(__file__).parent / "fixtures" / "catalog"
CARDS = load_catalog(FIX)


def test_deterministic_pipeline_has_no_llm():
    pipe = build_pipeline(CARDS, FakeEmbedder(256), Config.default())
    assert pipe.llm_client is None
    assert isinstance(pipe.resolver.medium_resolver, NullMediumResolver)


def test_llm_pipeline_wires_both_the_medium_resolver_and_the_plan_path():
    """pipe.llm_client drives the per-span plan path; medium_resolver drives the legacy one.
    Setting only one silently disables half the LLM's job."""
    client = FakeLLMClient(default=LLMResult(tool_call=ToolCall("open_window", {"is_open": True})))
    pipe = build_pipeline(CARDS, FakeEmbedder(256), Config.default(), llm_client=client)
    assert pipe.llm_client is client
    assert isinstance(pipe.resolver.medium_resolver, LLMResolver)


def test_executor_is_injectable():
    ex = MockExecutor()
    pipe = build_pipeline(CARDS, FakeEmbedder(256), Config.default(), executor=ex)
    assert pipe.executor is ex


def test_thresholds_override_the_config():
    loose = PERMISSIVE
    pipe = build_pipeline(CARDS, FakeEmbedder(256), Config.default(), thresholds=loose)
    assert pipe.gate.t is loose


def test_it_routes_end_to_end():
    pipe = build_pipeline(CARDS, FakeEmbedder(256), Config.default(),
                          thresholds=PERMISSIVE)
    assert pipe.route("把空调调到25度").reply == "已将当前区域温度设置为25°C。"
