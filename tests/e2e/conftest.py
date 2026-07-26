"""Deterministic end-to-end harness: the REAL Pipeline.route() over a 3-card fixture
catalog with FakeEmbedder. No model, no network, no GPU.

Thresholds are loosened so the hashed-n-gram FakeEmbedder reaches the HIGH band on the
fixture utterances; this mirrors tests/test_reply_e2e.py, which established the pattern.
"""
from __future__ import annotations
from pathlib import Path
import pytest

from t2f.cards import load_catalog
from t2f.config import Config
from t2f.embed import FakeEmbedder
from t2f.gate import ConfidenceGate, Thresholds
from t2f.pipeline import Pipeline, DeterministicResolver, LLMResolver
from t2f.score import Scorer

from .doubles import RecordingExecutor

FIXTURE_CATALOG = Path(__file__).parent.parent / "fixtures" / "catalog"


def build_pipeline(executor=None, llm_client=None, state=None, thresholds=None):
    """Return (pipeline, executor). `executor` defaults to a fresh RecordingExecutor."""
    cards = load_catalog(FIXTURE_CATALOG)
    cfg = Config.default()
    cfg.thresholds = thresholds or Thresholds(high_top1=0.2, high_margin=0.0, low_top1=0.05)
    executor = executor if executor is not None else RecordingExecutor()

    medium = LLMResolver(llm_client) if llm_client is not None else None
    resolver = DeterministicResolver({c.name: c for c in cards},
                                     executor=executor, medium_resolver=medium)
    pipe = Pipeline(cards, FakeEmbedder(256), Scorer(cfg.weights, cfg.domain_keywords),
                    ConfidenceGate(cfg.thresholds), cfg, resolver=resolver)
    if llm_client is not None:
        pipe.llm_client = llm_client          # enables the per-span plan path
    for key, value in (state or {}).items():
        pipe.state.set(key, value, layer="confirmed")
    return pipe, executor


@pytest.fixture
def pipeline():
    """(pipeline, executor) with a RecordingExecutor — the common case."""
    return build_pipeline()
