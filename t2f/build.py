"""Assemble the product.

The one place that wires a Pipeline together. Both the interactive session and the eval
harness's arms C and C_llm use it, so the thing a person tries by hand and the thing the
metrics describe cannot drift apart. Arms `baseline` and `D` stay in eval/arms.py: this
factory builds the product, the eval package builds experiments.
"""
from __future__ import annotations
from typing import Optional

from .config import Config
from .gate import ConfidenceGate, Thresholds
from .pipeline import Pipeline, DeterministicResolver, LLMResolver
from .score import Scorer
from .types import FunctionCard


def build_pipeline(cards: list[FunctionCard], embedder, config: Config, *,
                   llm_client=None, executor=None, ood_texts: Optional[list] = None,
                   thresholds: Optional[Thresholds] = None) -> Pipeline:
    """`llm_client` attaches the fallback; `executor` swaps the vehicle adapter; `thresholds`
    overrides the shipped gate. Every argument defaults to the shipped configuration."""
    medium = None
    if llm_client is not None:
        medium = LLMResolver(llm_client,
                             max_candidates=config.llm.get("max_candidates", 3),
                             max_retries=config.llm.get("max_retries", 1))
    resolver = DeterministicResolver({c.name: c for c in cards},
                                     executor=executor, medium_resolver=medium)
    pipe = Pipeline(cards, embedder, Scorer(config.weights, config.domain_keywords),
                    ConfidenceGate(thresholds or config.thresholds), config,
                    resolver=resolver, ood_texts=ood_texts)
    if llm_client is not None:
        pipe.llm_client = llm_client      # the per-span plan path reads this, not the resolver
    return pipe
