from __future__ import annotations
import time
from .types import RouteResult, ClauseResult, Band
from .normalize import normalize
from .segment import split
from .lexical import extract_features
from .retrieve import PrototypeStore, Retriever
from .params.extract import ParameterExtractor
from .validate import validate_tool_call
from .respond import render_response, build_clarification, build_low_confidence_clarification
from .execute import MockExecutor


class DeterministicResolver:
    def __init__(self, cards_by_name, executor=None):
        self.cards = cards_by_name
        self.executor = executor or MockExecutor()
        self.extractor = ParameterExtractor()

    def resolve(self, clause, features, decision) -> ClauseResult:
        cand_names = [c.function for c in decision.candidates]
        if decision.band == Band.LOW or decision.chosen is None:
            return ClauseResult(clause=clause, decision=decision,
                                clarification=build_low_confidence_clarification(),
                                needs_llm=False)
        card = self.cards[decision.chosen]
        params, missing = self.extractor.extract(clause, features, card)
        if missing and decision.band == Band.HIGH:
            clar = build_clarification(card, missing)
            return ClauseResult(clause=clause, decision=decision, clarification=clar)
        tc, errs = validate_tool_call(decision.chosen, params, self.cards, cand_names)
        if decision.band == Band.MEDIUM:
            return ClauseResult(clause=clause, decision=decision, tool_call=tc,
                                validation_errors=errs, needs_llm=True)
        if tc is None:
            return ClauseResult(clause=clause, decision=decision, validation_errors=errs)
        self.executor.execute(tc)
        return ClauseResult(clause=clause, decision=decision, tool_call=tc,
                            response=render_response(card, tc))


class Pipeline:
    def __init__(self, cards, embedder, scorer, gate, config, resolver=None):
        self.cards = cards
        self.cards_by_name = {c.name: c for c in cards}
        self.embedder = embedder
        self.scorer = scorer
        self.gate = gate
        self.config = config
        self.retriever = Retriever(PrototypeStore.build(cards, embedder))
        self.resolver = resolver or DeterministicResolver(self.cards_by_name)

    def route(self, utterance: str) -> RouteResult:
        clauses = split(normalize(utterance))
        results = []
        for clause in clauses:
            t0 = time.perf_counter()
            qv = self.embedder.encode([clause], is_query=True)[0]
            cands = self.retriever.retrieve(qv, top_k=self.config.top_k)
            feats = extract_features(clause)
            cands = self.scorer.rescore(clause, feats, cands, self.cards_by_name)
            decision = self.gate.decide(cands, feats, self.cards_by_name)
            cr = self.resolver.resolve(clause, feats, decision)
            cr.latency_ms = (time.perf_counter() - t0) * 1000.0
            results.append(cr)
        return RouteResult(utterance=utterance, clauses=results)
