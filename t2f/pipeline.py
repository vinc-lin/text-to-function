from __future__ import annotations
import time
from .types import RouteResult, ClauseResult, Band, ValidationError
from .normalize import normalize
from .segment import split
from .lexical import extract_features
from .retrieve import PrototypeStore, Retriever
from .params.extract import ParameterExtractor
from .validate import validate_tool_call
from .respond import render_response, build_clarification, build_low_confidence_clarification
from .execute import MockExecutor
from .llm.schema import REJECT_NAME


class NullMediumResolver:
    """Spec-1 behavior: mark needs_llm, attempt validation for the ceiling metric, do not execute."""
    def resolve(self, clause, features, decision, cards_by_name, extractor, executor) -> ClauseResult:
        card = cards_by_name[decision.chosen]
        params, _ = extractor.extract(clause, features, card)
        cand_names = [c.function for c in decision.candidates]
        tc, errs = validate_tool_call(decision.chosen, params, cards_by_name, cand_names)
        return ClauseResult(clause=clause, decision=decision, tool_call=tc,
                            validation_errors=errs, needs_llm=True)


class LLMResolver:
    def __init__(self, llm_client, max_candidates: int = 3, max_retries: int = 1):
        self.client = llm_client
        self.max_candidates = max_candidates
        self.max_retries = max_retries

    def resolve(self, clause, features, decision, cards_by_name, extractor=None, executor=None) -> ClauseResult:
        extractor = extractor or ParameterExtractor()
        cand_names = [c.function for c in decision.candidates]
        cards = [cards_by_name[n] for n in cand_names[:self.max_candidates] if n in cards_by_name]
        # validate against exactly the set offered to the LLM (defense-in-depth: never execute a
        # name that wasn't shown to the decoder, even if a lenient backend emits one)
        offered_names = [c.name for c in cards]
        chosen_card = cards_by_name[decision.chosen]
        extracted, _ = extractor.extract(clause, features, chosen_card)
        attempts = 0
        while True:
            res = self.client.complete_tool_call(clause, cards, extracted)
            if res.clarification == REJECT_NAME:  # LLM declined: out-of-scope, do NOT execute
                return ClauseResult(clause=clause, decision=decision,
                                    clarification=build_low_confidence_clarification(), needs_llm=True)
            if res.clarification and not res.tool_call:
                clar = build_clarification(chosen_card, chosen_card.required_params)
                return ClauseResult(clause=clause, decision=decision, clarification=clar, needs_llm=True)
            if res.tool_call is None:
                if attempts < self.max_retries:
                    attempts += 1
                    continue
                return ClauseResult(clause=clause, decision=decision, needs_llm=True,
                                    validation_errors=[ValidationError("llm_no_toolcall", res.error or "no tool_call")])
            tc, errs = validate_tool_call(res.tool_call.name, res.tool_call.parameters, cards_by_name, offered_names)
            if tc is not None:
                card = cards_by_name[tc.name]
                if executor is not None:
                    executor.execute(tc)
                return ClauseResult(clause=clause, decision=decision, tool_call=tc,
                                    response=render_response(card, tc), needs_llm=True)
            # invalid: retry once, then clarify/reject
            if attempts < self.max_retries:
                attempts += 1
                continue
            # missing-required -> clarification; else hard reject (never execute)
            if any(e.code == "missing_required" for e in errs):
                miss = [e.message.split()[-1] for e in errs if e.code == "missing_required"]
                clar = build_clarification(cards_by_name[res.tool_call.name] if res.tool_call.name in cards_by_name else chosen_card, miss)
                return ClauseResult(clause=clause, decision=decision, clarification=clar,
                                    validation_errors=errs, needs_llm=True)
            return ClauseResult(clause=clause, decision=decision, validation_errors=errs, needs_llm=True)


class DeterministicResolver:
    def __init__(self, cards_by_name, executor=None, medium_resolver=None):
        self.cards = cards_by_name
        self.executor = executor or MockExecutor()
        self.extractor = ParameterExtractor()
        self.medium_resolver = medium_resolver or NullMediumResolver()

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
        if decision.band == Band.MEDIUM:
            return self.medium_resolver.resolve(clause, features, decision, self.cards,
                                                self.extractor, self.executor)
        tc, errs = validate_tool_call(decision.chosen, params, self.cards, cand_names)
        if tc is None:
            return ClauseResult(clause=clause, decision=decision, validation_errors=errs)
        self.executor.execute(tc)
        return ClauseResult(clause=clause, decision=decision, tool_call=tc,
                            response=render_response(card, tc))


class Pipeline:
    def __init__(self, cards, embedder, scorer, gate, config, resolver=None,
                 medium_resolver=None, classifier_source=None, ood_texts=None):
        self.cards = cards
        self.cards_by_name = {c.name: c for c in cards}
        self.embedder = embedder
        self.scorer = scorer
        self.gate = gate
        self.config = config
        self.retriever = Retriever(PrototypeStore.build(cards, embedder, ood_texts=ood_texts))
        self.classifier_source = classifier_source
        self.resolver = resolver or DeterministicResolver(self.cards_by_name, medium_resolver=medium_resolver)

    def route(self, utterance: str) -> RouteResult:
        clauses = split(normalize(utterance))
        results = []
        for clause in clauses:
            t0 = time.perf_counter()
            qv = self.embedder.encode([clause], is_query=True)[0]
            cands = self.retriever.retrieve(qv, top_k=self.config.top_k)
            classifier_probs = None
            if self.classifier_source is not None:
                cands, classifier_probs = self.classifier_source.augment(cands, clause)
            feats = extract_features(clause)
            cands = self.scorer.rescore(clause, feats, cands, self.cards_by_name, classifier_probs=classifier_probs)
            decision = self.gate.decide(cands, feats, self.cards_by_name)
            cr = self.resolver.resolve(clause, feats, decision)
            cr.latency_ms = (time.perf_counter() - t0) * 1000.0
            results.append(cr)
        return RouteResult(utterance=utterance, clauses=results)
