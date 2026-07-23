from __future__ import annotations
import time
from .types import (RouteResult, ClauseResult, Band, ValidationError,
                    SpanRole, ActionPlan, PlannedAction, RelativeSpec)
from .normalize import normalize
from .segment import split, segment
from .lexical import extract_features
from .retrieve import PrototypeStore, Retriever
from .params.extract import ParameterExtractor
from .validate import validate_tool_call
from .respond import render_response, build_clarification, build_low_confidence_clarification
from .execute import MockExecutor
from .llm.schema import REJECT_NAME
from .state import VehicleState, primary_numeric_param
from .plan import PlanExecutor


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
        self.state = VehicleState()
        self.llm_client = None  # set by arms for the LLM plan path (Task 9)
        self.extractor = getattr(self.resolver, "extractor", None) or ParameterExtractor()
        self.executor = getattr(self.resolver, "executor", None) or MockExecutor()

    def route(self, utterance: str) -> RouteResult:
        norm = normalize(utterance)
        spans = segment(norm, self.cards, self.config.domain_keywords)
        action_spans = [s for s in spans if s.role == SpanRole.ACTION]
        context_spans = [s for s in spans if s.role == SpanRole.CONTEXT]
        # Plan path ONLY for genuine multi-intent, or a single action that has context to strip.
        # Everything else -> legacy (unchanged single-intent semantics). CRITICAL: when the filter
        # finds NO action span (navigation/media-play shapes lack an open/close/set cue and classify
        # as CONTEXT), fall back to legacy so the embedding router still handles it -- never dropped.
        if len(action_spans) >= 2 or (action_spans and context_spans):
            return self._route_plan(utterance, action_spans)
        return self._route_legacy(utterance)

    def _route_legacy(self, utterance: str) -> RouteResult:
        results = [self._route_one(c) for c in split(normalize(utterance))]
        return RouteResult(utterance=utterance, clauses=results)

    def _decide(self, clause: str):
        """Retrieve -> (classifier) -> score -> gate. Returns (decision, feats). NO execution."""
        qv = self.embedder.encode([clause], is_query=True)[0]
        cands = self.retriever.retrieve(qv, top_k=self.config.top_k)
        classifier_probs = None
        if self.classifier_source is not None:
            cands, classifier_probs = self.classifier_source.augment(cands, clause)
        feats = extract_features(clause)
        cands = self.scorer.rescore(clause, feats, cands, self.cards_by_name, classifier_probs=classifier_probs)
        return self.gate.decide(cands, feats, self.cards_by_name), feats

    def _route_one(self, clause: str) -> ClauseResult:
        """Legacy single-clause path: decide + resolve (resolver executes HIGH band). Unchanged."""
        t0 = time.perf_counter()
        decision, feats = self._decide(clause)
        cr = self.resolver.resolve(clause, feats, decision)
        cr.latency_ms = (time.perf_counter() - t0) * 1000.0
        return cr

    def _span_candidates(self, decision, rel):
        """Candidate cards for a span, best-first. A relative span (再开一点) can only be applied to
        a function with a numeric param to adjust, so restrict to those when possible; otherwise the
        relative overlay would have nothing to resolve against."""
        cards = [self.cards_by_name[c.function] for c in decision.candidates
                 if c.function in self.cards_by_name]
        if rel is not None:
            numeric = [c for c in cards if primary_numeric_param(c) is not None]
            if numeric:
                cards = numeric
        return cards

    @staticmethod
    def _relative_spec(feats):
        """A span is RELATIVE only with an increase/decrease cue AND an amount AND no explicit
        absolute value. If the user stated a value (调高到26度 / 亮度调到30% / 开到一半), it is
        ABSOLUTE — the StateResolver must not overwrite it with a state-derived guess."""
        if feats.operation in ("increase", "decrease") and feats.amount \
                and not (feats.percentages or feats.temperatures or feats.levels):
            return RelativeSpec(feats.operation, feats.amount)
        return None

    def _planned_from_span(self, span_text, decision, feats) -> PlannedAction:
        rel = self._relative_spec(feats)
        cands = self._span_candidates(decision, rel)
        fn = cands[0].name if (rel is not None and cands) else decision.chosen
        params = self.extractor.extract(span_text, feats, self.cards_by_name[fn])[0] if fn else {}
        return PlannedAction(span=span_text, function=fn, parameters=dict(params), relative=rel)

    def _route_plan(self, utterance: str, action_spans) -> RouteResult:
        t0 = time.perf_counter()
        decided = [(s, *self._decide(s.text)) for s in action_spans]            # (span, decision, feats)
        clause_results = [ClauseResult(clause=s.text, decision=d) for s, d, _ in decided]
        all_high = all(d.band == Band.HIGH for _, d, _ in decided)

        if not all_high and self.llm_client is not None:
            planned = self._llm_plan(utterance, action_spans, clause_results)   # Task 9
            source = "llm"
        else:
            planned = [self._planned_from_span(s.text, d, f)
                       for s, d, f in decided if d.band == Band.HIGH and d.chosen]
            source = "deterministic"

        plan = ActionPlan(actions=planned, source=source)
        executed, clar = PlanExecutor(self.cards_by_name, self.state, self.executor,
                                      self.config.relative_steps).finalize(plan)

        by_span = {a.span: a for a in plan.actions}
        for (s, d, _), cr in zip(decided, clause_results):
            a = by_span.get(s.text)
            if a is not None and a.status == "executed":
                cr.tool_call = a.tool_call
                cr.response = render_response(self.cards_by_name[a.function], a.tool_call)
                cr.needs_llm = (source == "llm")
            elif a is not None:
                cr.clarification = clar
                cr.needs_llm = (source == "llm")
            else:
                cr.needs_llm = (d.band == Band.MEDIUM)
                if d.band == Band.LOW:
                    cr.clarification = build_low_confidence_clarification()
        total = (time.perf_counter() - t0) * 1000.0
        for cr in clause_results:
            cr.latency_ms = total / max(len(clause_results), 1)
        return RouteResult(utterance=utterance, clauses=clause_results, plan=plan)

    def _llm_plan(self, utterance, action_spans, clause_results):
        """Per-span 'confirm-or-reject retrieval top-1'. Qwen3-0.6B is unreliable at multi-way
        function selection (it hallucinates functions/params), but retrieval top-1 is strong, so we
        offer the LLM only that single top-1 (plus the __reject__ escape): it fills params or
        abstains, and cannot substitute a different function. Abstention -> function None -> the
        barrier defers/rejects the span (context/OOD safety preserved)."""
        planned = []
        for s, cr in zip(action_spans, clause_results):
            feats = extract_features(s.text)
            rel = self._relative_spec(feats)
            cands = self._span_candidates(cr.decision, rel)
            if not cands:
                planned.append(PlannedAction(span=s.text, function=None, relative=rel))
                continue
            top = cands[0]
            extracted = self.extractor.extract(s.text, feats, top)[0]
            res = self.llm_client.complete_tool_call(s.text, [top], extracted)
            if res.tool_call is not None and res.clarification != REJECT_NAME:
                planned.append(PlannedAction(span=s.text, function=res.tool_call.name,
                                             parameters=dict(res.tool_call.parameters), relative=rel))
            else:  # LLM abstained or produced nothing -> defer (never execute an unconfirmed span)
                planned.append(PlannedAction(span=s.text, function=None, relative=rel))
        return planned
