from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .types import FunctionCard, Candidate
from .embed import Embedder


def _prototype_texts(card: FunctionCard) -> list[str]:
    texts = [card.description, *card.aliases, *card.utterances]
    return [t for t in texts if t]


@dataclass
class PrototypeStore:
    matrix: np.ndarray            # (P, dim), L2-normalized
    functions: list[str]          # length P, function name per row
    prototypes: list[str]         # length P, source text per row

    @classmethod
    def build(cls, cards: list[FunctionCard], embedder: Embedder) -> "PrototypeStore":
        texts, funcs = [], []
        for c in cards:
            for t in _prototype_texts(c):
                texts.append(t)
                funcs.append(c.name)
        matrix = embedder.encode(texts, is_query=False)
        return cls(matrix=matrix, functions=funcs, prototypes=texts)


class Retriever:
    def __init__(self, store: PrototypeStore):
        self.store = store

    def retrieve(self, query_vec: np.ndarray, top_k: int = 5) -> list[Candidate]:
        sims = self.store.matrix @ query_vec        # (P,)
        best: dict[str, tuple[float, str]] = {}
        for fn, proto, s in zip(self.store.functions, self.store.prototypes, sims):
            s = float(s)
            if fn not in best or s > best[fn][0]:
                best[fn] = (s, proto)
        cands = [Candidate(function=fn, score=sc, embedding_score=sc, best_prototype=proto)
                 for fn, (sc, proto) in best.items()]
        cands.sort(key=lambda c: c.score, reverse=True)
        return cands[:top_k]
