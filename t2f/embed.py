from __future__ import annotations
import hashlib
from abc import ABC, abstractmethod
import numpy as np

DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Given a Chinese in-car voice command, retrieve the vehicle-control "
    "function it invokes.\nQuery: ")


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class Embedder(ABC):
    dim: int

    @abstractmethod
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


class FakeEmbedder(Embedder):
    """Deterministic char-3gram hashed bag-of-ngrams embedder. No torch, no network."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        grams = [text[i:i + 3] for i in range(max(1, len(text) - 2))] or [text]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0
        return v

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        return _l2(np.vstack([self._vec(t) for t in texts]))


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_id: str = "Qwen/Qwen3-Embedding-0.6B",
                 instruction: str = DEFAULT_QUERY_INSTRUCTION, mrl_dim: int | None = None):
        from sentence_transformers import SentenceTransformer  # lazy
        self._model = SentenceTransformer(model_id)
        self.instruction = instruction
        self.mrl_dim = mrl_dim
        self.dim = mrl_dim or self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if is_query:
            texts = [self.instruction + t for t in texts]
        v = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        if self.mrl_dim:
            v = v[:, :self.mrl_dim]
        return _l2(v.astype(np.float32))


class GgufEmbedder(Embedder):  # Spec 3
    def __init__(self, *a, **k):
        raise NotImplementedError("GGUF/llama.cpp embedder is Spec 3")

    def encode(self, texts, is_query=False):
        raise NotImplementedError
