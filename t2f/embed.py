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


class TransformersEmbedder(Embedder):
    """Qwen3-Embedding via the transformers library with last-token pooling.

    Uses the GPU when available. Neutralizes a broken torchvision install (whose C++
    ops fail to register) so transformers' text path imports cleanly, and bypasses
    sentence-transformers entirely. This is the FP reference backend; on-device it is
    replaced by the GGUF/llama.cpp path (Spec 3), hence the shared Embedder interface.
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-Embedding-0.6B",
                 instruction: str = DEFAULT_QUERY_INSTRUCTION, mrl_dim: int | None = None,
                 max_length: int = 256, batch_size: int = 64, device: str | None = None,
                 dtype: str = "float16"):
        import sys as _sys
        _sys.modules.setdefault("torchvision", None)  # dodge broken torchvision::nms
        import torch
        from transformers import AutoTokenizer, AutoModel
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        td = torch.float16 if (dtype == "float16" and self.device == "cuda") else torch.float32
        self._tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self._model = AutoModel.from_pretrained(model_id, torch_dtype=td).to(self.device).eval()
        self.instruction = instruction
        self.mrl_dim = mrl_dim
        self.max_length = max_length
        self.batch_size = batch_size
        self.dim = mrl_dim or self._model.config.hidden_size

    def _pool(self, h, mask):
        # last-token pooling; handles left- or right-padding
        if bool((mask[:, -1].sum() == mask.shape[0]).item()):
            return h[:, -1]
        lengths = mask.sum(dim=1) - 1
        return h[self._torch.arange(h.shape[0], device=h.device), lengths]

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        torch = self._torch
        if is_query:
            texts = [self.instruction + t for t in texts]
        chunks = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self._tok(batch, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt").to(self.device)
                h = self._model(**enc).last_hidden_state
                emb = self._pool(h, enc["attention_mask"])
                chunks.append(emb.float().cpu().numpy())
        v = np.vstack(chunks)
        if self.mrl_dim:
            v = v[:, :self.mrl_dim]
        return _l2(v.astype(np.float32))


class GgufEmbedder(Embedder):  # Spec 3
    def __init__(self, *a, **k):
        raise NotImplementedError("GGUF/llama.cpp embedder is Spec 3")

    def encode(self, texts, is_query=False):
        raise NotImplementedError
