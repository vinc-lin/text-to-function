from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Classifier(ABC):
    @abstractmethod
    def fit(self, texts: list[str], labels: list[str]) -> "Classifier": ...
    @abstractmethod
    def predict_topk(self, text: str, k: int = 3) -> list[tuple[str, float]]: ...
    @abstractmethod
    def save(self, path: str) -> None: ...
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Classifier": ...


def _topk(classes, probs, k):
    idx = np.argsort(probs)[::-1][:k]
    return [(str(classes[i]), float(probs[i])) for i in idx]


class CharNgramLRClassifier(Classifier):
    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                     n_features=2 ** 18, alternate_sign=False)
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, texts, labels):
        self.clf.fit(self.vec.transform(texts), labels)
        return self

    def predict_topk(self, text, k=3):
        probs = self.clf.predict_proba(self.vec.transform([text]))[0]
        return _topk(self.clf.classes_, probs, k)

    def save(self, path):
        import joblib
        joblib.dump(self.clf, path)

    @classmethod
    def load(cls, path):
        import joblib
        obj = cls()
        obj.clf = joblib.load(path)
        return obj


class EmbeddingLRClassifier(Classifier):
    def __init__(self, embedder=None):
        from sklearn.linear_model import LogisticRegression
        self.embedder = embedder
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, texts, labels):
        X = self.embedder.encode(texts, is_query=True)
        self.clf.fit(X, labels)
        return self

    def predict_topk(self, text, k=3):
        X = self.embedder.encode([text], is_query=True)
        probs = self.clf.predict_proba(X)[0]
        return _topk(self.clf.classes_, probs, k)

    def save(self, path):
        import joblib
        joblib.dump(self.clf, path)

    @classmethod
    def load(cls, path, embedder=None):
        import joblib
        obj = cls(embedder)
        obj.clf = joblib.load(path)
        return obj
