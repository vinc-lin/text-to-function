from t2f.types import Candidate
from t2f.classify.source import ClassifierCandidateSource

class _StubClf:
    def predict_topk(self, text, k=3):
        return [("open_window", 0.7), ("set_temperature", 0.2), ("lock_doors", 0.1)][:k]

def test_augment_unions_and_returns_probs():
    cands = [Candidate("set_temperature", 0.6, embedding_score=0.6)]
    out, probs = ClassifierCandidateSource(_StubClf(), k=3).augment(cands, "开窗")
    names = {c.function for c in out}
    assert {"set_temperature", "open_window", "lock_doors"} <= names   # union, nothing removed
    assert probs["open_window"] == 0.7
    assert next(c for c in out if c.function == "lock_doors").embedding_score == 0.0
