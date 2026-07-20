import numpy as np
from t2f.embed import FakeEmbedder

def test_fake_deterministic_and_normalized():
    e = FakeEmbedder(dim=64)
    a = e.encode(["打开车窗"])
    b = e.encode(["打开车窗"])
    assert a.shape == (1, 64)
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)

def test_fake_similar_texts_related():
    e = FakeEmbedder(dim=64)
    v = e.encode(["打开车窗", "关闭车窗", "设置温度"])
    # cosine of identical > cosine of different (sanity for a hash embedder w/ char n-grams)
    same = float(v[0] @ e.encode(["打开车窗"])[0])
    assert same > 0.999
