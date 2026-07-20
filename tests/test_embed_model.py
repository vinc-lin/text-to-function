import pytest


@pytest.mark.model
def test_transformers_embedder_ranks_correctly():
    """Real Qwen3-Embedding: a temperature query must sit closer to the
    set_temperature prototype than to an unrelated one, and MRL dim is honored."""
    from t2f.embed import TransformersEmbedder
    e = TransformersEmbedder(mrl_dim=512)
    assert e.dim == 512
    protos = e.encode(["设置空调温度到指定摄氏度", "导航回家"], is_query=False)
    q = e.encode(["把副驾空调调到22度"], is_query=True)[0]
    import numpy as np
    assert np.allclose(np.linalg.norm(q), 1.0, atol=1e-3)
    assert float(q @ protos[0]) > float(q @ protos[1])
