from t2f.embed import FakeEmbedder
from t2f.retrieve import PrototypeStore, Retriever
from t2f.types import FunctionCard

def _cards():
    return [
        FunctionCard("set_temperature", "climate", "设置空调温度",
                     utterances=["把空调调到25度", "温度设成22度"], aliases=["温度"]),
        FunctionCard("open_window", "window", "打开车窗",
                     utterances=["开车窗", "把窗户打开"], aliases=["车窗"]),
    ]

def test_retrieve_ranks_correct_function_first():
    emb = FakeEmbedder(256)
    store = PrototypeStore.build(_cards(), emb)
    r = Retriever(store)
    q = emb.encode(["把空调调到25度"], is_query=True)[0]
    cands = r.retrieve(q, top_k=2)
    assert cands[0].function == "set_temperature"
    assert cands[0].embedding_score >= cands[1].embedding_score
    assert cands[0].best_prototype != ""

def test_retrieve_top_k_limit():
    emb = FakeEmbedder(256)
    store = PrototypeStore.build(_cards(), emb)
    assert len(Retriever(store).retrieve(emb.encode(["开车窗"])[0], top_k=1)) == 1
