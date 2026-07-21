from t2f.classify.classifiers import CharNgramLRClassifier, EmbeddingLRClassifier
from t2f.embed import FakeEmbedder

TEXTS = ["把空调调到25度", "温度调高", "空调设成22度", "风速调到三档", "风大一点", "把风量开到最大",
         "打开车窗", "关闭车窗", "开一下窗户"]
LABELS = ["set_temperature", "set_temperature", "set_temperature", "set_fan_speed", "set_fan_speed",
          "set_fan_speed", "open_window", "open_window", "open_window"]

def test_charngram_predicts_seen_class():
    c = CharNgramLRClassifier(); c.fit(TEXTS, LABELS)
    top = c.predict_topk("空调调到26度", k=3)
    assert top[0][0] == "set_temperature"
    assert 0.0 <= top[0][1] <= 1.0 and len(top) == 3

def test_embedding_classifier_with_fake_embedder():
    c = EmbeddingLRClassifier(FakeEmbedder(256)); c.fit(TEXTS, LABELS)
    names = [fn for fn, _ in c.predict_topk("开窗户", k=3)]
    assert "open_window" in names

def test_save_load_roundtrip(tmp_path):
    c = CharNgramLRClassifier(); c.fit(TEXTS, LABELS)
    p = tmp_path / "c.joblib"; c.save(str(p))
    c2 = CharNgramLRClassifier.load(str(p))
    assert c2.predict_topk("温度调到20度")[0][0] == "set_temperature"
