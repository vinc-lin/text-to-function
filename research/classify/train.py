from __future__ import annotations
import argparse
from pathlib import Path
from eval.dataset import load_dataset
from research.classify.classifiers import CharNgramLRClassifier, EmbeddingLRClassifier


def build_training_pairs(dataset_paths: list[str], splits: set[str] | None = None):
    pairs = []
    for p in dataset_paths:
        for r in load_dataset(p):
            if splits is not None and r.get("split") not in splits:
                continue
            if r.get("type") in ("single", "ambiguous") and len(r.get("expected_functions", [])) == 1:
                pairs.append((r["utterance"], r["expected_functions"][0]))
    return pairs


def train(silver_path, gold_path, out_dir="models", embedder=None) -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pairs = build_training_pairs([silver_path]) + build_training_pairs([gold_path], splits={"dev"})
    texts = [t for t, _ in pairs]
    labels = [l for _, l in pairs]
    paths = {}
    cn = CharNgramLRClassifier().fit(texts, labels)
    cn.save(f"{out_dir}/clf_charngram.joblib"); paths["char_ngram"] = f"{out_dir}/clf_charngram.joblib"
    if embedder is not None:
        em = EmbeddingLRClassifier(embedder).fit(texts, labels)
        em.save(f"{out_dir}/clf_embedding.joblib"); paths["embedding"] = f"{out_dir}/clf_embedding.joblib"
    return {"n_train": len(texts), "classes": sorted(set(labels)), "paths": paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", default="data/eval/silver.jsonl")
    ap.add_argument("--gold", default="data/eval/gold.jsonl")
    ap.add_argument("--out", default="models")
    ap.add_argument("--embedding", action="store_true", help="also train the embedding classifier")
    a = ap.parse_args()
    emb = None
    if a.embedding:
        from t2f.embed import TransformersEmbedder
        emb = TransformersEmbedder(mrl_dim=512)
    print(train(a.silver, a.gold, a.out, emb))


if __name__ == "__main__":
    main()
