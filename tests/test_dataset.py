from pathlib import Path

from eval.dataset import load_dataset, validate_against_catalog
from t2f.cards import load_catalog


def test_gold_rows_wellformed_and_reference_real_functions():
    rows = load_dataset("data/eval/gold.jsonl")
    assert len(rows) >= 300
    names = {c.name for c in load_catalog("data/catalog")}
    problems = validate_against_catalog(rows, names)
    assert problems == [], problems[:5]
    types = {r["type"] for r in rows}
    assert {"single", "multi_intent", "ood"} <= types
    assert sum(r["type"] == "multi_intent" for r in rows) >= 40
    assert sum(r["type"] == "ood" for r in rows) >= 40


def test_split_present():
    rows = load_dataset("data/eval/gold.jsonl")
    splits = {r.get("split") for r in rows}
    assert "dev" in splits and "test" in splits


def test_every_function_has_two_single_rows():
    rows = load_dataset("data/eval/gold.jsonl")
    names = {c.name for c in load_catalog("data/catalog")}
    counts = {n: 0 for n in names}
    for r in rows:
        if r["type"] == "single" and len(r.get("expected_functions", [])) == 1:
            fn = r["expected_functions"][0]
            if fn in counts:
                counts[fn] += 1
    missing = {n: c for n, c in counts.items() if c < 2}
    assert not missing, missing


def test_ambiguous_coverage():
    rows = load_dataset("data/eval/gold.jsonl")
    assert sum(r["type"] == "ambiguous" for r in rows) >= 20


def test_split_ratio_roughly_forty_percent_dev():
    rows = load_dataset("data/eval/gold.jsonl")
    dev = sum(r.get("split") == "dev" for r in rows)
    frac = dev / len(rows)
    assert 0.30 <= frac <= 0.50, frac


def test_silver_present_and_valid():
    rows = load_dataset("data/eval/silver.jsonl")
    assert len(rows) >= 1000
    names = {c.name for c in load_catalog("data/catalog")}
    assert validate_against_catalog(rows, names) == []
