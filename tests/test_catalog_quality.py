# tests/test_catalog_quality.py
from pathlib import Path
from t2f.cards import load_catalog

CATALOG = Path("data/catalog")


def test_catalog_size_and_quality():
    cards = load_catalog(CATALOG)
    assert len(cards) >= 80, f"only {len(cards)} functions"
    domains = {c.domain for c in cards}
    assert len(domains) >= 8
    for c in cards:
        assert len(c.utterances) >= 6, f"{c.name} has too few utterances"
        assert c.response_template, f"{c.name} missing response_template"
        assert c.description
    # every enum param has a non-empty enum list (loader guarantees, re-assert)
    for c in cards:
        for p in c.params:
            if p.type == "enum":
                assert p.enum
