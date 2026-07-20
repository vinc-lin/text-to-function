from pathlib import Path
import pytest
from t2f.cards import load_catalog, CatalogError

FIX = Path(__file__).parent / "fixtures" / "catalog"

def test_load_catalog_parses_cards():
    cards = load_catalog(FIX)
    names = {c.name for c in cards}
    assert {"set_temperature", "set_fan_speed"} <= names
    st = next(c for c in cards if c.name == "set_temperature")
    assert st.domain == "climate"
    assert st.param("temperature").minimum == 16
    assert st.param("position").enum == ["driver", "passenger", "rear", "all"]
    assert "把空调调到25度" in st.utterances

def test_load_catalog_rejects_duplicate_names(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "domain: x\nfunctions:\n- {name: dup, description: d}\n- {name: dup, description: d}\n",
        encoding="utf-8")
    with pytest.raises(CatalogError):
        load_catalog(tmp_path)

def test_load_catalog_rejects_bad_enum_default(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "domain: x\nfunctions:\n- name: f\n  description: d\n"
        "  params: [{name: p, type: enum}]\n", encoding="utf-8")
    with pytest.raises(CatalogError):  # enum type requires an 'enum' list
        load_catalog(tmp_path)
