from __future__ import annotations
from pathlib import Path
import yaml
from .types import FunctionCard, ParamSpec

VALID_TYPES = {"number", "integer", "string", "boolean", "enum"}


class CatalogError(ValueError):
    pass


def _parse_param(d: dict) -> ParamSpec:
    if "name" not in d or "type" not in d:
        raise CatalogError(f"param missing name/type: {d}")
    if d["type"] not in VALID_TYPES:
        raise CatalogError(f"bad param type {d['type']}")
    if d["type"] == "enum" and not d.get("enum"):
        raise CatalogError(f"enum param {d['name']} needs 'enum' list")
    return ParamSpec(
        name=d["name"], type=d["type"], required=bool(d.get("required", False)),
        enum=d.get("enum"), minimum=d.get("minimum"), maximum=d.get("maximum"),
        unit=d.get("unit"), description=d.get("description", ""))


def _parse_card(d: dict, domain: str) -> FunctionCard:
    if "name" not in d or "description" not in d:
        raise CatalogError(f"card missing name/description: {d}")
    return FunctionCard(
        name=d["name"], domain=domain, description=d["description"],
        params=[_parse_param(p) for p in d.get("params", [])],
        aliases=list(d.get("aliases", [])),
        utterances=list(d.get("utterances", [])),
        hard_negatives=list(d.get("hard_negatives", [])),
        response_template=d.get("response_template", ""))


def load_ood_prototypes(path: str | Path) -> list[str]:
    """Load out-of-domain/chitchat prototype utterances (one per line; '#' comments and blanks skipped)."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def load_catalog(path: str | Path) -> list[FunctionCard]:
    path = Path(path)
    cards: list[FunctionCard] = []
    for f in sorted(path.glob("*.yaml")):
        doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        domain = doc.get("domain", f.stem)
        for cd in doc.get("functions", []):
            cards.append(_parse_card(cd, domain))
    seen: set[str] = set()
    for c in cards:
        if c.name in seen:
            raise CatalogError(f"duplicate function name: {c.name}")
        seen.add(c.name)
    if not cards:
        raise CatalogError(f"no cards found under {path}")
    return cards
