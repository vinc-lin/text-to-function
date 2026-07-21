from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from .gate import Thresholds
from .score import DEFAULT_WEIGHTS


@dataclass
class Config:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    thresholds: Thresholds = field(default_factory=Thresholds)
    domain_keywords: dict = field(default_factory=dict)
    top_k: int = 5
    mrl_dim: int | None = None
    model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    llm: dict = field(default_factory=dict)
    classifier: dict = field(default_factory=dict)
    dialog: dict = field(default_factory=dict)

    @classmethod
    def default(cls) -> "Config":
        return cls()

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        t = d.get("thresholds", {})
        return cls(
            weights=d.get("weights", dict(DEFAULT_WEIGHTS)),
            thresholds=Thresholds(**t) if t else Thresholds(),
            domain_keywords=d.get("domain_keywords", {}),
            top_k=d.get("top_k", 5), mrl_dim=d.get("mrl_dim"),
            model_id=d.get("model_id", "Qwen/Qwen3-Embedding-0.6B"),
            llm=d.get("llm", {}), classifier=d.get("classifier", {}), dialog=d.get("dialog", {}))


def load_config(path: str | Path) -> Config:
    return Config.load(path)
