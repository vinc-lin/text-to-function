"""What arrives, and where it came from.

One shape for all three inputs, so provenance and timing are captured once at the edge rather
than three different ways (or, for voice, not at all).

There is deliberately NO `kind` field: the payload's type is the kind. A `kind` beside a
payload is two statements about one fact, and eventually they disagree.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Union


@dataclass(frozen=True)
class Utterance:
    text: str


@dataclass(frozen=True)
class Percept:
    key: str
    value: Any
    confidence: float
    ttl: float


@dataclass(frozen=True)
class SignalWrite:
    entity: str
    attribute: str
    value: Any


Payload = Union[Utterance, Percept, SignalWrite]


@dataclass(frozen=True)
class Input:
    source: str
    at: float
    payload: Payload

    def __post_init__(self):
        # Validated at construction, not at dispatch: an Input that exists is one that could
        # have happened, so nothing downstream needs to re-ask.
        from .sources import SOURCES
        src = SOURCES.get(self.source)
        if src is None:
            raise ValueError(f"{self.source!r} is not a declared source")
        if not isinstance(self.payload, src.accepts):
            raise ValueError(
                f"{self.source!r} produces {src.accepts.__name__}, "
                f"not {type(self.payload).__name__}")
