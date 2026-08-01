"""Where an input can come from, and what it is allowed to produce.

A declared registry rather than a free string. Before this, `source` was decoration: every
observation defaulted to "cabin_cam" including vehicle-namespace ones, nothing validated it,
and only the display ever read it. Declaring what a source produces makes the field a claim
something can check.
"""
from __future__ import annotations
from dataclasses import dataclass

from .envelope import Percept, SignalWrite, Utterance


@dataclass(frozen=True)
class Source:
    name: str
    accepts: type
    # Re-stamps its held values when pumped. Only a continuous measurement has anything to
    # re-stamp -- an utterance is an event, not a level -- which test_only_a_signal_source_
    # may_publish enforces.
    publishes: bool = False


SOURCES: dict[str, Source] = {
    "mic":       Source("mic",       accepts=Utterance),
    "cabin_cam": Source("cabin_cam", accepts=Percept),
    "front_cam": Source("front_cam", accepts=Percept),
    "can0":      Source("can0",      accepts=SignalWrite, publishes=True),
}
